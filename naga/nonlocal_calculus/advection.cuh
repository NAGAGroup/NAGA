
// BSD 3-Clause License
//
// Copyright (c) 2023 Jack Myers
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "divergence.cuh"
#include <scalix/algorithm/elementwise_reduce.cuh>
#include <scalix/algorithm/transform.cuh>
#include <scalix/assign_array.cuh>

namespace naga::nonlocal_calculus {

template<class T>
class forward_euler {
  public:
    template<class U, class V>
    __host__ __device__ T operator()(U&& f, V&& df_dt) const {
        return f + dt_ * df_dt;
    }

    explicit forward_euler(const T& dt) : dt_(dt) {}

  private:
    T dt_;
};

template<class T, uint Dimensions>
class constant_velocity_field {
  public:
    using point_type = point_t<T, Dimensions>;

    template<class Velocity>
    static constant_velocity_field create(Velocity&& v) {
        constant_velocity_field map;
        for (uint i = 0; i < Dimensions; ++i) {
            map.velocity_[i] = v[i];
        }
        return map;
    }

    __host__ __device__ point_type operator[](const sclx::index_t& index
    ) const {
        point_type vector_value;
        for (uint i = 0; i < Dimensions; ++i) {
            vector_value[i] = velocity_[i];
        }
        return vector_value;
    }

  private:
    constant_velocity_field() = default;

    T velocity_[Dimensions];
};

template<class T, uint Dimensions>
class advection_operator {
  public:
    advection_operator() = default;

    friend class operator_builder<T, Dimensions>;

    template<class VelocityFieldMap>
    class divergence_field_map {
      public:
        using point_type = point_t<T, Dimensions>;
        friend class advection_operator;

        divergence_field_map(
            const VelocityFieldMap& velocity_field,
            const sclx::array<T, 1>& scalar_field,
            T centering_offset = T(0)
        )
            : velocity_field_(velocity_field),
              scalar_field_(scalar_field),
              centering_offset_(centering_offset) {}

        __host__ __device__ point_type operator[](const sclx::index_t& index
        ) const {
            point_type velocity = velocity_field_[index];
            T scalar            = scalar_field_[index];
            for (uint i = 0; i < Dimensions; ++i) {
                velocity[i] *= -(scalar - centering_offset_);
            }

            return velocity;
        }

        __host__ __device__ point_type
        operator[](const sclx::md_index_t<1>& index) const {
            return (*this)[index[0]];
        }

        __host__ __device__ size_t size() const {
            return scalar_field_.elements();
        }

      private:
        VelocityFieldMap velocity_field_;
        sclx::array<T, 1> scalar_field_;
        T centering_offset_;
    };

    static advection_operator create(const sclx::array<T, 2>& domain) {
        advection_operator op;
        op.divergence_op_
            = std::make_shared<divergence_operator<T, Dimensions>>(
                divergence_operator<T, Dimensions>::create(domain)
            );
        op.domain_size_ = domain.shape()[1];
        op.set_max_concurrent_threads(op.max_concurrent_threads_);
        return op;
    }

    void set_max_concurrent_threads(uint max_concurrent_threads) {
        std::lock_guard<std::mutex> lock(*mutex_);
        max_concurrent_threads_ = max_concurrent_threads;
        uint num_rk_df_dt_arrays = 4 * max_concurrent_threads;
        for (uint i = rk_df_dt_list_->size(); i < num_rk_df_dt_arrays; ++i) {
            rk_df_dt_list_->push_back(
                std::pair<std::shared_future<void>, sclx::array<T, 1>>{
                    std::shared_future<void>{},
                    sclx::array<T, 1>{domain_size_}}
            );
        }
    }

    [[nodiscard]] uint max_concurrent_threads() const {
        return max_concurrent_threads_;
    }

    template<class FieldMap>
    std::future<void> step_forward(
        FieldMap& velocity_field,
        sclx::array<T, 1>& f0,
        sclx::array<T, 1>& f,
        T dt,
        T centering_offset = T(0)
    ) {
        std::lock_guard<std::mutex> lock(*mutex_);
        sclx::array<T, 1> rk_df_dt_list[4];
        for (auto& df_dt : rk_df_dt_list) {
            auto [fut, array] = rk_df_dt_list_->front();
            if (fut.valid()) {
                fut.wait();
            }
            df_dt = std::move(array);
            rk_df_dt_list_->pop_front();
        }
        std::shared_future<void> ready_future;
        auto fut = std::async(std::launch::async, [=, &ready_future]() mutable {
            std::promise<void> ready_promise;
            ready_future = ready_promise.get_future();
            divergence_field_map<FieldMap> div_input_field{
                velocity_field,
                f0,
                centering_offset};

            divergence_op_->apply(div_input_field, rk_df_dt_list[0]);

            div_input_field.scalar_field_ = f;

            std::vector<std::future<void>> transform_futures;

            for (int i = 0; i < 3; ++i) {
                sclx::algorithm::elementwise_reduce(
                    forward_euler<T>{dt * runge_kutta_4::df_dt_weights[i]},
                    f,
                    f0,
                    rk_df_dt_list[i]
                );

                auto t_fut = sclx::algorithm::transform(
                    rk_df_dt_list[i],
                    rk_df_dt_list[i],
                    runge_kutta_4::summation_weights[i],
                    sclx::algorithm::multiplies<>{}
                );
                transform_futures.push_back(std::move(t_fut));

                divergence_op_->apply(div_input_field, rk_df_dt_list[i + 1]);
            }

            sclx::algorithm::transform(
                rk_df_dt_list[3],
                rk_df_dt_list[3],
                runge_kutta_4::summation_weights[3],
                sclx::algorithm::multiplies<>{}
            )
                .get();

            for (auto& t_fut : transform_futures) {
                t_fut.get();
            }

            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>{},
                rk_df_dt_list[0],
                rk_df_dt_list[0],
                rk_df_dt_list[1],
                rk_df_dt_list[2],
                rk_df_dt_list[3]
            );

            sclx::algorithm::elementwise_reduce(
                forward_euler<T>{dt},
                f,
                f0,
                rk_df_dt_list[0]
            );

            ready_promise.set_value();
        });
        for (auto& arr : rk_df_dt_list) {
            while (!ready_future.valid()) {
                std::this_thread::yield();
            }
            rk_df_dt_list_->emplace_back(ready_future, arr);
        }
        return fut;
    }

    const divergence_operator<T, Dimensions>& divergence_op() const {
        return *divergence_op_;
    }

    template<class Archive>
    void save(Archive& ar) const {
        ar(*divergence_op_);
        ar(domain_size_);
        ar(max_concurrent_threads_);
    }

    template<class Archive>
    void load(Archive& ar) {
        divergence_op_ = std::make_shared<divergence_operator<T, Dimensions>>();
        rk_df_dt_list_->clear();
        ar(*divergence_op_);
        ar(domain_size_);
        ar(max_concurrent_threads_);
        set_max_concurrent_threads(max_concurrent_threads_);
    }

  private:
    std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;
    using queue_type
        = std::deque<std::pair<std::shared_future<void>, sclx::array<T, 1>>>;
    std::shared_ptr<queue_type> rk_df_dt_list_ = std::make_shared<queue_type>();
    std::shared_ptr<std::mutex> mutex_         = std::make_shared<std::mutex>();
    size_t domain_size_;
    uint max_concurrent_threads_ = 4;

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[3] = {1. / 2., 1. / 2., 1.};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };
};

}  // namespace naga::nonlocal_calculus
