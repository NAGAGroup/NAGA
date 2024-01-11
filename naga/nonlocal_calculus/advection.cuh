
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

#include <Eigen/Eigen>

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

    using matrix_type
        = naga::linalg::matrix<T, naga::linalg::storage_type::sparse_csr>;
    using vector_type
        = naga::linalg::vector<T, naga::linalg::storage_type::dense>;
    using mat_mult_type
        = naga::linalg::matrix_mult<matrix_type, vector_type, vector_type>;

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
        return op;
    }

    class advection_task {
      public:
        template<class FieldMap>
        advection_task(
            std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_,
            FieldMap field_map,
            sclx::array<T, 1> f0,
            sclx::array<T, 1> f,
            T dt,
            T centering_offset,
            size_t boundary_idx_start = 0
        )
            : field_map_(std::make_shared<FieldMap>(std::move(field_map))),
              f0_(std::move(f0)),
              f_(std::move(f)),
              dt_(dt),
              centering_offset_(centering_offset),
              divergence_op_(divergence_op_),
              impl_(&advection_task::impl<FieldMap>),
              boundary_idx_start_(boundary_idx_start) {}

        void operator()(sclx::array<T, 1> (&rk_df_dt_list)[4]) {
            sclx::array<T, 1> sliced_rk_df_dt_list[4];
            for (int i = 0; i < 4; ++i) {
                if (rk_df_dt_list[i].elements() == f0_.elements()) {
                    sliced_rk_df_dt_list[i] = rk_df_dt_list[i];
                } else if (rk_df_dt_list[i].elements() > f0_.elements()) {
                    sliced_rk_df_dt_list[i] = sclx::array<T, 1>{
                        f0_.shape(),
                        rk_df_dt_list[i].data()
                    };
                    sliced_rk_df_dt_list[i].set_primary_devices();
                } else {
                    rk_df_dt_list[i]        = sclx::array<T, 1>{f0_.shape()};
                    sliced_rk_df_dt_list[i] = rk_df_dt_list[i];
                }
            }
            impl_(*this, sliced_rk_df_dt_list);
        }

        template<class FieldMap>
        static void
        impl(advection_task& task, sclx::array<T, 1> (&rk_df_dt_list)[4]) {
            auto& velocity_field
                = *(static_cast<FieldMap*>(task.field_map_.get()));
            auto& f0               = task.f0_;
            auto& f                = task.f_;
            auto& dt               = task.dt_;
            auto& centering_offset = task.centering_offset_;
            auto& divergence_op_   = task.divergence_op_;

            divergence_field_map<FieldMap> div_input_field{
                velocity_field,
                f0,
                centering_offset
            };

            divergence_op_->apply(div_input_field, rk_df_dt_list[0]);

            sclx::assign_array(f0, f).get();
            div_input_field.scalar_field_ = f;

            std::vector<std::future<void>> transform_futures;

            size_t end_slice = f0.elements();
            if (task.boundary_idx_start_ > 0) {
                end_slice = task.boundary_idx_start_;
            }

            for (int i = 0; i < 3; ++i) {
                sclx::algorithm::elementwise_reduce(
                    forward_euler<T>{dt * runge_kutta_4::df_dt_weights[i]},
                    f.get_range({0}, {end_slice}),
                    f0.get_range({0}, {end_slice}),
                    rk_df_dt_list[i].get_range({0}, {end_slice})
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
                f.get_range({0}, {end_slice}),
                f0.get_range({0}, {end_slice}),
                rk_df_dt_list[0].get_range({0}, {end_slice})
            )
                .get();
        }

      private:
        std::shared_ptr<void> field_map_;
        using impl_t = void (*)(
            advection_task& task,
            sclx::array<T, 1> (&rk_df_dt_list)[4]
        );
        impl_t impl_;

        sclx::array<T, 1> f0_;
        sclx::array<T, 1> f_;
        T dt_;
        T centering_offset_;
        std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;
        size_t boundary_idx_start_;
    };

    class advection_executor {
      public:
        ~advection_executor() {
            if (thread_data_ == nullptr) {
                return;
            }
            thread_data_->stop_thread.store(true);
            thread_->join();
        }

        template<class FieldMap>
        static std::future<void> submit(
            std::shared_ptr<divergence_operator<T, Dimensions>>& divergence_op_,
            FieldMap& velocity_field,
            sclx::array<T, 1>& f0,
            sclx::array<T, 1>& f,
            T dt,
            T centering_offset,
            size_t boundary_idx_start
        ) {
            std::shared_ptr<advection_executor> executor = get_executor();
            auto task_ptr = std::make_unique<advection_task>(
                divergence_op_,
                velocity_field,
                f0,
                f,
                dt,
                centering_offset,
                boundary_idx_start
            );
            auto expected = false;
            while (!executor->thread_data_->is_preparing
                        .compare_exchange_weak(expected, true)) {
                expected = false;
                std::this_thread::yield();
            }
            while (executor->thread_data_->is_task_submitted.load()) {
                std::this_thread::yield();
            }
            executor->thread_data_->task          = std::move(task_ptr);
            executor->thread_data_->ready_promise = std::promise<void>();
            auto fut = executor->thread_data_->ready_promise.get_future();
            executor->thread_data_->is_task_submitted.store(true);
            executor->thread_data_->is_preparing.store(false);
            if (executor->thread_data_->stop_thread.load()) {
                sclx::throw_exception<std::runtime_error>(
                    "Executor thread was stopped before task was submitted.",
                    "naga::nonlocal_calculus::advection_operator::"
                );
            }
            return fut;
        }

        advection_executor(const advection_executor&)            = default;
        advection_executor& operator=(const advection_executor&) = default;
        advection_executor(advection_executor&&)                 = default;
        advection_executor& operator=(advection_executor&&)      = default;

        static uint max_concurrent_threads() {
            std::lock_guard<std::mutex> lock(get_executor_pool().executors_mutex
            );
            return static_cast<uint>(get_executor_pool().executors.size());
        }

        static void set_max_concurrent_threads(uint max_concurrent_threads) {
            std::lock_guard<std::mutex> lock(get_executor_pool().executors_mutex
            );
            std::vector<std::shared_ptr<advection_executor>>& old_executors
                = get_executor_pool().executors;
            std::vector<std::shared_ptr<advection_executor>> new_executors;
            for (uint i = 0; i < max_concurrent_threads; ++i) {
                if (i < old_executors.size()) {
                    new_executors.push_back(old_executors[i]);
                } else {
                    new_executors.push_back(
                        std::make_shared<advection_executor>(advection_executor{
                        })
                    );
                    new_executors.back()->init_thread();
                }
            }
            old_executors = new_executors;
        }

      private:
        advection_executor() = default;

        std::unique_ptr<std::thread> thread_{nullptr};

        void init_thread() {
            thread_data_      = std::make_shared<thread_data_t>();
            auto& thread_data = thread_data_;
            thread_ = std::make_unique<std::thread>([thread_data]() mutable {
                sclx::array<T, 1> rk_df_dt_list[4];
                thread_data->is_running.store(true);
                while (!thread_data->stop_thread.load()) {
                    if (thread_data->is_task_submitted.load()) {
                        (*thread_data->task)(rk_df_dt_list);
                        thread_data->ready_promise.set_value();
                        thread_data->is_task_submitted.store(false);
                    } else {
                        std::this_thread::yield();
                    }
                }
                thread_data->is_running.store(false);
            });
            while (!thread_data->is_running.load()) {
                std::this_thread::yield();
            }
        }

        struct thread_data_t {
            std::atomic<bool> is_task_submitted{false};
            std::atomic<bool> stop_thread{false};
            std::atomic<bool> is_preparing{false};
            std::atomic<bool> is_running{false};
            std::shared_ptr<advection_task> task{nullptr};
            std::promise<void> ready_promise;
        };

        struct executor_pool {
            std::vector<std::shared_ptr<advection_executor>> executors;
            std::mutex executors_mutex;
            uint next_executor{0};
            executor_pool() {
                int max_concurrent_threads_ = 1;
                for (uint i = 0; i < max_concurrent_threads_; ++i) {
                    executors.push_back(std::make_shared<advection_executor>(
                        advection_executor{}
                    ));
                    executors.back()->init_thread();
                }
            }
        };

        static executor_pool& get_executor_pool() {
            static executor_pool executor_pool;
            return executor_pool;
        }

        static std::shared_ptr<advection_executor> get_executor() {
            auto& executor_pool = get_executor_pool();
            std::lock_guard<std::mutex> lock(executor_pool.executors_mutex);
            auto executor
                = executor_pool.executors[executor_pool.next_executor];
            executor_pool.next_executor = (executor_pool.next_executor + 1)
                                        % executor_pool.executors.size();
            return executor;
        }

        std::shared_ptr<thread_data_t> thread_data_{nullptr};
    };

    void set_max_concurrent_threads(uint max_concurrent_threads) {
        max_concurrent_threads = std::min(
            std::thread::hardware_concurrency(),
            max_concurrent_threads
        );
        advection_executor::set_max_concurrent_threads(max_concurrent_threads);
    }

    [[nodiscard]] uint max_concurrent_threads() const {
        return advection_executor::max_concurrent_threads();
    }

    template<class FieldMap>
    std::future<void> step_forward(
        FieldMap& velocity_field,
        sclx::array<T, 1>& f0,
        sclx::array<T, 1>& f,
        T dt,
        T centering_offset        = T(0),
        size_t boundary_idx_start = 0
    ) {
        if (f0.elements() != domain_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "f0 must have the same number of elements as the domain for "
                "which this operator was created.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }
        if (f.elements() != domain_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "f must have the same number of elements as the domain for "
                "which this operator was created.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }
        return advection_executor::submit(
            divergence_op_,
            velocity_field,
            f0,
            f,
            dt,
            centering_offset,
            boundary_idx_start
        );
    }

    const divergence_operator<T, Dimensions>& divergence_op() const {
        return *divergence_op_;
    }

    template<class Archive>
    void save(Archive& ar) const {
        ar(*divergence_op_);
        ar(domain_size_);
        auto max_concurrent_threads_ = max_concurrent_threads();
        ar(max_concurrent_threads_);
    }

    template<class Archive>
    void load(Archive& ar) {
        divergence_op_ = std::make_shared<divergence_operator<T, Dimensions>>();
        ar(*divergence_op_);
        ar(domain_size_);
        auto max_concurrent_threads_ = max_concurrent_threads();
        ar(max_concurrent_threads_);
        set_max_concurrent_threads(max_concurrent_threads_);
    }

    advection_operator slice(size_t new_size) const {
        advection_operator new_op;
        new_op.divergence_op_
            = std::make_shared<divergence_operator<T, Dimensions>>(
                divergence_op_->slice(new_size)
            );
        new_op.domain_size_ = new_size;
        return new_op;
    }

    //    void enable_cusparse_algorithm() {
    //        divergence_op_->enable_cusparse_algorithm();
    //
    //        if (cusparse_desc_ == nullptr) {
    //            cusparse_desc_ = std::make_shared<cusparse_algo_desc>();
    //
    //            auto& identity_mat1 = cusparse_desc_->identity_mat1;
    //            sclx::array<T, 2> ones{1, domain_size_};
    //            sclx::fill(ones, T(1));
    //            sclx::array<uint, 2> indices{1, domain_size_};
    //            std::iota(indices.begin(), indices.end(), 0);
    //            identity_mat1
    //                = matrix_type::create_from_index_stencil(indices, ones);
    //            cusparse_desc_->A = matrix_type();
    //        }
    //    }

    //    advection_operator
    //    make_fast_operator_for_velocity(const T
    //    (&constant_velocity)[Dimensions]) {
    //        bool was_cusparse_enabled  = cusparse_algorithm_enabled();
    //        advection_operator fast_op = *this;
    //        fast_op.cusparse_desc_     = nullptr;
    //        fast_op.enable_cusparse_algorithm();
    //        fast_op.compute_A_matrix(constant_velocity);
    //        return fast_op;
    //    }
    //
    //    void disable_cusparse_algorithm() {
    //        divergence_op_->disable_cusparse_algorithm();
    //        cusparse_desc_ = nullptr;
    //    }
    //
    //    bool cusparse_algorithm_enabled() const {
    //        return divergence_op_->cusparse_algorithm_enabled();
    //    }

  public:
    std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;
    size_t domain_size_;

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[3] = {1. / 2., 1. / 2., 1.};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };

    //    using matrix_type =
    //        typename divergence_operator<T, Dimensions>::matrix_type;
    //    using vector_type =
    //        typename divergence_operator<T, Dimensions>::vector_type;
    //    struct cusparse_algo_desc {
    //        matrix_type identity_mat1;
    //        matrix_type identity_mat2;
    //        matrix_type A;
    //        naga::linalg::matrix_mult<matrix_type, matrix_type, matrix_type>
    //            mat_mult;
    //    };
    //    std::shared_ptr<cusparse_algo_desc> cusparse_desc_{nullptr};
};

}  // namespace naga::nonlocal_calculus
