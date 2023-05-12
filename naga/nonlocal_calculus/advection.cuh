
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
    friend class operator_builder<T, Dimensions>;

    template<class FieldMap>
    class divergence_field_map {
      public:
        using point_type = point_t<T, Dimensions>;

        divergence_field_map(
            const FieldMap& velocity_field,
            const sclx::array<T, 1>& scalar_field,
            T centering_offset = T(0)
        )
            : velocity_field_(velocity_field),
              scalar_field_(scalar_field) {}

        __host__ __device__ point_type operator[](const sclx::index_t& index
        ) const {
            point_type velocity      = velocity_field_[index];
            T scalar                 = scalar_field_[index];
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
        FieldMap velocity_field_;
        sclx::array<T, 1> scalar_field_;
        T centering_offset_;
    };

    static advection_operator create(const sclx::array<T, 2>& domain) {
        advection_operator op;
        op.divergence_op_
            = std::make_shared<divergence_operator<T, Dimensions>>(
                divergence_operator<T, Dimensions>::create(domain)
            );
        for (auto& rk_df_dt : op.rk_df_dt_list_) {
            rk_df_dt = sclx::array<T, 1>{domain.shape()[1]};
        }
        return op;
    }

    template<class FieldMap>
    void step_forward(
        FieldMap& velocity_field,
        sclx::array<T, 1>& f0,
        sclx::array<T, 1>& f,
        T dt,
        T centering_offset = T(0)
    ) {

        sclx::assign_array(static_cast<sclx::array<const T, 1>>(f0), f);

        divergence_field_map<FieldMap> div_input_field{
            velocity_field,
            f,
            centering_offset};

        divergence_op_
            ->apply(div_input_field, rk_df_dt_list_[0], centering_offset);

        for (int i = 0; i < 3; ++i) {
            sclx::algorithm::elementwise_reduce(
                forward_euler<T>{dt * runge_kutta_4::df_dt_weights[i]},
                f,
                f0,
                rk_df_dt_list_[i]
            );

            sclx::algorithm::transform(
                rk_df_dt_list_[i],
                rk_df_dt_list_[i],
                runge_kutta_4::summation_weights[i],
                sclx::algorithm::multiplies<>{}
            );

            divergence_op_->apply(
                div_input_field,
                rk_df_dt_list_[i + 1],
                centering_offset
            );
            sclx::assign_array(
                static_cast<sclx::array<const T, 1>>(f0),
                f
            );
        }

        sclx::algorithm::transform(
            rk_df_dt_list_[3],
            rk_df_dt_list_[3],
            runge_kutta_4::summation_weights[3],
            sclx::algorithm::multiplies<>{}
        );

        sclx::algorithm::elementwise_reduce(
            sclx::algorithm::plus<>{},
            rk_df_dt_list_[0],
            rk_df_dt_list_[0],
            rk_df_dt_list_[1],
            rk_df_dt_list_[2],
            rk_df_dt_list_[3]
        );

        sclx::algorithm::elementwise_reduce(
            forward_euler<T>{dt},
            f,
            f0,
            rk_df_dt_list_[0]
        );
    }

  private:
    advection_operator() = default;
    std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[3] = {1. / 2., 1. / 2., 1.};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };
    sclx::array<T, 1> rk_df_dt_list_[4];
};

}  // namespace naga::nonlocal_calculus
