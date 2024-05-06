
// BSD 3-Clause License
//
// Copyright (c) 2024 Jack Myers
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
#include <naga/nonlocal_calculus/advection.cuh>
#include <scalix/algorithm/elementwise_reduce.cuh>
#include <scalix/algorithm/transform.cuh>
#include <scalix/array.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

template<class RHS, size_t StateCount>
struct rk4_solver {

    using T = typename RHS::value_type;

    rk4_solver(std::shared_ptr<RHS> rhs) : rhs(rhs) {}

    void step_forward(std::array<sclx::array<T, 1>, StateCount> f0, std::array<sclx::array<T, 1>, StateCount> f, T t0, T dt) {
        for (size_t i = 0; i < StateCount; ++i) {
            sclx::assign_array(f0[i], f[i]).get();
        }

        auto t = t0;

        auto k1 = rhs->operator()(f, t, 0.f);
        for (size_t i = 0; i < StateCount; ++i) {
            if (df_dt[i].elements() != k1[i].elements()) {
                df_dt[i] = sclx::array<T, 1>{k1[i].elements()};
            }
            sclx::algorithm::elementwise_reduce(
                naga::nonlocal_calculus::forward_euler<T>{
                    dt * runge_kutta_4::df_dt_weights[0]
                },
                f[i],
                f0[i],
                k1[i]
            )
                .get();
            sclx::algorithm::transform(
                k1[i],
                k1[i],
                runge_kutta_4::summation_weights[0],
                sclx::algorithm::multiplies<>{}
            )
                .get();
            sclx::assign_array(k1[i], df_dt[i]).get();
        }

        t = t0 + dt * runge_kutta_4::df_dt_weights[0];

        auto k2 = rhs->operator()(f, t, dt * runge_kutta_4::df_dt_weights[0]);
        for (size_t i = 0; i < StateCount; ++i) {
            sclx::algorithm::elementwise_reduce(
                naga::nonlocal_calculus::forward_euler<T>{
                    dt * runge_kutta_4::df_dt_weights[1]
                },
                f[i],
                f0[i],
                k2[i]
            )
                .get();
            sclx::algorithm::transform(
                k2[i],
                k2[i],
                runge_kutta_4::summation_weights[1],
                sclx::algorithm::multiplies<>{}
            )
                .get();
            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>(),
                df_dt[i],
                df_dt[i],
                k2[i]
            )
                .get();
        }

        t = t0 + dt * runge_kutta_4::df_dt_weights[1];

        auto k3 = rhs->operator()(f, t, dt * runge_kutta_4::df_dt_weights[1]);
        for (size_t i = 0; i < StateCount; ++i) {
            sclx::algorithm::elementwise_reduce(
                naga::nonlocal_calculus::forward_euler<T>{
                    dt * runge_kutta_4::df_dt_weights[2]
                },
                f[i],
                f0[i],
                k3[i]
            )
                .get();
            sclx::algorithm::transform(
                k3[i],
                k3[i],
                runge_kutta_4::summation_weights[2],
                sclx::algorithm::multiplies<>{}
            )
                .get();
            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>(),
                df_dt[i],
                df_dt[i],
                k3[i]
            )
                .get();
        }

        t = t0 + dt * runge_kutta_4::df_dt_weights[2];

        auto k4 = rhs->operator()(f, t, dt * runge_kutta_4::df_dt_weights[2]);
        for (size_t i = 0; i < StateCount; ++i) {
            sclx::algorithm::transform(
                k4[i],
                k4[i],
                runge_kutta_4::summation_weights[3],
                sclx::algorithm::multiplies<>{}
            )
                .get();
            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>(),
                df_dt[i],
                df_dt[i],
                k4[i]
            )
                .get();
        }

        for (size_t i = 0; i < StateCount; ++i) {
            sclx::algorithm::elementwise_reduce(
                naga::nonlocal_calculus::forward_euler<T>{dt},
                f[i],
                f0[i],
                df_dt[i]
            )
                .get();
        }

        rhs->finalize(f0, f, t0, dt);
    }

    std::shared_ptr<RHS> rhs;
    std::array<sclx::array<T, 1>, StateCount> df_dt{};

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[3] = {1. / 2., 1. / 2., 1.};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };
};

}  // namespace naga::fluids::nonlocal_lbm::detail
