
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

namespace naga::nonlocal_calculus::detail {

template<class T, uint Dimensions>
using divergence_operator_type = divergence_operator<T, Dimensions>;

template<class T, uint Dimensions>
void compute_divergence_weights(
    const sclx::array<T, 3>& weights,
    const sclx::array<T, 2>& domain,
    const sclx::array<T, 1>& interaction_radii,
    const sclx::array<T, 2>& quad_interp_weights,
    const sclx::array<sclx::index_t, 2>& support_indices
) {
    detail::quadrature_point_map<T, Dimensions> quadrature_points_map(
        domain,
        interaction_radii
    );

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        sclx::local_array<T, 3> local_weights(
            handler,
            {Dimensions, num_interp_support, 64}
        );

        handler.launch(
            sclx::md_range_t<1>{weights.shape()[2]},
            weights,
            [=] __device__(
                const sclx::md_index_t<1>& idx,
                const sclx::kernel_info<>& info
            ) mutable {
                for (uint s = 0; s < num_interp_support; ++s) {
                    for (uint d = 0; d < Dimensions; ++d) {
                        local_weights(d, s, info.local_thread_id()[0]) = T(0);
                    }
                }

                T delta                        = interaction_radii[idx];
                constexpr uint num_quad_points = (Dimensions == 2)
                                                   ? num_quad_points_2d
                                                   : num_quad_points_3d;

                T* x_i = &domain(0, idx[0]);
                distance_functions::loopless::euclidean<Dimensions>
                    distance_func;
                for (uint q = 0; q < num_quad_points; ++q) {
                    auto x_k
                        = quadrature_points_map[num_quad_points * idx[0] + q];
                    T alpha[Dimensions];
                    T r    = 2.f * distance_func(x_i, x_k) / delta;
                    T _int = (3.f * (3.f * r - 4.f)) / 4.f;
                    _int   = (r < 1.f)
                               ? _int
                               : ((3.f * (r - 2.f) * (r - 2.f)) / (4.f * r));

                    for (uint d = 0; d < Dimensions; ++d) {
                        alpha[d] = (x_k[d] - x_i[d]) * 2.f / delta * _int;
                        if constexpr (Dimensions == 2) {
                            alpha[d] *= 10.f / (14.f);
                        } else {
                            alpha[d]
                                *= 2.f
                                 * distance_functions::loopless::euclidean<2>{}(
                                       x_i,
                                       x_k
                                 )
                                 / delta;
                            alpha[d] *= 16.f / (30.f * math::pi<T>);
                        }
                        alpha[d] *= r * 4.f * 2.f / delta;
                    }

                    T quad_weight = const_radial_quad_weights<
                        T>[q % num_radial_quad_points];
                    quad_weight *= 2.f * math::pi<T>
                                 / static_cast<T>(num_theta_quad_points);
                    if (Dimensions == 3) {
                        quad_weight *= math::pi<T>
                                     / static_cast<T>(num_phi_quad_points);
                    }

                    for (uint index = 0;
                         index < num_interp_support * Dimensions;
                         ++index) {
                        uint d = index % Dimensions;
                        uint s = index / Dimensions;
                        local_weights(d, s, info.local_thread_id()[0])
                            += alpha[d] * quad_weight
                             * (quad_interp_weights(
                                    s,
                                    num_quad_points * idx[0] + q
                                )
                                + static_cast<T>(s == 0));
                    }
                }

                memcpy(
                    &weights(0, 0, idx[0]),
                    &local_weights(0, 0, info.local_thread_id()[0]),
                    sizeof(T) * num_interp_support * Dimensions
                );
            },
            {64}
        );
    }).get();
}

}  // namespace naga::nonlocal_calculus::detail