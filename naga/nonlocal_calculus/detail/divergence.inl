
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

    template<class KernelType, uint Dimensions>
    struct antisymmetric_divergence_kernel {
        static_assert(Dimensions == Dimensions, "Kernel not implemented for provided template parameters");

        template<class D, class PointTypeT, class PointTypeU>
        __host__ __device__ auto
        operator()(D &&delta, PointTypeT &&x, PointTypeU &&y)
        -> naga::point_t<std::decay_t<decltype(x[0])>, Dimensions> {}
    };


    class w4_bspline;

    template<>
    struct antisymmetric_divergence_kernel<w4_bspline, 2> {

        template<class D, class PointTypeT, class PointTypeU>
        __host__ __device__ static auto
        compute(D &&delta, PointTypeT &&x, PointTypeU &&y)
        -> naga::point_t<std::decay_t<decltype(x[0])>, 2> {
            using T = std::decay_t<decltype(x[0])>;
            naga::point_t<T, 2> xy({y[0] - x[0], y[1] - x[1]});
            T r = math::loopless::norm<2>(xy);
            T cos_theta = xy[0] / r;
            T sin_theta = xy[1] / r;

            auto &alpha = xy;

            if (2.f * r / delta < 1) {
                alpha[0] = (240.0f / 7.0f) * r * (2.f * delta - 3.f * r) * cos_theta /
                           (math::pi<T> * math::loopless::pow<5>(delta));
                alpha[1] = (240.0f / 7.0f) * r * (2.f * delta - 3.f * r) * sin_theta /
                           (math::pi<T> * math::loopless::pow<5>(delta));
            } else if (2.f * r / delta < 2) {
                alpha[0] = (240.0f / 7.0f) * math::loopless::pow<2>(delta - r) * cos_theta /
                           (math::pi<T> * math::loopless::pow<5>(delta));
                alpha[1] = (240.0f / 7.0f) * math::loopless::pow<2>(delta - r) * sin_theta /
                           (math::pi<T> * math::loopless::pow<5>(delta));
            } else {
                alpha[0] = 0;
                alpha[1] = 0;
            }

            return alpha;
        }
    };

    template<>
    struct antisymmetric_divergence_kernel<w4_bspline, 3> {

        template<class D, class PointTypeT, class PointTypeU>
        __host__ __device__ static auto
        compute(D &&delta, PointTypeT &&x, PointTypeU &&y)
        -> naga::point_t<std::decay_t<decltype(x[0])>, 3> {
            using T = std::decay_t<decltype(x[0])>;
            naga::point_t<T, 3> xy({y[0] - x[0], y[1] - x[1], y[2] - x[2]});
            T r = math::loopless::norm<3>(xy);
            T r_x1x2 = math::loopless::norm<2>(xy);
            T sin_phi = r_x1x2 / r;
            T cos_phi = xy[2] / r;
            T cos_theta = (r_x1x2 != 0) ? xy[0] / r_x1x2 : 0;
            T sin_theta = (r_x1x2 != 0) ? xy[1] / r_x1x2 : 0;

            auto &alpha = xy;

            if (2.f * r / delta < 1) {
                alpha[0] = 48.f * r * (2.f * delta - 3.f * r) * sin_phi * cos_theta /
                           (math::pi<T> * math::loopless::pow<6>(delta));
                alpha[1] = 48.f * r * (2.f * delta - 3.f * r) * sin_phi * sin_theta /
                           (math::pi<T> * math::loopless::pow<6>(delta));
                alpha[2] = 48.f * r * (2.f * delta - 3.f * r) * cos_phi / (math::pi<T> * math::loopless::pow<6>(delta));
            } else if (2.f * r / delta < 2) {
                alpha[0] = 48.f * math::loopless::pow<2>(delta - r) * sin_phi * cos_theta /
                           (math::pi<T> * math::loopless::pow<6>(delta));
                alpha[1] = 48.f * math::loopless::pow<2>(delta - r) * sin_phi * sin_theta /
                           (math::pi<T> * math::loopless::pow<6>(delta));
                alpha[2] = 48.f * math::loopless::pow<2>(delta - r) * cos_phi /
                           (math::pi<T> * math::loopless::pow<6>(delta));
            } else {
                alpha[0] = 0;
                alpha[1] = 0;
                alpha[2] = 0;
            }

            return alpha;
        }
    };

    template<class T, uint Dimensions>
    void compute_divergence_weights(
            const sclx::array<T, 3> &weights,
            const sclx::array<T, 2> &domain,
            const sclx::array<T, 1> &interaction_radii,
            const sclx::array<T, 2> &quad_interp_weights,
            const sclx::array<sclx::index_t, 2> &support_indices
    ) {
        detail::quadrature_point_map<T, Dimensions> quadrature_points_map(
                domain,
                interaction_radii
        );

        sclx::execute_kernel([&](sclx::kernel_handler &handler) {
            sclx::local_array<T, 3> local_weights(
                    handler,
                    {Dimensions, num_interp_support, 64}
            );

            handler.launch(
                    sclx::md_range_t<1>{weights.shape()[2]},
                    weights,
                    [=] __device__(
                            const sclx::md_index_t<1> &idx,
                            const sclx::kernel_info<> &info
                    ) mutable {
                        for (uint s = 0; s < num_interp_support; ++s) {
                            for (uint d = 0; d < Dimensions; ++d) {
                                local_weights(d, s, info.local_thread_id()[0]) = T(0);
                            }
                        }

                        T delta = interaction_radii[idx];
                        constexpr uint num_quad_points = (Dimensions == 2)
                                                         ? num_quad_points_2d
                                                         : num_quad_points_3d;

                        T *x_i = &domain(0, idx[0]);
                        for (uint q = 0; q < num_quad_points; ++q) {
                            auto x_k
                                    = quadrature_points_map[num_quad_points * idx[0] + q];
                            auto alpha = antisymmetric_divergence_kernel<w4_bspline, Dimensions>::compute(
                                    delta,
                                    x_i,
                                    x_k
                            );

                            const uint &r_idx = q % num_radial_quad_points;
                            const uint &theta_idx = (q / num_radial_quad_points)
                                                    % num_theta_quad_points;
                            T quad_weight = const_radial_quad_weights<T>[r_idx] * delta / 2.f;
                            quad_weight *= const_radial_quad_weights<T>[theta_idx] * 2.f * math::pi<T> / 2.f;
                            if (Dimensions == 3) {
                                const uint &phi_idx = q / (num_radial_quad_points * num_theta_quad_points);
                                quad_weight *= const_radial_quad_weights<T>[phi_idx] * math::pi<T> / 2.f;
                            }

                            if (Dimensions == 2) {
                                point_t<T, 2> xy({x_k[0] - x_i[0], x_k[1] - x_i[1]});
                                T r = math::loopless::norm<2>(xy);
                                quad_weight *= r;
                            } else {
                                point_t<T, 3> xy({x_k[0] - x_i[0], x_k[1] - x_i[1], x_k[2] - x_i[2]});
                                T r = math::loopless::norm<3>(xy);
                                T r_x1x2 = math::loopless::norm<2>(xy);
                                T sin_phi = r_x1x2 / r;
                                quad_weight *= math::loopless::pow<2>(r) * sin_phi;
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