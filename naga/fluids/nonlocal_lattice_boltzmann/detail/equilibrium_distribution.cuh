
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

#include "../../../math.cuh"

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
__device__ typename lattice_traits<Lattice>::value_type
compute_equilibrium_distribution(
    const typename lattice_traits<Lattice>::value_type& unitless_density,
    const typename lattice_traits<Lattice>::value_type* unitless_velocity,
    const typename lattice_traits<Lattice>::value_type* lattice_velocity,
    const typename lattice_traits<Lattice>::value_type& lattice_weight
) {
    using value_type          = typename lattice_traits<Lattice>::value_type;
    constexpr uint dimensions = lattice_traits<Lattice>::dimensions;

    value_type u_dot_u
        = math::loopless::dot<dimensions>(unitless_velocity, unitless_velocity);
    value_type u_dot_c
        = math::loopless::dot<dimensions>(unitless_velocity, lattice_velocity);

    constexpr value_type c_s = lattice_traits<Lattice>::lattice_speed_of_sound;

    using namespace math::loopless;

    return lattice_weight * unitless_density
         * (1 + u_dot_c / (pow<2>(c_s)) + pow<2>(u_dot_c) / (2 * pow<4>(c_s))
            - u_dot_u);
}

template<class Lattice>
struct compute_equilibrium_subtask {
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;

    __device__ void operator()(
        const sclx::md_index_t<1>& idx,
        const sclx::kernel_info<>& info
    ) {
        if (info.local_thread_linear_id() == 0) {
            for (int i = 0; i < dimensions * lattice_size; ++i) {
                lattice_velocities(i % dimensions, i / dimensions)
                    = lattice_interface<Lattice>::lattice_velocities()
                          .vals[i / dimensions][i % dimensions];

                if (i % dimensions == 0) {
                    lattice_weights(i / dimensions)
                        = lattice_interface<Lattice>::lattice_weights()
                              .vals[i / dimensions];
                }
            }
        }
        handler.syncthreads();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_shared(alpha, info.local_thread_linear_id())
                = lattice_distributions[alpha][idx[0]];
        }

        value_type unitless_density = fluid_density[idx] / density_scale;
        value_type unitless_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitless_velocity[d] = fluid_velocity(d, idx[0]) / velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<Lattice>(
                unitless_density,
                unitless_velocity,
                &lattice_velocities(0, alpha),
                lattice_weights(alpha)
            );

            lattice_equilibrium_distributions[alpha][idx] = f_tilde_eq;
        }
    }
    sclx::kernel_handler handler;

    sclx::array_list<value_type, 1, lattice_size>
        lattice_equilibrium_distributions;

    sclx::array_list<const value_type, 1, lattice_size> lattice_distributions;

    sclx::array<const value_type, 1> fluid_density;
    sclx::array<const value_type, 2> fluid_velocity;

    sclx::local_array<value_type, 2> f_shared;
    sclx::local_array<value_type, 2> lattice_velocities;
    sclx::local_array<value_type, 1> lattice_weights;
    value_type density_scale;
    value_type velocity_scale;
};

}  // namespace naga::fluids::nonlocal_lbm::detail
