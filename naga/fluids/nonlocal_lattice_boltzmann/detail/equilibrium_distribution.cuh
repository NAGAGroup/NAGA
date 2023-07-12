
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
#include "../lattices.cuh"
#include "subtask_factory.h"

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
class compute_equilibrium_subtask;

template<class Lattice>
struct subtask_factory<compute_equilibrium_subtask<Lattice>> {
    static compute_equilibrium_subtask<Lattice> create(
        const simulation_engine<Lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::
            array_list<typename Lattice::value_type, 1, Lattice::size>&
                lattice_equilibrium_values
    );
};

}  // namespace naga::fluids::nonlocal_lbm::detail
