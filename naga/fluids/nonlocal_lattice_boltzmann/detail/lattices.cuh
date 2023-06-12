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

namespace naga::fluids::nonlocal_lbm::detail {

template<class T, uint Dimensions, uint LatticeSize>
struct lattice_velocities_t {
    T vals[LatticeSize][Dimensions];
};

template<class T, uint LatticeSize>
struct lattice_weights_t {
    T vals[LatticeSize];
};

template<class T, uint LatticeSize>
struct collision_matrix_t {
    T vals[LatticeSize][LatticeSize];
};

template<class Lattice>
struct lattice_interface {
    static constexpr uint size       = Lattice::size;
    static constexpr uint dimensions = Lattice::dimensions;
    using value_type                 = typename Lattice::value_type;
    static_assert(
        std::is_same_v<Lattice, Lattice>,
        "lattice_interface not specialized for this lattice"
    );

    static constexpr lattice_velocities_t<value_type, dimensions, size>
    lattice_velocities() {
        return {};
    }

    static constexpr lattice_weights_t<value_type, size> lattice_weights() {
        return {};
    }

    static constexpr collision_matrix_t<value_type, size> collision_matrix() {
        return {};
    }

    __host__ __device__ static void compute_moment_projection(
        value_type* projection,
        const value_type* distribution,
        const value_type& density,
        const value_type* velocity,
        const value_type& lattice_viscosity,
        const value_type& lattice_time_step
    ) {
        auto& k        = projection;
        auto& f        = distribution;
        auto& rho      = density;
        auto& u        = velocity;
        auto& lat_nu   = lattice_viscosity;
        auto& lat_dt       = lattice_time_step;

        value_type omega_ab = 1.f / (3.f * lat_nu + .5f * lat_dt);
    }

    static constexpr int get_bounce_back_idx(const int& alpha) { return -1; }
};

}  // namespace naga::fluids::nonlocal_lbm::detail
