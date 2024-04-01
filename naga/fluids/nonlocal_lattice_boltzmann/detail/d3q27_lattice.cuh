
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

#include "../../../math.hpp"
#include "lattices.cuh"

namespace naga::fluids::nonlocal_lbm {
template<class T>
struct d3q27_lattice;
}

namespace naga::fluids::nonlocal_lbm::detail {

template<class T>
struct d3q27_lattice_velocities : lattice_velocities_t<T, 3, 27> {
    T vals[27][3] = {
        {0, 0, 0},   {1, 0, 0},    {-1, 0, 0},  {0, 1, 0},   {0, -1, 0},
        {0, 0, 1},   {0, 0, -1},   {1, 1, 0},   {-1, 1, 0},  {1, -1, 0},
        {-1, -1, 0}, {1, 0, 1},    {-1, 0, 1},  {1, 0, -1},  {-1, 0, -1},
        {0, 1, 1},   {0, -1, 1},   {0, 1, -1},  {0, -1, -1}, {1, 1, 1},
        {-1, 1, 1},  {1, -1, 1},   {-1, -1, 1}, {1, 1, -1},  {-1, 1, -1},
        {1, -1, -1}, {-1, -1, -1},
    };
};

template<class T>
struct d3q27_lattice_weights : lattice_weights_t<T, 27> {
    T vals[27]
        = {8. / 27.,  2. / 27.,  2. / 27.,  2. / 27.,  2. / 27.,  2. / 27.,
           2. / 27.,  1. / 54.,  1. / 54.,  1. / 54.,  1. / 54.,  1. / 54.,
           1. / 54.,  1. / 54.,  1. / 54.,  1. / 54.,  1. / 54.,  1. / 54.,
           1. / 54.,  1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216.,
           1. / 216., 1. / 216., 1. / 216.};
};

// clang-format off
template<class T>
__constant__ T d3q27_collision_matrix_data[27][27]
    = {
        { 1, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,-8,},
        { 1, 1, 0, 0, 0, 0, 0, 1, 1,-1,-4, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 4, 0, 0, 4,},
        { 1,-1, 0, 0, 0, 0, 0, 1, 1,-1, 4, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0,-4, 0, 0, 4,},
        { 1, 0, 1, 0, 0, 0, 0,-1, 1,-1, 0,-4, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 4, 0, 4,},
        { 1, 0,-1, 0, 0, 0, 0,-1, 1,-1, 0, 4, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0,-4, 0, 4,},
        { 1, 0, 0, 1, 0, 0, 0, 0,-2,-1, 0, 0,-4, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 4, 4,},
        { 1, 0, 0,-1, 0, 0, 0, 0,-2,-1, 0, 0, 4, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0,-4, 4,},
        { 1, 1, 1, 0, 1, 0, 0, 0, 2, 0,-1,-1, 0, 1, 1, 0, 0,-1, 1, 1, 0, 0,-2,-2,-2, 0,-2,},
        { 1,-1, 1, 0,-1, 0, 0, 0, 2, 0, 1,-1, 0,-1, 1, 0, 0,-1, 1, 1, 0, 0, 2, 2,-2, 0,-2,},
        { 1, 1,-1, 0,-1, 0, 0, 0, 2, 0,-1, 1, 0, 1,-1, 0, 0,-1, 1, 1, 0, 0, 2,-2, 2, 0,-2,},
        { 1,-1,-1, 0, 1, 0, 0, 0, 2, 0, 1, 1, 0,-1,-1, 0, 0,-1, 1, 1, 0, 0,-2, 2, 2, 0,-2,},
        { 1, 1, 0, 1, 0, 1, 0, 1,-1, 0,-1, 0,-1,-1, 0, 1, 0,-1, 1,-1, 0,-2, 0,-2, 0,-2,-2,},
        { 1,-1, 0, 1, 0,-1, 0, 1,-1, 0, 1, 0,-1, 1, 0, 1, 0,-1, 1,-1, 0, 2, 0, 2, 0,-2,-2,},
        { 1, 1, 0,-1, 0,-1, 0, 1,-1, 0,-1, 0, 1,-1, 0,-1, 0,-1, 1,-1, 0, 2, 0,-2, 0, 2,-2,},
        { 1,-1, 0,-1, 0, 1, 0, 1,-1, 0, 1, 0, 1, 1, 0,-1, 0,-1, 1,-1, 0,-2, 0, 2, 0, 2,-2,},
        { 1, 0, 1, 1, 0, 0, 1,-1,-1, 0, 0,-1,-1, 0,-1,-1, 0,-1,-2, 0,-2, 0, 0, 0,-2,-2,-2,},
        { 1, 0,-1, 1, 0, 0,-1,-1,-1, 0, 0, 1,-1, 0, 1,-1, 0,-1,-2, 0, 2, 0, 0, 0, 2,-2,-2,},
        { 1, 0, 1,-1, 0, 0,-1,-1,-1, 0, 0,-1, 1, 0,-1, 1, 0,-1,-2, 0, 2, 0, 0, 0,-2, 2,-2,},
        { 1, 0,-1,-1, 0, 0, 1,-1,-1, 0, 0, 1, 1, 0, 1, 1, 0,-1,-2, 0,-2, 0, 0, 0, 2, 2,-2,},
        { 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,},
        { 1,-1, 1, 1,-1,-1, 1, 0, 0, 1,-2, 2, 2, 0, 0, 0,-1, 1, 0, 0, 1,-1,-1,-1, 1, 1, 1,},
        { 1, 1,-1, 1,-1, 1,-1, 0, 0, 1, 2,-2, 2, 0, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1,-1, 1, 1,},
        { 1,-1,-1, 1, 1,-1,-1, 0, 0, 1,-2,-2, 2, 0, 0, 0, 1, 1, 0, 0,-1,-1, 1,-1,-1, 1, 1,},
        { 1, 1, 1,-1, 1,-1,-1, 0, 0, 1, 2, 2,-2, 0, 0, 0,-1, 1, 0, 0,-1,-1, 1, 1, 1,-1, 1,},
        { 1,-1, 1,-1,-1, 1,-1, 0, 0, 1,-2, 2,-2, 0, 0, 0, 1, 1, 0, 0,-1, 1,-1,-1, 1,-1, 1,},
        { 1, 1,-1,-1,-1,-1, 1, 0, 0, 1, 2,-2,-2, 0, 0, 0, 1, 1, 0, 0, 1,-1,-1, 1,-1,-1, 1,},
        { 1,-1,-1,-1, 1, 1, 1, 0, 0, 1,-2,-2,-2, 0, 0, 0,-1, 1, 0, 0, 1, 1, 1,-1,-1,-1, 1,},
    };
// clang-format on

template<class T>
struct d3q27_collision_matrix : collision_matrix_t<T, 27> {
    const decltype(d3q27_collision_matrix_data<T>)& vals
        = d3q27_collision_matrix_data<T>;
};

template<class T>
struct lattice_interface<d3q27_lattice<T>> {
    static constexpr uint size       = d3q27_lattice<T>::size;
    static constexpr uint dimensions = d3q27_lattice<T>::dimensions;
    using value_type                 = typename d3q27_lattice<T>::value_type;

    static constexpr d3q27_lattice_velocities<T> lattice_velocities() {
        return d3q27_lattice_velocities<T>{};
    }

    static constexpr d3q27_lattice_weights<T> lattice_weights() {
        return d3q27_lattice_weights<T>{};
    }

    static constexpr d3q27_collision_matrix<T> collision_matrix() {
        return d3q27_collision_matrix<T>{};
    }

    __host__ __device__ static void compute_moment_projection(
        value_type* projection,
        const value_type* distribution,
        const value_type& density,
        const value_type* velocity,
        const value_type& lattice_viscosity,
        const value_type& lattice_time_step,
        const value_type& lattice_characteristic_frequency
    ) {
        auto& k      = projection;
        auto& f      = distribution;
        auto& rho    = density;
        auto& u      = velocity;
        auto& lat_nu = lattice_viscosity;
        auto& lat_dt = lattice_time_step;

        k[0]                 = value_type{0};
        k[1]                 = value_type{0};
        k[2]                 = value_type{0};
        k[3]                 = value_type{0};
        const value_type& v0 = u[0];
        const value_type& v1 = u[1];
        const value_type& v2 = u[2];
        value_type v0sq      = v0 * v0;
        value_type v1sq      = v1 * v1;
        value_type v2sq      = v2 * v2;

        value_type Kxy     = compute_central_moment<1, 1, 0>(f, u);
        value_type Kxz     = compute_central_moment<1, 0, 1>(f, u);
        value_type Kyz     = compute_central_moment<0, 1, 1>(f, u);
        value_type Kxx     = compute_central_moment<2, 0, 0>(f, u);
        value_type Kyy     = compute_central_moment<0, 2, 0>(f, u);
        value_type Kzz     = compute_central_moment<0, 0, 2>(f, u);
        value_type Kxxy    = compute_central_moment<2, 1, 0>(f, u);
        value_type Kxxz    = compute_central_moment<2, 0, 1>(f, u);
        value_type Kxyy    = compute_central_moment<1, 2, 0>(f, u);
        value_type Kyyz    = compute_central_moment<0, 2, 1>(f, u);
        value_type Kxzz    = compute_central_moment<1, 0, 2>(f, u);
        value_type Kyzz    = compute_central_moment<0, 1, 2>(f, u);
        value_type Kxyz    = compute_central_moment<1, 1, 1>(f, u);
        value_type Kxyzz   = compute_central_moment<1, 1, 2>(f, u);
        value_type Kxxyz   = compute_central_moment<2, 1, 1>(f, u);
        value_type Kxyyz   = compute_central_moment<1, 2, 1>(f, u);
        value_type Kxxyy   = compute_central_moment<2, 2, 0>(f, u);
        value_type Kxxzz   = compute_central_moment<2, 0, 2>(f, u);
        value_type Kyyzz   = compute_central_moment<0, 2, 2>(f, u);
        value_type Kxxyzz  = compute_central_moment<2, 1, 2>(f, u);
        value_type Kxyyzz  = compute_central_moment<1, 2, 2>(f, u);
        value_type Kxxyyz  = compute_central_moment<2, 2, 1>(f, u);
        value_type Kxxyyzz = compute_central_moment<2, 2, 2>(f, u);

        value_type omega_ab;

        omega_ab = 1.f / (3.f * lat_nu / 2.f + .5f * lat_dt);

        k[4] = omega_ab * (-Kxy / 12);

        k[5] = omega_ab * (-Kxz / 12);

        k[6] = omega_ab * (-Kyz / 12);

        k[7] = omega_ab * (-(Kxx - Kyy) / 12);

        k[8] = omega_ab * (-(Kxx + Kyy - 2 * Kzz) / 36);

//        auto lat_nu_bulk = 1e-6f;
//        omega_ab         = 1.f / (3.f * lat_nu_bulk / 2.f + .5f * lat_dt);

        k[9] = omega_ab * (-(Kxx + Kyy + Kzz - rho) / 18);

        auto lat_nu_high_order = .11f;
        omega_ab         = 1.f / (lat_nu_high_order + .5f * lat_dt);

        k[10] = omega_ab
              * (-(6 * Kxy * v1 + 3 * Kxyy + 6 * Kxz * v2 + 3 * Kxzz
                   + 3 * Kyy * v0 + 3 * Kzz * v0 - 2 * rho * v0)
                 / 72);

        k[11] = omega_ab
              * (-(3 * Kxx * v1 + 3 * Kxxy + 6 * Kxy * v0 + 6 * Kyz * v2
                   + 3 * Kyzz + 3 * Kzz * v1 - 2 * rho * v1)
                 / 72);

        k[12] = omega_ab
              * (-(3 * Kxx * v2 + 3 * Kxxz + 6 * Kxz * v0 + 3 * Kyy * v2
                   + 3 * Kyyz + 6 * Kyz * v1 - 2 * rho * v2)
                 / 72);

        k[13] = omega_ab
              * (-(2 * Kxy * v1 + Kxyy - 2 * Kxz * v2 - Kxzz + Kyy * v0
                   - Kzz * v0)
                 / 8);

        k[14] = omega_ab
              * (-(Kxx * v1 + Kxxy + 2 * Kxy * v0 - 2 * Kyz * v2 - Kyzz
                   - Kzz * v1)
                 / 8);

        k[15] = omega_ab
              * (-(Kxx * v2 + Kxxz + 2 * Kxz * v0 - Kyy * v2 - Kyyz
                   - 2 * Kyz * v1)
                 / 8);

        k[16] = omega_ab * (-(Kxy * v2 + Kxyz + Kxz * v1 + Kyz * v0) / 8);

        k[17] = omega_ab
              * (-(3 * Kxx * v1sq + 3 * Kxx * v2sq - 4 * Kxx + 6 * Kxxy * v1
                   + 3 * Kxxyy + 6 * Kxxz * v2 + 3 * Kxxzz + 12 * Kxy * v0 * v1
                   + 6 * Kxyy * v0 + 12 * Kxz * v0 * v2 + 6 * Kxzz * v0
                   + 3 * Kyy * v0sq + 3 * Kyy * v2sq - 4 * Kyy + 6 * Kyyz * v2
                   + 3 * Kyyzz + 12 * Kyz * v1 * v2 + 6 * Kyzz * v1
                   + 3 * Kzz * v0sq + 3 * Kzz * v1sq - 4 * Kzz - 2 * rho * v0sq
                   - 2 * rho * v1sq - 2 * rho * v2sq + 3 * rho)
                 / 36);

        k[18] = omega_ab
              * (-(3 * Kxx * v1sq + 3 * Kxx * v2sq - 4 * Kxx + 6 * Kxxy * v1
                   + 3 * Kxxyy + 6 * Kxxz * v2 + 3 * Kxxzz + 12 * Kxy * v0 * v1
                   + 6 * Kxyy * v0 + 12 * Kxz * v0 * v2 + 6 * Kxzz * v0
                   + 3 * Kyy * v0sq - 6 * Kyy * v2sq + 2 * Kyy - 12 * Kyyz * v2
                   - 6 * Kyyzz - 24 * Kyz * v1 * v2 - 12 * Kyzz * v1
                   + 3 * Kzz * v0sq - 6 * Kzz * v1sq + 2 * Kzz - 2 * rho * v0sq
                   + rho * v1sq + rho * v2sq)
                 / 72);

        k[19] = omega_ab
              * (-(3 * Kxx * v1sq - 3 * Kxx * v2sq + 6 * Kxxy * v1 + 3 * Kxxyy
                   - 6 * Kxxz * v2 - 3 * Kxxzz + 12 * Kxy * v0 * v1
                   + 6 * Kxyy * v0 - 12 * Kxz * v0 * v2 - 6 * Kxzz * v0
                   + 3 * Kyy * v0sq - 2 * Kyy - 3 * Kzz * v0sq + 2 * Kzz
                   - rho * v1sq + rho * v2sq)
                 / 24);

        // omega_ab = .51f / lat_dt;

        k[20] = omega_ab
              * (-(3 * Kxx * v1 * v2 + 3 * Kxxy * v2 + 3 * Kxxyz + 3 * Kxxz * v1
                   + 6 * Kxy * v0 * v2 + 6 * Kxyz * v0 + 6 * Kxz * v0 * v1
                   + 3 * Kyz * v0sq - 2 * Kyz - rho * v1 * v2)
                 / 24);

        k[21] = omega_ab
              * (-(6 * Kxy * v1 * v2 + 3 * Kxyy * v2 + 3 * Kxyyz + 6 * Kxyz * v1
                   + 3 * Kxz * v1sq - 2 * Kxz + 3 * Kyy * v0 * v2
                   + 3 * Kyyz * v0 + 6 * Kyz * v0 * v1 - rho * v0 * v2)
                 / 24);

        k[22] = omega_ab
              * (-(3 * Kxy * v2sq - 2 * Kxy + 6 * Kxyz * v2 + 3 * Kxyzz
                   + 6 * Kxz * v1 * v2 + 3 * Kxzz * v1 + 6 * Kyz * v0 * v2
                   + 3 * Kyzz * v0 + 3 * Kzz * v0 * v1 - rho * v0 * v1)
                 / 24);

        k[23] = omega_ab
              * (-(6 * Kxy * v1 * v2sq - 4 * Kxy * v1 + 3 * Kxyy * v2sq
                   - 2 * Kxyy + 6 * Kxyyz * v2 + 3 * Kxyyzz
                   + 12 * Kxyz * v1 * v2 + 6 * Kxyzz * v1 + 6 * Kxz * v1sq * v2
                   - 4 * Kxz * v2 + 3 * Kxzz * v1sq - 2 * Kxzz
                   + 3 * Kyy * v0 * v2sq - 2 * Kyy * v0 + 6 * Kyyz * v0 * v2
                   + 3 * Kyyzz * v0 + 12 * Kyz * v0 * v1 * v2
                   + 6 * Kyzz * v0 * v1 + 3 * Kzz * v0 * v1sq - 2 * Kzz * v0
                   - rho * v0 * v1sq - rho * v0 * v2sq + rho * v0)
                 / 24);

        k[24] = omega_ab
              * (-(3 * Kxx * v1 * v2sq - 2 * Kxx * v1 + 3 * Kxxy * v2sq
                   - 2 * Kxxy + 6 * Kxxyz * v2 + 3 * Kxxyzz + 6 * Kxxz * v1 * v2
                   + 3 * Kxxzz * v1 + 6 * Kxy * v0 * v2sq - 4 * Kxy * v0
                   + 12 * Kxyz * v0 * v2 + 6 * Kxyzz * v0
                   + 12 * Kxz * v0 * v1 * v2 + 6 * Kxzz * v0 * v1
                   + 6 * Kyz * v0sq * v2 - 4 * Kyz * v2 + 3 * Kyzz * v0sq
                   - 2 * Kyzz + 3 * Kzz * v0sq * v1 - 2 * Kzz * v1
                   - rho * v0sq * v1 - rho * v1 * v2sq + rho * v1)
                 / 24);

        k[25] = omega_ab
              * (-(3 * Kxx * v1sq * v2 - 2 * Kxx * v2 + 6 * Kxxy * v1 * v2
                   + 3 * Kxxyy * v2 + 3 * Kxxyyz + 6 * Kxxyz * v1
                   + 3 * Kxxz * v1sq - 2 * Kxxz + 12 * Kxy * v0 * v1 * v2
                   + 6 * Kxyy * v0 * v2 + 6 * Kxyyz * v0 + 12 * Kxyz * v0 * v1
                   + 6 * Kxz * v0 * v1sq - 4 * Kxz * v0 + 3 * Kyy * v0sq * v2
                   - 2 * Kyy * v2 + 3 * Kyyz * v0sq - 2 * Kyyz
                   + 6 * Kyz * v0sq * v1 - 4 * Kyz * v1 - rho * v0sq * v2
                   - rho * v1sq * v2 + rho * v2)
                 / 24);

        k[26] = omega_ab
              * (-(27 * Kxx * v1sq * v2sq - 18 * Kxx * v1sq - 18 * Kxx * v2sq
                   + 12 * Kxx + 54 * Kxxy * v1 * v2sq - 36 * Kxxy * v1
                   + 27 * Kxxyy * v2sq - 18 * Kxxyy + 54 * Kxxyyz * v2
                   + 27 * Kxxyyzz + 108 * Kxxyz * v1 * v2 + 54 * Kxxyzz * v1
                   + 54 * Kxxz * v1sq * v2 - 36 * Kxxz * v2 + 27 * Kxxzz * v1sq
                   - 18 * Kxxzz + 108 * Kxy * v0 * v1 * v2sq
                   - 72 * Kxy * v0 * v1 + 54 * Kxyy * v0 * v2sq - 36 * Kxyy * v0
                   + 108 * Kxyyz * v0 * v2 + 54 * Kxyyzz * v0
                   + 216 * Kxyz * v0 * v1 * v2 + 108 * Kxyzz * v0 * v1
                   + 108 * Kxz * v0 * v1sq * v2 - 72 * Kxz * v0 * v2
                   + 54 * Kxzz * v0 * v1sq - 36 * Kxzz * v0
                   + 27 * Kyy * v0sq * v2sq - 18 * Kyy * v0sq - 18 * Kyy * v2sq
                   + 12 * Kyy + 54 * Kyyz * v0sq * v2 - 36 * Kyyz * v2
                   + 27 * Kyyzz * v0sq - 18 * Kyyzz + 108 * Kyz * v0sq * v1 * v2
                   - 72 * Kyz * v1 * v2 + 54 * Kyzz * v0sq * v1 - 36 * Kyzz * v1
                   + 27 * Kzz * v0sq * v1sq - 18 * Kzz * v0sq - 18 * Kzz * v1sq
                   + 12 * Kzz - 9 * rho * v0sq * v1sq - 9 * rho * v0sq * v2sq
                   + 9 * rho * v0sq - 9 * rho * v1sq * v2sq + 9 * rho * v1sq
                   + 9 * rho * v2sq - 7 * rho)
                 / 216);
    }

    static constexpr int get_bounce_back_idx(const int& alpha) {

        constexpr int bb_idcs[]
            = {0,  2,  1,  4,  3,  6,  5,  10, 9,  8,  7,  14, 13, 12,
               11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19};

        return bb_idcs[alpha];
    }

  private:
    template<uint m, uint n, uint p>
    __host__ __device__ static value_type
    compute_central_moment(const value_type* fi, const value_type* u) {
        value_type sum = 0;

        constexpr d3q27_lattice_velocities<value_type> lattice_velocities
            = lattice_interface::lattice_velocities();

        for (uint alpha = 0; alpha < size; ++alpha) {
            sum += fi[alpha]
                 * math::loopless::pow<m>(
                       lattice_velocities.vals[alpha][0] - u[0]
                 )
                 * math::loopless::pow<n>(
                       lattice_velocities.vals[alpha][1] - u[1]
                 )
                 * math::loopless::pow<p>(
                       lattice_velocities.vals[alpha][2] - u[2]
                 );
        }

        return sum;
    }
};

}  // namespace naga::fluids::nonlocal_lbm::detail