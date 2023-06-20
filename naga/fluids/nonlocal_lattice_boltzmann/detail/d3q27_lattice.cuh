
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

#include "lattices.cuh"
#include "../../../math.cuh"

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
        const value_type& lattice_time_step
    ) {
        auto& k      = projection;
        auto& f      = distribution;
        auto& rho    = density;
        auto& u      = velocity;
        auto& lat_nu = lattice_viscosity;
        auto& lat_dt = lattice_time_step;

        value_type omega_ab = 1.f / (3.f * lat_nu + .5f * lat_dt);

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
        k[4]               = omega_ab / -12 * (Kxy);
        k[5]               = omega_ab / -12 * (Kxz);
        k[6]               = omega_ab / -12 * (Kyz);
        k[7]               = omega_ab / -12 * (Kxx - Kyy);
        k[8]               = omega_ab / -36 * (Kxx + Kyy - 2 * Kzz);

        omega_ab = 1.f / lat_dt;
        k[9]     = omega_ab / -18 * (Kxx + Kyy + Kzz - rho);
        k[10]    = omega_ab / -24
              * (Kxyy + Kxzz - 24 * (k[4] / omega_ab) * v1
                 - 24 * (k[5] / omega_ab) * v2
                 + 6 * v0
                       * ((k[7] / omega_ab) + (k[8] / omega_ab)
                          - 2 * (k[9] / omega_ab)));
        k[13] = omega_ab / -8
              * (Kxyy - Kxzz - 24 * (k[4] / omega_ab) * v1
                 + 24 * (k[5] / omega_ab) * v2
                 + 6 * v0 * ((k[7] / omega_ab) - 3 * (k[8] / omega_ab)));
        k[11] = omega_ab / 24
              * (-Kxxy - Kyzz + 24 * (k[4] / omega_ab) * v0
                 + 24 * (k[6] / omega_ab) * v2
                 + 6 * v1
                       * ((k[7] / omega_ab) - (k[8] / omega_ab)
                          + 2 * (k[9] / omega_ab)));
        k[14] = omega_ab / 8
              * (-Kxxy + Kyzz + 24 * (k[4] / omega_ab) * v0
                 - 24 * (k[6] / omega_ab) * v2
                 + 6 * v1 * ((k[7] / omega_ab) + 3 * (k[8] / omega_ab)));
        k[12] = omega_ab / 24
              * (-Kxxz - Kyyz + 24 * (k[5] / omega_ab) * v0
                 + 24 * (k[6] / omega_ab) * v1
                 + 12 * v2 * ((k[8] / omega_ab) + (k[9] / omega_ab)));
        k[15] = omega_ab / 8
              * (-Kxxz + Kyyz + 24 * (k[5] / omega_ab) * v0
                 - 24 * (k[6] / omega_ab) * v1 + 12 * (k[7] / omega_ab) * v2);
        k[16] = omega_ab / 8
              * (-Kxyz + 12 * (k[4] / omega_ab) * v2
                 + 12 * (k[5] / omega_ab) * v1 + 12 * (k[6] / omega_ab) * v0);
        k[17] = omega_ab / 36
              * (-3 * Kxxyy - 3 * Kxxzz - 3 * Kyyzz
                 + 144 * (k[12] / omega_ab) * v2 - 72 * (k[9] / omega_ab) + rho
                 + 18 * v0sq
                       * ((k[7] / omega_ab) + (k[8] / omega_ab)
                          - 2 * (k[9] / omega_ab))
                 - 144 * v0
                       * (-(k[10] / omega_ab) + (k[4] / omega_ab) * v1
                          + (k[5] / omega_ab) * v2)
                 - 18 * v1sq
                       * ((k[7] / omega_ab) - (k[8] / omega_ab)
                          + 2 * (k[9] / omega_ab))
                 + 144 * v1 * ((k[11] / omega_ab) - (k[6] / omega_ab) * v2)
                 - 36 * v2sq * ((k[8] / omega_ab) + (k[9] / omega_ab)));
        k[18] = omega_ab / 24
              * (-Kxxyy - Kxxzz + 2 * Kyyzz - 12 * (k[7] / omega_ab)
                 - 12 * (k[8] / omega_ab)
                 + 6 * v0sq
                       * ((k[7] / omega_ab) + (k[8] / omega_ab)
                          - 2 * (k[9] / omega_ab))
                 - 48 * v0
                       * (-(k[10] / omega_ab) + (k[4] / omega_ab) * v1
                          + (k[5] / omega_ab) * v2)
                 - 6 * v1sq
                       * ((k[7] / omega_ab) + 5 * (k[8] / omega_ab)
                          - (k[9] / omega_ab))
                 + 24 * v1
                       * (-(k[11] / omega_ab) + (k[14] / omega_ab)
                          + 4 * (k[6] / omega_ab) * v2)
                 + 6 * v2sq
                       * (-3 * (k[7] / omega_ab) + (k[8] / omega_ab)
                          + (k[9] / omega_ab))
                 - 24 * v2 * ((k[12] / omega_ab) - (k[15] / omega_ab)));
        k[19] = omega_ab / 8
              * (-Kxxyy + Kxxzz + 4 * (k[7] / omega_ab) - 12 * (k[8] / omega_ab)
                 + 6 * v0sq * ((k[7] / omega_ab) - 3 * (k[8] / omega_ab))
                 + 16 * v0
                       * ((k[13] / omega_ab) - 3 * (k[4] / omega_ab) * v1
                          + 3 * (k[5] / omega_ab) * v2)
                 - 6 * v1sq
                       * ((k[7] / omega_ab) + (k[8] / omega_ab)
                          + (k[9] / omega_ab))
                 + 8 * v1 * (3 * (k[11] / omega_ab) + (k[14] / omega_ab))
                 + 6 * v2sq
                       * ((k[7] / omega_ab) + (k[8] / omega_ab)
                          + (k[9] / omega_ab))
                 - 8 * v2 * (3 * (k[12] / omega_ab) + (k[15] / omega_ab)));
        k[20] = omega_ab / -8
              * (Kxxyz + 12 * (k[6] / omega_ab) * v0sq + 8 * (k[6] / omega_ab)
                 + 8 * v0
                       * (-2 * (k[16] / omega_ab) + 3 * (k[4] / omega_ab) * v2
                          + 3 * (k[5] / omega_ab) * v1)
                 - 2 * v1
                       * (6 * (k[12] / omega_ab) + 2 * (k[15] / omega_ab)
                          - 3 * v2
                                * ((k[7] / omega_ab) + (k[8] / omega_ab)
                                   + (k[9] / omega_ab)))
                 - 4 * v2 * (3 * (k[11] / omega_ab) + (k[14] / omega_ab)));
        k[21]
            = omega_ab / -8
            * (Kxyyz + 12 * (k[5] / omega_ab) * v1sq + 8 * (k[5] / omega_ab)
               + 2 * v0
                     * (-6 * (k[12] / omega_ab) + 2 * (k[15] / omega_ab)
                        + 12 * (k[6] / omega_ab) * v1
                        + 3 * v2
                              * (-(k[7] / omega_ab) + (k[8] / omega_ab)
                                 + (k[9] / omega_ab)))
               - 8 * v1 * (2 * (k[16] / omega_ab) - 3 * (k[4] / omega_ab) * v2)
               - 4 * v2 * (3 * (k[10] / omega_ab) + (k[13] / omega_ab)));
        k[22]
            = omega_ab / 8
            * (-Kxyzz + 16 * (k[16] / omega_ab) * v2
               - 12 * (k[4] / omega_ab) * v2sq - 8 * (k[4] / omega_ab)
               + 2 * v0
                     * (6 * (k[11] / omega_ab) - 2 * (k[14] / omega_ab)
                        - 12 * (k[6] / omega_ab) * v2
                        + 3 * v1 * (2 * (k[8] / omega_ab) - (k[9] / omega_ab)))
               - 4 * v1
                     * (-3 * (k[10] / omega_ab) + (k[13] / omega_ab)
                        + 6 * (k[5] / omega_ab) * v2));
        k[23]
            = omega_ab / -8
            * (Kxyyzz + 16 * (k[10] / omega_ab)
               + 2 * v0
                     * (-2 * (k[17] / omega_ab) + 4 * (k[18] / omega_ab)
                        + 2 * (k[7] / omega_ab) + 2 * (k[8] / omega_ab)
                        - 4 * (k[9] / omega_ab)
                        + 3 * v1sq * (2 * (k[8] / omega_ab) - (k[9] / omega_ab))
                        - 4 * v1
                              * (-3 * (k[11] / omega_ab) + (k[14] / omega_ab)
                                 + 6 * (k[6] / omega_ab) * v2)
                        + 3 * v2sq
                              * ((k[7] / omega_ab) - (k[8] / omega_ab)
                                 - (k[9] / omega_ab))
                        + 4 * v2 * (3 * (k[12] / omega_ab) - (k[15] / omega_ab))
                     )
               + 4 * v1sq
                     * (3 * (k[10] / omega_ab) - (k[13] / omega_ab)
                        - 6 * (k[5] / omega_ab) * v2)
               - 8 * v1
                     * (-4 * (k[16] / omega_ab) * v2 + 2 * (k[22] / omega_ab)
                        + 3 * (k[4] / omega_ab) * v2sq + 2 * (k[4] / omega_ab))
               + 4 * v2sq * (3 * (k[10] / omega_ab) + (k[13] / omega_ab))
               - 16 * v2 * ((k[21] / omega_ab) + (k[5] / omega_ab)));
        k[24]
            = omega_ab / -8
            * (Kxxyzz + 16 * (k[11] / omega_ab)
               + 2 * v0sq
                     * (6 * (k[11] / omega_ab) - 2 * (k[14] / omega_ab)
                        - 12 * (k[6] / omega_ab) * v2
                        + 3 * v1 * (2 * (k[8] / omega_ab) - (k[9] / omega_ab)))
               - 8 * v0
                     * (-4 * (k[16] / omega_ab) * v2 + 2 * (k[22] / omega_ab)
                        + 3 * (k[4] / omega_ab) * v2sq + 2 * (k[4] / omega_ab)
                        + v1
                              * (-3 * (k[10] / omega_ab) + (k[13] / omega_ab)
                                 + 6 * (k[5] / omega_ab) * v2))
               - 2 * v1
                     * (2 * (k[17] / omega_ab) + 2 * (k[18] / omega_ab)
                        - 2 * (k[19] / omega_ab) + 2 * (k[7] / omega_ab)
                        - 2 * (k[8] / omega_ab) + 4 * (k[9] / omega_ab)
                        + 3 * v2sq
                              * ((k[7] / omega_ab) + (k[8] / omega_ab)
                                 + (k[9] / omega_ab))
                        - 4 * v2 * (3 * (k[12] / omega_ab) + (k[15] / omega_ab))
                     )
               + 4 * v2sq * (3 * (k[11] / omega_ab) + (k[14] / omega_ab))
               - 16 * v2 * ((k[20] / omega_ab) + (k[6] / omega_ab)));
        k[25] = omega_ab / 8
              * (-Kxxyyz - 16 * (k[12] / omega_ab)
                 + v0sq
                       * (-12 * (k[12] / omega_ab) + 4 * (k[15] / omega_ab)
                          + 24 * (k[6] / omega_ab) * v1
                          + 6 * v2
                                * (-(k[7] / omega_ab) + (k[8] / omega_ab)
                                   + (k[9] / omega_ab)))
                 + 8 * v0
                       * (2 * (k[21] / omega_ab) + 3 * (k[5] / omega_ab) * v1sq
                          + 2 * (k[5] / omega_ab)
                          - 2 * v1
                                * (2 * (k[16] / omega_ab)
                                   - 3 * (k[4] / omega_ab) * v2)
                          - v2 * (3 * (k[10] / omega_ab) + (k[13] / omega_ab)))
                 + v1sq
                       * (-12 * (k[12] / omega_ab) - 4 * (k[15] / omega_ab)
                          + 6 * v2
                                * ((k[7] / omega_ab) + (k[8] / omega_ab)
                                   + (k[9] / omega_ab)))
                 + 8 * v1
                       * (2 * (k[20] / omega_ab) + 2 * (k[6] / omega_ab)
                          - v2 * (3 * (k[11] / omega_ab) + (k[14] / omega_ab)))
                 + 4 * v2
                       * ((k[17] / omega_ab) + (k[18] / omega_ab)
                          + (k[19] / omega_ab) + 2 * (k[8] / omega_ab)
                          + 2 * (k[9] / omega_ab)));
        k[26]
            = omega_ab / 216
            * (-27 * Kxxyyzz - 216 * (k[17] / omega_ab)
               - 216 * (k[9] / omega_ab) + rho
               + 54 * v0sq
                     * (-2 * (k[17] / omega_ab) + 4 * (k[18] / omega_ab)
                        + 2 * (k[7] / omega_ab) + 2 * (k[8] / omega_ab)
                        - 4 * (k[9] / omega_ab)
                        + 3 * v1sq * (2 * (k[8] / omega_ab) - (k[9] / omega_ab))
                        - 4 * v1
                              * (-3 * (k[11] / omega_ab) + (k[14] / omega_ab)
                                 + 6 * (k[6] / omega_ab) * v2)
                        + 3 * v2sq
                              * ((k[7] / omega_ab) - (k[8] / omega_ab)
                                 - (k[9] / omega_ab))
                        + 4 * v2 * (3 * (k[12] / omega_ab) - (k[15] / omega_ab))
                     )
               + 216 * v0
                     * (4 * (k[10] / omega_ab) + 2 * (k[23] / omega_ab)
                        + v1sq
                              * (3 * (k[10] / omega_ab) - (k[13] / omega_ab)
                                 - 6 * (k[5] / omega_ab) * v2)
                        - 2 * v1
                              * (-4 * (k[16] / omega_ab) * v2
                                 + 2 * (k[22] / omega_ab)
                                 + 3 * (k[4] / omega_ab) * v2sq
                                 + 2 * (k[4] / omega_ab))
                        + v2sq * (3 * (k[10] / omega_ab) + (k[13] / omega_ab))
                        - 4 * v2 * ((k[21] / omega_ab) + (k[5] / omega_ab)))
               + 54 * v1sq
                     * (-2 * (k[17] / omega_ab) - 2 * (k[18] / omega_ab)
                        + 2 * (k[19] / omega_ab) - 2 * (k[7] / omega_ab)
                        + 2 * (k[8] / omega_ab) - 4 * (k[9] / omega_ab)
                        - 3 * v2sq
                              * ((k[7] / omega_ab) + (k[8] / omega_ab)
                                 + (k[9] / omega_ab))
                        + 4 * v2 * (3 * (k[12] / omega_ab) + (k[15] / omega_ab))
                     )
               + 216 * v1
                     * (4 * (k[11] / omega_ab) + 2 * (k[24] / omega_ab)
                        + v2sq * (3 * (k[11] / omega_ab) + (k[14] / omega_ab))
                        - 4 * v2 * ((k[20] / omega_ab) + (k[6] / omega_ab)))
               - 108 * v2sq
                     * ((k[17] / omega_ab) + (k[18] / omega_ab)
                        + (k[19] / omega_ab) + 2 * (k[8] / omega_ab)
                        + 2 * (k[9] / omega_ab))
               + 432 * v2 * (2 * (k[12] / omega_ab) + (k[25] / omega_ab)));
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