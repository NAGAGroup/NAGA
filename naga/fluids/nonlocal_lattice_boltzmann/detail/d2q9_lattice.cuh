
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

namespace naga::fluids::nonlocal_lbm {
template<class T>
struct d2q9_lattice;
}

namespace naga::fluids::nonlocal_lbm::detail {

template<class T>
struct d2q9_lattice_velocities : lattice_velocities_t<T, 2, 9> {
    T vals[9][2]
        = {{0, 0},
           {-1, 1},
           {-1, 0},
           {-1, -1},
           {0, -1},
           {1, -1},
           {1, 0},
           {1, 1},
           {0, 1}};
};

template<class T>
struct d2q9_lattice_weights : lattice_weights_t<T, 9> {
    T vals[9]
        = {4.0 / 9.0,
           1.0 / 36.0,
           1.0 / 9.0,
           1.0 / 36.0,
           1.0 / 9.0,
           1.0 / 36.0,
           1.0 / 9.0,
           1.0 / 36.0,
           1.0 / 9.0};
};

template<class T>
__constant__ T d2q9_collision_matrix_data[9][9]
    = {{1, 0, 0, -4, 0, 0, 0, 0, 4},
       {1, -1, 1, 2, 0, 1, -1, 1, 1},
       {1, -1, 0, -1, 1, 0, 0, -2, -2},
       {1, -1, -1, 2, 0, -1, 1, 1, 1},
       {1, 0, -1, -1, -1, 0, -2, 0, -2},
       {1, 1, -1, 2, 0, 1, 1, -1, 1},
       {1, 1, 0, -1, 1, 0, 0, 2, -2},
       {1, 1, 1, 2, 0, -1, -1, -1, 1},
       {1, 0, 1, -1, -1, 0, 2, 0, -2}};

template<class T>
struct d2q9_collision_matrix : collision_matrix_t<T, 9> {
    const decltype(d2q9_collision_matrix_data<T>)& vals
        = d2q9_collision_matrix_data<T>;
};

template<class T>
struct lattice_interface<d2q9_lattice<T>> {
    static constexpr uint size       = d2q9_lattice<T>::size;
    static constexpr uint dimensions = d2q9_lattice<T>::dimensions;
    using value_type                 = typename d2q9_lattice<T>::value_type;

    typedef enum {
        r  = 0,
        nw = 1,
        w  = 2,
        sw = 3,
        s  = 4,
        se = 5,
        e  = 6,
        ne = 7,
        n  = 8,
    } dir2d;

    static constexpr d2q9_lattice_velocities<T> lattice_velocities() {
        return d2q9_lattice_velocities<T>{};
    }

    static constexpr d2q9_lattice_weights<T> lattice_weights() {
        return d2q9_lattice_weights<T>{};
    }

    static constexpr d2q9_collision_matrix<T> collision_matrix() {
        return d2q9_collision_matrix<T>{};
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

        k[0] = value_type{0};
        k[1] = value_type{0};
        k[2] = value_type{0};

        k[3] = compute_k3(1.f / lat_dt, rho, u, f);
        k[4] = compute_k4(omega_ab, rho, u, f);
        k[5] = compute_k5(omega_ab, rho, u, f);
        k[6] = compute_k6(1.f / lat_dt, rho, u, f, k[3], k[4], k[5]);
        k[7] = compute_k7(1.f / lat_dt, rho, u, f, k[3], k[4], k[5]);
        k[8]
            = compute_k8(1.f / lat_dt, rho, u, f, k[3], k[4], k[5], k[6], k[7]);
    }

    static constexpr int get_bounce_back_idx(const int& alpha) {
        switch (alpha) {
        case dir2d::r: return dir2d::r;
        case dir2d::n: return dir2d::s;
        case dir2d::ne: return dir2d::sw;
        case dir2d::e: return dir2d::w;
        case dir2d::se: return dir2d::nw;
        case dir2d::s: return dir2d::n;
        case dir2d::sw: return dir2d::ne;
        case dir2d::w: return dir2d::e;
        case dir2d::nw: return dir2d::se;
        }
    }

  private:
    __host__ __device__ static value_type compute_k3(
        const value_type& omega_3,
        const value_type& density,
        const value_type* velocity,
        const value_type* f
    ) {
        return omega_3
             * (density
                    * (velocity[0] * velocity[0] + velocity[1] * velocity[1])
                - f[dir2d::e] - f[dir2d::n] - f[dir2d::s] - f[dir2d::w]
                - 2
                      * (f[dir2d::se] + f[dir2d::sw] + f[dir2d::ne]
                         + f[dir2d::nw] - density / 3.f))
             / 12.f;
    }

    __host__ __device__ static value_type compute_k4(
        const value_type& omega_4,
        const value_type& density,
        const value_type* velocity,
        const value_type* f
    ) {
        return omega_4
             * (f[dir2d::n] + f[dir2d::s] - f[dir2d::e] - f[dir2d::w]
                + density
                      * (velocity[0] * velocity[0] - velocity[1] * velocity[1]))
             / 4.f;
    }

    __host__ __device__ static value_type compute_k5(
        const value_type& omega_5,
        const value_type& density,
        const value_type* velocity,
        const value_type* f
    ) {
        return omega_5
             * (f[dir2d::ne] + f[dir2d::sw] - f[dir2d::nw] - f[dir2d::se]
                + density * (velocity[0] * velocity[1]))
             / 4.f;
    }

    __host__ __device__ static value_type compute_k6(
        const value_type& omega_6,
        const value_type& density,
        const value_type* velocity,
        const value_type* f,
        const value_type& k3,
        const value_type& k4,
        const value_type& k5
    ) {
        value_type term1
            = (f[dir2d::se] + f[dir2d::sw] - f[dir2d::ne] - f[dir2d::nw]
               - 2 * velocity[0] * velocity[0] * velocity[1] * density
               + velocity[1]
                     * (density - f[dir2d::n] - f[dir2d::s] - f[dir2d::r]))
            / 4.f;

        value_type term2
            = velocity[0] / 2.f
            * (f[dir2d::ne] - f[dir2d::nw] - f[dir2d::se] + f[dir2d::sw]);

        value_type term3
            = -velocity[1] / 2.f * (-3 * k3 - k4) - 2 * velocity[0] * k5;

        return -omega_6 * (term1 + term2) + term3;
    }

    __host__ __device__ static value_type compute_k7(
        const value_type& omega_7,
        const value_type& density,
        const value_type* velocity,
        const value_type* f,
        const value_type& k3,
        const value_type& k4,
        const value_type& k5
    ) {
        value_type term1
            = (f[dir2d::sw] + f[dir2d::nw] - f[dir2d::se] - f[dir2d::ne]
               - 2 * velocity[0] * velocity[1] * velocity[1] * density
               + velocity[0]
                     * (density - f[dir2d::w] - f[dir2d::e] - f[dir2d::r]))
            / 4.f;

        value_type term2
            = velocity[1] / 2.f
            * (f[dir2d::ne] + f[dir2d::sw] - f[dir2d::se] - f[dir2d::nw]);

        value_type term3
            = -velocity[0] / 2.f * (-3 * k3 + k4) - 2 * velocity[1] * k5;

        return -omega_7 * (term1 + term2) + term3;
    }

    __host__ __device__ static value_type compute_k8(
        const value_type& omega_8,
        const value_type& density,
        const value_type* velocity,
        const value_type* f,
        const value_type& k3,
        const value_type& k4,
        const value_type& k5,
        const value_type& k6,
        const value_type& k7
    ) {

        value_type term1 = density / 9.f - f[dir2d::ne] - f[dir2d::nw]
                         - f[dir2d::se] - f[dir2d::sw];
        value_type term2
            = 2
            * (velocity[0]
                   * (f[dir2d::ne] - f[dir2d::nw] + f[dir2d::se] - f[dir2d::sw])
               + velocity[1]
                     * (f[dir2d::ne] + f[dir2d::nw] - f[dir2d::se]
                        - f[dir2d::sw]));
        value_type term3
            = 4 * velocity[0] * velocity[1]
            * (f[dir2d::nw] - f[dir2d::ne] + f[dir2d::se] - f[dir2d::sw]);
        value_type term4 = -velocity[0] * velocity[0]
                         * (f[dir2d::n] + f[dir2d::ne] + f[dir2d::nw]
                            + f[dir2d::s] + f[dir2d::se] + f[dir2d::sw]);
        value_type term5 = velocity[1] * velocity[1]
                         * (3 * velocity[0] * velocity[0] * density
                            - f[dir2d::e] - f[dir2d::ne] - f[dir2d::nw]
                            - f[dir2d::se] - f[dir2d::sw] - f[dir2d::w]);

        value_type term6
            = -2 * k3 - 2 * velocity[0] * k7 - 2 * velocity[1] * k6
            + 4 * velocity[0] * velocity[1] * k5
            - (3.f / 2.f * k3 - k4 / 2.f)
                  * (velocity[0] * velocity[0] + velocity[1] * velocity[1]);

        return omega_8 * (1.f / 4.f * (term1 + term2 + term3 + term4 + term5))
             + term6;
    }
};

}  // namespace naga::fluids::nonlocal_lbm::detail