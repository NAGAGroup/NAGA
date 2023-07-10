
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

#include "../../../point.cuh"
#include "../lattices.cuh"
#include "subtask_factory.h"
#include <scalix/execute_kernel.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

template<class T, uint Dimensions>
class pml_div_Q1_field_map {
  public:
    using point_type = point_t<T, Dimensions>;

    __host__ __device__ pml_div_Q1_field_map(
        const T* c0,
        const sclx::array<const T, 1>& absorption_coeff,
        const sclx::array<const T, 1>& Q1,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
          Q1(Q1),
          pml_start_index(pml_start_index),
          pml_end_index(pml_end_index) {

        for (uint d = 0; d < Dimensions; d++) {
            this->c0[d] = c0[d];
        }
    }

    __host__ __device__ point_type operator[](sclx::index_t i) const {
        point_type c;
        if (i < pml_start_index || i >= pml_end_index) {
            for (uint d = 0; d < Dimensions; d++) {
                c[d] = 0.f;
            }
        } else {
            size_t pml_index = i - pml_start_index;
            T coeff          = absorption_coeff[pml_index];
            for (uint d = 0; d < Dimensions; d++) {
                c[d] = -c0[d] * coeff * Q1[pml_index];
            }
        }

        return c;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return absorption_coeff.elements();
    }

  private:
    T c0[Dimensions];
    sclx::array<const T, 1> absorption_coeff;
    sclx::array<const T, 1> Q1;
    size_t pml_start_index;
    size_t pml_end_index;
};

/**
 * @brief Adds the PML absorption term to distribution, sans divergence terms
 * @tparam Lattice
 */
template<class Lattice>
struct partial_pml_2d_absorption_subtask {
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
                                    )
                                  - lattice_weights(alpha);

            // note that lattice_pml_Q_values is also used to store the
            // previous value of f_tilde_eq
            const value_type& f_tilde_eq_prev
                = lattice_pml_Q_values[alpha][idx] - lattice_weights(alpha);

            value_type Q_value
                = (f_tilde_eq + f_tilde_eq_prev) * lattice_dt / 2.f;

            lattice_pml_Q_values[alpha][idx] = Q_value;

            if (idx[0] < absorption_layer_start
                || idx[0] >= absorption_layer_end) {
                continue;
            }

            const value_type& sigma
                = absorption_coefficients[idx[0] - absorption_layer_start];

            lattice_distributions[alpha][idx[0]]
                -= sigma * (2.f * f_tilde_eq + sigma * Q_value);
        }
    }
    sclx::kernel_handler handler;

    sclx::index_t absorption_layer_start{};
    sclx::index_t absorption_layer_end{};

    sclx::array_list<value_type, 1, lattice_size> lattice_distributions;
    sclx::array_list<value_type, 1, lattice_size> lattice_pml_Q_values;

    sclx::array<const value_type, 1> absorption_coefficients;
    sclx::array<const value_type, 1> fluid_density;
    sclx::array<const value_type, 2> fluid_velocity;

    sclx::local_array<value_type, 2> lattice_velocities;
    sclx::local_array<value_type, 1> lattice_weights;
    value_type density_scale;
    value_type velocity_scale;
    value_type lattice_dt;
};

}  // namespace naga::fluids::nonlocal_lbm::detail