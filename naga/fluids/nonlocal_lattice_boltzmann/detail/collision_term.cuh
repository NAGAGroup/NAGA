
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
#include "../simulation_variables.cuh"
#include "lattices.cuh"

namespace naga::fluids::nonlocal_lbm::detail {

template<class>
struct collision_term;

template<class T>
struct collision_term<d3q27_lattice<T>> {
    static constexpr auto dimensions
        = lattice_traits<d3q27_lattice<T>>::dimensions;
    static constexpr auto lattice_size = lattice_traits<d3q27_lattice<T>>::size;
    using value_type                   = T;

    collision_term(
        problem_parameters<T>& parameters,
        state_variables<d3q27_lattice<T>>& state
    )
        : parameters_(&parameters),
          state_(&state) {
        auto domain_size = state.lattice_distributions[0].elements();
        for (int i = 0; i < lattice_size; ++i) {
            collision_terms_[i] = sclx::array<T, 1>{domain_size};
        }
    }

    std::future<std::array<sclx::array<T, 1>, lattice_size>> operator()() {
        auto domain_size = state_->lattice_distributions[0].elements();
        for (auto& term : collision_terms_) {
            sclx::fill(term, T{0});
        }
        auto& params          = *parameters_;
        auto& solution        = *state_;
        auto& collision_terms = this->collision_terms_;
        auto kernel_fut       = sclx::execute_kernel([domain_size,
                                                params,
                                                solution,
                                                collision_terms](
                                                   const sclx::kernel_handler&
                                                       handler
                                               ) {
            handler.launch(
                sclx::md_range_t<1>{domain_size},
                sclx::array_list<T, 1, lattice_size>(collision_terms),
                [=] __device__(
                    const sclx::md_index_t<1>& idx,
                    const sclx::kernel_info<>& info
                ) {
                    value_type k[lattice_size];

                    auto& lattice_distributions
                        = solution.lattice_distributions;

                    value_type f[lattice_size];
                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        f[alpha] = lattice_distributions[alpha][idx[0]];
                    }

                    value_type rho
                        = solution.macroscopic_values.fluid_density(idx[0])
                        / params.nominal_density;
                    value_type u[dimensions];
                    sclx::constexpr_assign_array<dimensions>(
                        u,
                        &solution.macroscopic_values.fluid_velocity(0, idx[0])
                    );
                    for (int d = 0; d < dimensions; ++d) {
                        u[d]
                            *= (lattice_traits<
                                          d3q27_lattice<T>>::lattice_speed_of_sound
                                / params.speed_of_sound);
                    }
                    value_type lat_nu = params.lattice_viscosity;
                    value_type lat_dt = params.lattice_time_step;

                    lattice_interface<d3q27_lattice<T>>::
                        compute_moment_projection(
                            k,
                            f,
                            rho,
                            u,
                            lat_nu,
                            lat_dt,
                            params.lattice_characteristic_frequency
                        );

                    auto K
                        = lattice_interface<d3q27_lattice<T>>::collision_matrix(
                        )
                              .vals;

                    for (int l = 0; l < lattice_size; ++l) {
                        value_type df_dt = 0;
                        for (int j = 0; j < lattice_size; ++j) {
                            df_dt += K[l][j] * k[j];
                        }
                        collision_terms[l][idx[0]] = df_dt;
                    }
                }
            );
        });

        return std::async(
            std::launch::deferred,
            [collision_terms, kernel_fut = std::move(kernel_fut)]() mutable {
                kernel_fut.wait();
                return collision_terms;
            }
        );
    }

    problem_parameters<T>* parameters_;
    state_variables<d3q27_lattice<T>>* state_;
    std::array<sclx::array<T, 1>, lattice_size> collision_terms_{};
};

}  // namespace naga::fluids::nonlocal_lbm::detail
