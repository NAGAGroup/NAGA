
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
#include "pml_absorption.cuh"
#include "pml_absorption.inl"

namespace naga::fluids::nonlocal_lbm::detail {

template<class>
struct pml_term;

template<class T>
struct pml_term<d3q27_lattice<T>> {
    static constexpr auto dimensions
        = lattice_traits<d3q27_lattice<T>>::dimensions;
    static constexpr auto lattice_size = lattice_traits<d3q27_lattice<T>>::size;
    using value_type                   = T;
    using lattice                      = d3q27_lattice<T>;

    pml_term(
        simulation_nodes<T>& domain,
        problem_parameters<T>& parameters,
        state_variables<d3q27_lattice<T>>& state,
        naga::nonlocal_calculus::divergence_operator<T, dimensions>&
            divergence_op
    )
        : domain_(&domain),
          parameters_(&parameters),
          state_(&state),
          divergence_op_(&divergence_op) {
        auto domain_size = state.lattice_distributions[0].elements();
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_tilde_eq_[alpha] = sclx::array<T, 1>{domain_size};
            Q1_[alpha]         = sclx::array<T, 1>{domain_size};
            sclx::fill(
                f_tilde_eq_[alpha],
                lattice_interface<d3q27_lattice<T>>::lattice_weights()
                    .vals[alpha]
            );
            sclx::fill(
                Q1_[alpha],
                lattice_interface<d3q27_lattice<T>>::lattice_weights()
                    .vals[alpha]
            );
        }
        for (auto& Q2_alpha : Q2_) {
            Q2_alpha = sclx::array<T, 1>{domain_size};
            sclx::fill(Q2_alpha, T{0});
        }
        for (auto& term : pml_terms_) {
            term = sclx::array<T, 1>{domain_size};
        }
        scratch_ = sclx::array<T, 1>{domain_size};
    }

    std::future<std::array<sclx::array<T, 1>, lattice_size>> operator()(
        const std::array<sclx::array<T, 1>, lattice_size>& f,
        value_type dt
    ) {
        auto& domain       = *domain_;
        auto& solution     = *state_;
        auto& params       = *parameters_;
        auto domain_size   = state_->lattice_distributions[0].elements();
        auto& f_tilde_eq   = f_tilde_eq_;
        auto& Q1           = Q1_;
        auto& Q2           = Q2_;
        auto& pml_terms    = pml_terms_;
        auto& scratch      = scratch_;
        auto divergence_op = divergence_op_;

        auto compute_fut = std::async(
            std::launch::async,
            [domain,
             solution,
             params,
             f,
             dt,
             f_tilde_eq,
             Q1,
             Q2,
             pml_terms,
             scratch,
             divergence_op]() mutable {
                auto domain_size = solution.lattice_distributions[0].elements();
                for (auto& term : pml_terms) {
                    sclx::fill(term, T{0});
                }
                auto velocity_scale
                    = params.speed_of_sound
                    / lattice_traits<lattice>::lattice_speed_of_sound;

                for (int alpha = 0; alpha < lattice_size; ++alpha) {
                    pml_div_Q1_field_map<lattice> field_map_Q1{
                        lattice_interface<lattice>::lattice_velocities()
                            .vals[alpha],
                        domain.layer_absorption,
                        Q1[alpha],
                        domain.num_bulk_points,
                        domain.num_bulk_points + domain.num_layer_points
                    };

                    divergence_op->apply(field_map_Q1, scratch);

                    sclx::algorithm::elementwise_reduce(
                        sclx::algorithm::plus<>(),
                        pml_terms[alpha],
                        pml_terms[alpha],
                        scratch
                    );

                    pml_div_Q2_field_map<value_type> field_map_Q2{
                        lattice_interface<lattice>::lattice_velocities()
                            .vals[alpha],
                        domain.layer_absorption,
                        Q2[alpha],
                        domain.num_bulk_points,
                        domain.num_bulk_points + domain.num_layer_points
                    };

                    divergence_op->apply(field_map_Q2, scratch);

                    sclx::algorithm::elementwise_reduce(
                        sclx::algorithm::plus<>(),
                        pml_terms[alpha],
                        pml_terms[alpha],
                        scratch
                    );
                }

                auto& absorption_coefficients = domain.layer_absorption;
                auto& num_layer_points        = domain.num_layer_points;
                auto& num_bulk_points         = domain.num_bulk_points;
                auto absorption_layer_start   = num_bulk_points;
                auto absorption_layer_end = num_bulk_points + num_layer_points;

                sclx::execute_kernel([=](const sclx::kernel_handler& handler) {
                    sclx::local_array<value_type, 2> lattice_velocities;
                    sclx::local_array<value_type, 1> lattice_weights;
                    handler.launch(
                        sclx::md_range_t<1>{domain_size},
                        sclx::array_list<T, 1, lattice_size>{pml_terms},
                        [=] __device__(
                            const sclx::md_index_t<1>& idx,
                            const sclx::kernel_info<>& info
                        ) mutable {
                            if (info.local_thread_linear_id() == 0) {
                                for (int i = 0; i < dimensions * lattice_size;
                                     ++i) {
                                    lattice_velocities(
                                        i % dimensions,
                                        i / dimensions
                                    )
                                        = lattice_interface<
                                              lattice>::lattice_velocities()
                                              .vals[i / dimensions]
                                                   [i % dimensions];

                                    if (i % dimensions == 0) {
                                        lattice_weights(i / dimensions)
                                            = lattice_interface<
                                                  lattice>::lattice_weights()
                                                  .vals[i / dimensions];
                                    }
                                }
                            }
                            handler.syncthreads();

                            value_type unitless_density
                                = solution.macroscopic_values.fluid_density[idx]
                                / params.nominal_density;
                            value_type unitless_velocity[dimensions];
                            for (uint d = 0; d < dimensions; ++d) {
                                unitless_velocity[d]
                                    = solution.macroscopic_values
                                          .fluid_velocity(d, idx[0])
                                    / velocity_scale;
                            }

                            for (uint alpha = 0; alpha < lattice_size;
                                 ++alpha) {
                                value_type f_tilde_eq_alpha_prev
                                    = f_tilde_eq[alpha][idx];

                                const value_type& Q1_alpha_prev
                                    = Q1[alpha][idx];

                                value_type f_tilde_eq_alpha
                                    = compute_equilibrium_distribution<lattice>(
                                          unitless_density,
                                          unitless_velocity,
                                          &lattice_velocities(0, alpha),
                                          lattice_weights(alpha)
                                      )
                                    - lattice_weights(alpha);

                                value_type Q1_alpha
                                    = (f_tilde_eq_alpha + f_tilde_eq_alpha_prev)
                                    * dt / 2.f;

                                value_type Q2_alpha
                                    = (Q1_alpha + Q1_alpha_prev) * dt / 2.f;

                                f_tilde_eq[alpha][idx] = f_tilde_eq_alpha;

                                Q1[alpha][idx] = Q1_alpha;

                                Q2[alpha][idx] = Q2_alpha;

                                if (idx[0] < absorption_layer_start
                                    || idx[0] >= absorption_layer_end) {
                                    continue;
                                }

                                const value_type& sigma
                                    = absorption_coefficients
                                        [idx[0] - absorption_layer_start];

                                using namespace math::loopless;

                                pml_terms[alpha][idx]
                                    -= sigma
                                     * (3.f * f_tilde_eq_alpha
                                        + 3.f * sigma * Q1_alpha
                                        + pow<2>(sigma) * Q2_alpha);
                            }
                        }
                    );
                });

                for (auto& term : pml_terms) {
                    auto bulk_portion
                        = term.get_range({0}, {absorption_layer_start});
                    sclx::fill(bulk_portion, T{0});
                    auto after_absorption_layer
                        = term.get_range({absorption_layer_end}, {domain_size});
                    sclx::fill(after_absorption_layer, T{0});
                }
            }
        );

        return std::async(
            std::launch::deferred,
            [pml_terms, compute_fut = std::move(compute_fut)]() mutable {
                compute_fut.wait();
                return pml_terms;
            }
        );
    }

    simulation_nodes<T>* domain_;
    problem_parameters<T>* parameters_;
    state_variables<d3q27_lattice<T>>* state_;
    naga::nonlocal_calculus::divergence_operator<T, dimensions>* divergence_op_;
    std::array<sclx::array<T, 1>, lattice_size> f_tilde_eq_{};
    std::array<sclx::array<T, 1>, lattice_size> Q1_{};
    std::array<sclx::array<T, 1>, lattice_size> Q2_{};
    std::array<sclx::array<T, 1>, lattice_size> pml_terms_{};
    sclx::array<T, 1> scratch_;
};

}  // namespace naga::fluids::nonlocal_lbm::detail
