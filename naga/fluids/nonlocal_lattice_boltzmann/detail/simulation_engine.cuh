
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

#include "../../../interpolation/radial_point_method.cuh"
#include "../../../nonlocal_calculus/advection.cuh"
#include "../density_source.cuh"
#include "../lattices.cuh"
#include "../node_provider.cuh"
#include "../simulation_nodes.cuh"
#include "../simulation_observer.cuh"
#include "../simulation_variables.cuh"
#include "equilibrium_distribution.cuh"
#include "pml_absorption.cuh"
#include "subtask_factory.h"
#include <scalix/fill.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

extern float boundary_absorption_coefficient;

template<class Lattice>
__device__ void compute_unitless_macroscopic_quants(
    typename lattice_traits<Lattice>::value_type& density_result,
    typename lattice_traits<Lattice>::value_type* velocity_result,
    sclx::local_array<typename lattice_traits<Lattice>::value_type, 2>&
        f_shared,
    const sclx::index_t linear_thread_id,
    sclx::local_array<typename lattice_traits<Lattice>::value_type, 2>&
        lattice_velocities
) {
    using value_type            = typename lattice_traits<Lattice>::value_type;
    constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    constexpr uint lattice_size = lattice_traits<Lattice>::size;

    auto& rho = density_result;
    auto& u   = velocity_result;
    for (uint d = 0; d < dimensions; ++d) {
        u[d] = 0;
    }
    for (uint alpha = 0; alpha < lattice_size; ++alpha) {
        rho += f_shared(alpha, linear_thread_id);
        for (uint d = 0; d < dimensions; ++d) {
            u[d] += f_shared(alpha, linear_thread_id)
                  * lattice_velocities(d, alpha);
        }
    }
    for (uint d = 0; d < dimensions; ++d) {
        u[d] /= rho;
    }
}

template<class Lattice>
class simulation_engine {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    using lattice_type                 = Lattice;

    simulation_engine() = default;

    void set_problem_parameters(
        value_type fluid_viscosity,
        value_type nominal_density,
        value_type time_step,
        value_type speed_of_sound
    ) {
        parameters_.fluid_viscosity = fluid_viscosity;
        parameters_.nominal_density = nominal_density;
        parameters_.time_step       = time_step;
        parameters_.speed_of_sound  = speed_of_sound;
        auto lattice_to_physical_time_ratio
            = lattice_traits<Lattice>::lattice_speed_of_sound / speed_of_sound;
        parameters_.lattice_time_step
            = time_step / lattice_to_physical_time_ratio;
        parameters_.lattice_viscosity
            = fluid_viscosity / lattice_to_physical_time_ratio;
    }

    void init_domain(const simulation_nodes<value_type>& domain) {
        domain_ = domain;
        if (domain_.points.shape()[0] != dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the simulation domain does not "
                "match the number of dimensions in the lattice.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        size_t total_points = domain_.num_bulk_points + domain_.num_layer_points
                            + domain_.num_boundary_points
                            + domain_.num_ghost_nodes;
        if (domain_.points.shape()[1] != total_points) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of points in the simulation domain does not match "
                "the sum of the number of bulk, layer, and boundary points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain_.boundary_normals.shape()[0] != dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the boundary normals does not "
                "match the number of dimensions in the lattice.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain_.boundary_normals.shape()[1]
            != domain_.num_boundary_points) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of boundary normals does not match the number of "
                "boundary points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain_.layer_absorption.elements()
            != domain_.num_layer_points + domain_.num_ghost_nodes) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of layer absorption coefficients does not match "
                "the number of layer points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }

        using div_op_t = naga::nonlocal_calculus::
            divergence_operator<value_type, dimensions>;
        auto divergence_operator = div_op_t::create(
            domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            ),
            domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes
                 - domain_.num_boundary_points}
            ),
            /*override_query_size*/ domain_.points.shape()[1]
        );
        {
            auto boundary_div_operator = div_op_t::create(
                domain_.points,
                domain_.points.get_range(
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes
                     - domain_.num_boundary_points},
                    {domain_.points.shape()[1]}
                )
            );

            divergence_operator.set_subspace_to(
                boundary_div_operator,
                domain_.points.shape()[1] - domain_.num_ghost_nodes
                    - domain_.num_boundary_points,
                domain_.points.shape()[1]
            );
        }

        advection_operator_ptr_ = std::make_shared<advection_operator_t>(
            advection_operator_t::create(divergence_operator)
        );

        {
            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            sclx::array<value_type, 2> bulk_points = domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes
                 - domain_.num_boundary_points}
            );
            sclx::array<value_type, 2> boundary_points
                = domain_.points.get_range(
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes
                     - domain_.num_boundary_points},
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes}
                );

            uint num_interp_points = 32;
            segmentation::nd_cubic_segmentation<value_type, dimensions>
                source_segmentation(bulk_points, num_interp_points);
            naga::default_point_map<value_type, dimensions> boundary_point_map{
                boundary_points
            };
            auto [distances_squared, indices]
                = naga::segmentation::batched_nearest_neighbors(
                    num_interp_points,
                    boundary_point_map,
                    source_segmentation
                );

            boundary_interpolator_ptr_ = std::make_shared<interpolater_t>(
                interpolater_t::create_interpolator(
                    bulk_points,
                    indices,
                    boundary_point_map,
                    domain_.nodal_spacing
                )
            );
        }

        if (domain_.num_ghost_nodes > 0) {
            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            sclx::array<value_type, 2> bulk_points = domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes}
            );
            sclx::array<value_type, 2> ghost_points = domain_.points.get_range(
                {domain_.points.shape()[1] - domain_.num_ghost_nodes},
                {domain_.points.shape()[1]}
            );

            uint num_interp_points = 32;
            segmentation::nd_cubic_segmentation<value_type, dimensions>
                source_segmentation(bulk_points, num_interp_points);
            naga::default_point_map<value_type, dimensions> ghost_point_map{
                ghost_points
            };
            auto [distances_squared, indices]
                = naga::segmentation::batched_nearest_neighbors(
                    num_interp_points,
                    ghost_point_map,
                    source_segmentation
                );

            ghost_interpolator_ptr_ = std::make_shared<interpolater_t>(
                interpolater_t::create_interpolator(
                    bulk_points,
                    indices,
                    ghost_point_map,
                    domain_.nodal_spacing
                )
            );
        }

        linalg::detail::batched_matrix_inverse_executor<
            value_type>::clear_problem_definitions();
        max_concurrency_ = std::min(
            std::thread::hardware_concurrency() / 2,
            max_concurrency_
        );
        advection_operator_ptr_->set_max_concurrent_threads(max_concurrency_);

        for (auto& f_alpha : solution_.lattice_distributions) {
            f_alpha = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        for (int t = 0; t < lattice_size; ++t) {
            temporary_distributions_[t]
                = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }

        solution_.macroscopic_values.fluid_velocity
            = sclx::zeros<value_type, 2>(domain_.points.shape());
        solution_.macroscopic_values.fluid_density
            = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        sclx::fill(
            solution_.macroscopic_values.fluid_density,
            parameters_.nominal_density
        );

        density_source_term_
            = sclx::zeros<value_type, 1>({domain_.points.shape()[1]});

        auto absorption_coeffs    = domain_.layer_absorption;
        auto new_num_layer_points = domain_.num_layer_points
                                  + domain_.num_boundary_points
                                  + domain_.num_ghost_nodes;
        auto boundary_absorption = boundary_absorption_coefficient;
        sclx::array<value_type, 1> new_layer_absorption{{new_num_layer_points}};
        //        std::copy(
        //            absorption_coeffs.begin(),
        //            absorption_coeffs.begin() + domain_.num_layer_points,
        //            new_layer_absorption.begin()
        //        );
        //        std::fill(
        //            new_layer_absorption.begin() + domain_.num_layer_points,
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points,
        //            boundary_absorption
        //        );
        //        std::fill(
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points,
        //            new_layer_absorption.end(),
        //            0.f
        //        );

        if (absorption_coeffs.elements() > 0) {
            std::copy(
                absorption_coeffs.begin(),
                absorption_coeffs.begin() + domain_.num_layer_points,
                new_layer_absorption.begin()
            );
        }
        std::fill(
            new_layer_absorption.begin() + domain_.num_layer_points,
            new_layer_absorption.end(),
            boundary_absorption
        );

        /*
         * Below is the version that makes ghost nodes nonzero absorption
         */
        //        std::copy(
        //            absorption_coeffs.begin(),
        //            absorption_coeffs.begin() + domain_.num_layer_points,
        //            new_layer_absorption.begin()
        //        );
        //        std::fill(
        //            new_layer_absorption.begin() + domain_.num_layer_points,
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points,
        //            boundary_absorption
        //        );
        //        std::copy(
        //            absorption_coeffs.begin() + domain_.num_layer_points,
        //            absorption_coeffs.end(),
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points
        //        );
        //        std::transform(
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points,
        //            new_layer_absorption.end(),
        //            new_layer_absorption.begin() + domain_.num_layer_points
        //                + domain_.num_boundary_points,
        //            [&](value_type val) { return boundary_absorption * val; }
        //        );

        domain_.layer_absorption = new_layer_absorption;
        domain_.num_layer_points = new_num_layer_points;

        pml_absorption_operator_ = pml_absorption_operator<Lattice>{this};

        // before the boundary condition change to use acoustic damping instead
        // of bounce back, the total number of points was num_bulk + num_layer +
        // num_boundary now, it is num_bulk + num_layer, so we need to subtract
        // num_boundary as observers still expect the old way, for the
        // absorption step it will be added back temporarily
        domain_.num_layer_points -= domain_.num_boundary_points;
        domain_.num_layer_points -= domain_.num_ghost_nodes;

        auto& points        = domain_.points;
        auto bulk_end       = domain_.num_bulk_points;
        auto boundary_start = points.shape()[1] - domain_.num_boundary_points
                            - domain_.num_ghost_nodes;
        auto boundary_end      = points.shape()[1] - domain_.num_ghost_nodes;
        auto ghost_start       = boundary_end;
        auto& boundary_normals = domain_.boundary_normals;

        //        // below we scale the absorption rates so that the same rate
        //        absorbs
        //        // similarly across all speeds of sounds. the scale value
        //        below was
        //        // found empirically and chosen such that, from a user's
        //        perspective
        //        // 0 is no absorption and 1 is max absorption
        //        value_type absorption_scale = speed_of_sound() * 3.f / 100.f;
        //        sclx::algorithm::transform(
        //            domain_.layer_absorption,
        //            domain_.layer_absorption,
        //            absorption_scale,
        //            sclx::algorithm::multiplies<>{}
        //        );

        reset();
    }

    void init_domain(const node_provider<Lattice>& nodes) {
        this->init_domain(nodes.get());
    }

    value_type speed_of_sound() const { return parameters_.speed_of_sound; }

    void compute_density_source_terms() {
        sclx::fill(density_source_term_, value_type{0});

        for (density_source<Lattice>* source : density_sources_) {
            source
                ->add_density_source(
                    domain_,
                    parameters_,
                    solution_,
                    time(),
                    density_source_term_
                )
                .get();
        }
    }

    void compute_macroscopic_values() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 2> f_shared(
                handler,
                {lattice_size,
                 sclx::cuda::traits::kernel::default_block_shape[0]}
            );
            sclx::local_array<value_type, 2> lattice_velocities(
                handler,
                {dimensions, lattice_size}
            );

            sclx::array_tuple<
                sclx::array<value_type, 1>,
                sclx::array<value_type, 2>>
                result_arrays{
                    solution_.macroscopic_values.fluid_density,
                    solution_.macroscopic_values.fluid_velocity
                };
            sclx::array<value_type, 1> lattice_distributions[lattice_size];
            for (int alpha = 0; alpha < lattice_size; ++alpha) {
                lattice_distributions[alpha]
                    = solution_.lattice_distributions[alpha];
            }
            auto parameters = parameters_;

            handler.launch(
                sclx::md_range_t<1>{domain_.points.shape()[1]},
                result_arrays,
                [=] __device__(
                    const sclx::md_index_t<1>& idx,
                    const sclx::kernel_info<>& info
                ) mutable {
                    if (info.local_thread_linear_id() == 0) {
                        for (int i = 0; i < dimensions * lattice_size; ++i) {
                            lattice_velocities(
                                i % dimensions,
                                i / dimensions
                            ) = lattice_interface<Lattice>::lattice_velocities()
                                    .vals[i / dimensions][i % dimensions];
                        }
                    }
                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        f_shared(alpha, info.local_thread_linear_id())
                            = lattice_distributions[alpha][idx[0]];
                    }
                    handler.syncthreads();

                    value_type rho{};
                    value_type u[dimensions]{};
                    compute_unitless_macroscopic_quants<Lattice>(
                        rho,
                        u,
                        f_shared,
                        info.local_thread_linear_id(),
                        lattice_velocities
                    );
                    thrust::get<0>(result_arrays)[idx[0]]
                        = rho * parameters.nominal_density;
                    for (int d = 0; d < dimensions; ++d) {
                        thrust::get<1>(result_arrays)(d, idx[0])
                            = u[d] * parameters.speed_of_sound
                            / lattice_traits<Lattice>::lattice_speed_of_sound;
                    }
                }
            );
        }).get();
    }

    void collision_step() {

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            sclx::array_list<value_type, 1, lattice_size> lattice_distributions(
                solution_.lattice_distributions
            );

            auto& parameters = parameters_;
            auto& solution   = solution_;

            handler.launch(
                sclx::md_range_t<1>{domain_.points.shape()[1]},
                lattice_distributions,
                [=] __device__(
                    const sclx::md_index_t<1>& idx,
                    const sclx::kernel_info<>& info
                ) {
                    value_type k[lattice_size];

                    value_type f[lattice_size];
                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        f[alpha] = lattice_distributions[alpha][idx[0]];
                    }

                    value_type rho
                        = solution.macroscopic_values.fluid_density(idx[0])
                        / parameters.nominal_density;
                    value_type u[dimensions];
                    sclx::constexpr_assign_array<dimensions>(
                        u,
                        &solution.macroscopic_values.fluid_velocity(0, idx[0])
                    );
                    for (int d = 0; d < dimensions; ++d) {
                        u[d]
                            *= (lattice_traits<Lattice>::lattice_speed_of_sound
                                / parameters.speed_of_sound);
                    }
                    value_type lat_nu = parameters.lattice_viscosity;
                    value_type lat_dt = parameters.lattice_time_step;

                    lattice_interface<Lattice>::compute_moment_projection(
                        k,
                        f,
                        rho,
                        u,
                        lat_nu,
                        lat_dt
                    );

                    auto K
                        = lattice_interface<Lattice>::collision_matrix().vals;

                    for (int l = 0; l < lattice_size; ++l) {
                        value_type df_dt = 0;
                        for (int j = 0; j < lattice_size; ++j) {
                            df_dt += K[l][j] * k[j];
                        }
                        lattice_distributions[l][idx[0]]
                            = f[l] + df_dt * lat_dt;
                    }
                }
            );
        }).get();
    }

    void interpolate_boundaries() {
        std::vector<std::future<void>> interpolation_futures;
        interpolation_futures.reserve(lattice_size);
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& f_alpha = solution_.lattice_distributions[alpha];
            sclx::array<value_type, 1> boundary_f_alpha = f_alpha.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );
            const sclx::array<value_type, 1>& bulk_f_alpha = f_alpha.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes}
            );
            interpolation_futures.emplace_back(
                boundary_interpolator_ptr_->interpolate(
                    bulk_f_alpha,
                    boundary_f_alpha,
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
                )
            );
        }

        for (auto& fut : interpolation_futures) {
            fut.get();
        }
    }

    void interpolate_ghost_nodes(bool interpolate_Q) {
        if (domain_.num_ghost_nodes == 0) {
            return;
        }
        std::vector<std::future<void>> interpolation_futures;
        interpolation_futures.reserve(lattice_size);
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& f_alpha = solution_.lattice_distributions[alpha];
            const sclx::array<value_type, 1>& bulk_f_alpha = f_alpha.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes}
            );
            const sclx::array<value_type, 1>& ghost_f_alpha = f_alpha.get_range(
                {domain_.points.shape()[1] - domain_.num_ghost_nodes},
                {domain_.points.shape()[1]}
            );
            interpolation_futures.emplace_back(
                ghost_interpolator_ptr_->interpolate(
                    bulk_f_alpha,
                    ghost_f_alpha,
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
                )
            );

            if (!interpolate_Q) {
                continue;
            }

            auto Q1_alpha = pml_absorption_operator_.get_Q1_values()[alpha];
            const sclx::array<value_type, 1>& bulk_Q1_alpha
                = Q1_alpha.get_range(
                    {0},
                    {domain_.points.shape()[1] - domain_.num_boundary_points
                     - domain_.num_ghost_nodes}
                );
            const sclx::array<value_type, 1>& ghost_Q1_alpha
                = Q1_alpha.get_range(
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes},
                    {domain_.points.shape()[1]}
                );
            interpolation_futures.emplace_back(
                ghost_interpolator_ptr_->interpolate(
                    bulk_Q1_alpha,
                    ghost_Q1_alpha,
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
                )
            );

            if constexpr (dimensions == 2) {
                continue;
            }

            auto Q2_alpha = pml_absorption_operator_.get_Q2_values()[alpha];
            const sclx::array<value_type, 1>& bulk_Q2_alpha
                = Q2_alpha.get_range(
                    {0},
                    {domain_.points.shape()[1] - domain_.num_boundary_points
                     - domain_.num_ghost_nodes}
                );
            const sclx::array<value_type, 1>& ghost_Q2_alpha
                = Q2_alpha.get_range(
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes},
                    {domain_.points.shape()[1]}
                );
            interpolation_futures.emplace_back(
                ghost_interpolator_ptr_->interpolate(
                    bulk_Q2_alpha,
                    ghost_Q2_alpha,
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
                )
            );
        }

        for (auto& fut : interpolation_futures) {
            fut.get();
        }
    }

    void bounce_back_step_boundary(bool bounce_back_Q) {
        sclx::array<value_type, 1> f_boundary[lattice_size];
        sclx::array<value_type, 1> pml_Q1_boundary[lattice_size];
        sclx::array<value_type, 1> pml_Q2_boundary[lattice_size];
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_boundary[alpha]
                = solution_.lattice_distributions[alpha].get_range(
                    {domain_.points.shape()[1] - domain_.num_boundary_points
                     - domain_.num_ghost_nodes},
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes}
                );
            pml_Q1_boundary[alpha]
                = pml_absorption_operator_.get_Q1_values()[alpha].get_range(
                    {domain_.points.shape()[1] - domain_.num_boundary_points
                     - domain_.num_ghost_nodes},
                    {domain_.points.shape()[1] - domain_.num_ghost_nodes}
                );
            if constexpr (dimensions == 3) {
                pml_Q2_boundary[alpha]
                    = pml_absorption_operator_.get_Q2_values()[alpha].get_range(
                        {domain_.points.shape()[1] - domain_.num_boundary_points
                         - domain_.num_ghost_nodes},
                        {domain_.points.shape()[1] - domain_.num_ghost_nodes}
                    );
            }
        }

        sclx::array_list<value_type, 1, lattice_size> result_f_boundary(
            f_boundary
        );
        sclx::array_list<value_type, 1, lattice_size> result_pml_Q1_boundary(
            pml_Q1_boundary
        );
        sclx::array_list<value_type, 1, lattice_size> result_pml_Q2_boundary(
            pml_Q2_boundary
        );
        auto boundary_normals = domain_.boundary_normals;

        auto bounce_back_lambda =
            [&](sclx::array_list<value_type, 1, lattice_size>& result_arrays) {
                return sclx::execute_kernel([&, result_arrays](
                                                sclx::kernel_handler& handler
                                            ) {
                    sclx::local_array<value_type, 2> lattice_velocities(
                        handler,
                        {dimensions, lattice_size}
                    );
                    sclx::local_array<value_type, 1> lattice_weights(
                        handler,
                        {lattice_size}
                    );

                    handler.launch(
                        sclx::md_range_t<1>{domain_.num_ghost_nodes},
                        result_arrays,
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
                                              Lattice>::lattice_velocities()
                                              .vals[i / dimensions]
                                                   [i % dimensions];
                                    if (i % dimensions == 0) {
                                        continue;
                                    }
                                    lattice_weights(i / dimensions)
                                        = lattice_interface<
                                              Lattice>::lattice_weights()
                                              .vals[i / dimensions];
                                }
                            }
                            handler.syncthreads();

                            value_type normal[dimensions]{};
                            for (int d = 0; d < dimensions; ++d) {
                                normal[d] = boundary_normals(d, idx[0]);
                            }

                            auto max_angle = .1f;
                            auto normal_norm
                                = math::loopless::norm<dimensions>(normal);
                            for (int alpha = 1; alpha < lattice_size; ++alpha) {
                                auto normal_dot_calpha
                                    = math::loopless::dot<dimensions>(
                                        normal,
                                        &lattice_velocities(0, alpha)
                                    );
                                //                                            auto
                                //                                            calpha_norm
                                //                                                    = math::loopless::norm<dimensions>(
                                //                                                            &lattice_velocities(0, alpha)
                                //                                                    );
                                //                                            calpha_norm
                                //                                                    = isnan(calpha_norm) ? 0 : calpha_norm;
                                //                                            auto
                                //                                            angle_normal_calpha
                                //                                                    = naga::math::abs(math::acos(
                                //                                                            normal_dot_calpha
                                //                                                            / (normal_norm * calpha_norm)
                                //                                                    ));
                                //                                            if
                                //                                            (angle_normal_calpha
                                //                                            <
                                //                                            max_angle
                                //                                                || (naga::math::pi<value_type>
                                //                                                    - angle_normal_calpha)
                                //                                                   < max_angle) {
                                //                                                continue;
                                //                                            }

                                if (normal_dot_calpha <= 0) {
                                    continue;
                                }
                                result_arrays[alpha][idx[0]]
                                    = result_arrays[lattice_interface<
                                        Lattice>::get_bounce_back_idx(alpha)]
                                                   [idx[0]];
                            }
                        }
                    );
                });
            };

        auto f_fut = bounce_back_lambda(result_f_boundary);

        if (!bounce_back_Q) {
            f_fut.get();
            return;
        }

        auto Q1_fut = bounce_back_lambda(result_pml_Q1_boundary);

        f_fut.get();
        Q1_fut.get();

        if constexpr (dimensions == 3) {
            auto Q2_fut = bounce_back_lambda(result_pml_Q2_boundary);
            Q2_fut.get();
        }
    }

    void bounce_back_step() {
        //        bounce_back_step_ghost();
        bounce_back_step_boundary(true);
    }

    void apply_density_source_terms() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );
            sclx::array_list<value_type, 1, lattice_size> result_arrays(
                solution_.lattice_distributions
            );
            auto& density_source_term = density_source_term_;
            auto& density_scale       = parameters_.nominal_density;
            auto& densities = solution_.macroscopic_values.fluid_density;

            handler.launch(
                sclx::md_range_t<2>{lattice_size, domain_.points.shape()[1]},
                result_arrays,
                [=] __device__(
                    const sclx::md_index_t<2>& idx,
                    const sclx::kernel_info<2>& info
                ) mutable {
                    if (info.local_thread_linear_id() == 0) {
                        for (int alpha = 0; alpha < lattice_size; ++alpha) {
                            lattice_weights[alpha]
                                = lattice_interface<
                                      lattice_type>::lattice_weights()
                                      .vals[alpha];
                        }
                    }
                    handler.syncthreads();

                    auto& f_alpha = result_arrays[idx[0]];
                    f_alpha[idx[1]] += lattice_weights[idx[0]]
                                     * density_source_term[idx[1]]
                                     / density_scale;
                    if (idx[0] == 0) {
                        densities[idx[1]] += density_source_term[idx[1]] / 2.f;
                    }
                }
            );
        });
    }

    void streaming_step() {
        using velocity_map = ::naga::nonlocal_calculus::
            constant_velocity_field<value_type, dimensions>;

        auto lattice_velocities
            = lattice_interface<Lattice>::lattice_velocities();
        auto lattice_weights = lattice_interface<Lattice>::lattice_weights();

        value_type time_step = parameters_.lattice_time_step / 2.f;

        size_t boundary_start = domain_.points.shape()[1]
                              - domain_.num_ghost_nodes
                              - domain_.num_boundary_points;

        std::vector<std::pair<int, std::future<void>>> streaming_futures(
            max_concurrency_
        );
        for (int alpha = 0; alpha < lattice_size; ++alpha) {

            auto velocity_map
                = velocity_map::create(&(lattice_velocities.vals[alpha][0]));

            auto& f_alpha0 = solution_.lattice_distributions[alpha];
            auto& f_alpha  = temporary_distributions_[alpha];

            value_type centering_offset = lattice_weights.vals[alpha];

            if (streaming_futures[alpha % max_concurrency_].second.valid()) {
                streaming_futures[alpha % max_concurrency_].second.get();
                sclx::assign_array(
                    temporary_distributions_
                        [streaming_futures[alpha % max_concurrency_].first],
                    solution_.lattice_distributions
                        [streaming_futures[alpha % max_concurrency_].first]
                );
            }
            auto fut = advection_operator_ptr_->step_forward(
                velocity_map,
                f_alpha0,
                f_alpha,
                time_step,
                centering_offset,
                boundary_start
            );
            streaming_futures[alpha % max_concurrency_]
                = {alpha, std::move(fut)};
        }

        for (auto& [alpha, fut] : streaming_futures) {
            if (!fut.valid()) {
                continue;
            }
            fut.get();
            sclx::assign_array(
                temporary_distributions_[alpha],
                solution_.lattice_distributions[alpha]
            );
        }
    }

    void step_forward() {
        compute_density_source_terms();
        apply_density_source_terms();

        compute_macroscopic_values();
        collision_step();

        interpolate_boundaries();
        compute_macroscopic_values();
        update_observers(time());

        interpolate_ghost_nodes(/*interpolate_Q=*/true);
        domain_.num_layer_points += domain_.num_boundary_points;
        domain_.num_layer_points += domain_.num_ghost_nodes;
        pml_absorption_operator_.apply();
        domain_.num_layer_points -= domain_.num_boundary_points;
        domain_.num_layer_points -= domain_.num_ghost_nodes;

        streaming_step();
        interpolate_boundaries();

        bounce_back_step();

        streaming_step();

        ++frame_number_;
    }

    void reset() {
        frame_number_ = 0;
        init_distribution();
        compute_macroscopic_values();
    }

    value_type time() const { return parameters_.time_step * frame_number_; }

    void register_density_source(density_source<Lattice>& source) {
        density_sources_.push_back(&source);
        source.notify_registered(this);
    }

    void init_distribution() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );
            sclx::array_list<value_type, 1, lattice_size> result_arrays(
                solution_.lattice_distributions
            );
            handler.launch(
                sclx::md_range_t<2>{lattice_size, domain_.points.shape()[1]},
                result_arrays,
                [=] __device__(
                    const sclx::md_index_t<2>& idx,
                    const sclx::kernel_info<2>& info
                ) mutable {
                    if (info.local_thread_linear_id() == 0) {
                        for (int alpha = 0; alpha < lattice_size; ++alpha) {
                            lattice_weights[alpha]
                                = lattice_interface<
                                      lattice_type>::lattice_weights()
                                      .vals[alpha];
                        }
                    }
                    handler.syncthreads();

                    auto& f_alpha   = result_arrays[idx[0]];
                    f_alpha[idx[1]] = lattice_weights[idx[0]];
                }
            );
        });
    }

    void attach_observer(simulation_observer<Lattice>& observer) {
        observers_.push_back(&observer);
    }

    void detach_observer(simulation_observer<Lattice>& observer) {
        observers_.erase(
            std::remove(observers_.begin(), observers_.end(), &observer),
            observers_.end()
        );
    }

    void update_observers(const value_type& time) {
        for (auto& observer : observers_) {
            observer->update(time, domain_, parameters_, solution_);
        }
    }

    ~simulation_engine() {
        for (auto& source : density_sources_) {
            source->registered_engine_ = nullptr;
        }
    }

    template<class Archive>
    void save_state(Archive& ar) const {
        ar(parameters_.fluid_viscosity,
           parameters_.time_step,
           parameters_.nominal_density,
           parameters_.speed_of_sound,
           parameters_.lattice_time_step,
           parameters_.lattice_viscosity);

        for (const auto& arr : solution_.lattice_distributions) {
            sclx::serialize_array(ar, arr);
        }
        sclx::serialize_array(ar, solution_.macroscopic_values.fluid_velocity);
        sclx::serialize_array(ar, solution_.macroscopic_values.fluid_density);

        sclx::serialize_array(ar, domain_.points);
        sclx::serialize_array(ar, domain_.boundary_normals);
        sclx::serialize_array(ar, domain_.layer_absorption);
        ar(domain_.num_bulk_points,
           domain_.num_layer_points,
           domain_.num_boundary_points,
           domain_.nodal_spacing,
           domain_.num_ghost_nodes);

        ar(*advection_operator_ptr_);
        ar(*boundary_interpolator_ptr_);
        ar(*ghost_interpolator_ptr_);

        sclx::serialize_array(ar, density_source_term_);
        for (auto& tmp : temporary_distributions_) {
            sclx::serialize_array(ar, tmp);
        }
        ar(frame_number_);

        ar(pml_absorption_operator_);
    }

    template<class Archive>
    void load_state(Archive& ar) {
        ar(parameters_.fluid_viscosity,
           parameters_.time_step,
           parameters_.nominal_density,
           parameters_.speed_of_sound,
           parameters_.lattice_time_step,
           parameters_.lattice_viscosity);

        for (auto& arr : solution_.lattice_distributions) {
            sclx::deserialize_array(ar, arr);
        }
        sclx::deserialize_array(
            ar,
            solution_.macroscopic_values.fluid_velocity
        );
        sclx::deserialize_array(ar, solution_.macroscopic_values.fluid_density);

        sclx::deserialize_array(ar, domain_.points);
        sclx::deserialize_array(ar, domain_.boundary_normals);
        sclx::deserialize_array(ar, domain_.layer_absorption);
        ar(domain_.num_bulk_points,
           domain_.num_layer_points,
           domain_.num_boundary_points,
           domain_.nodal_spacing,
           domain_.num_ghost_nodes);

        advection_operator_ptr_ = std::make_shared<advection_operator_t>();
        ar(*advection_operator_ptr_);
        boundary_interpolator_ptr_ = std::make_shared<interpolater_t>();
        ar(*boundary_interpolator_ptr_);
        ghost_interpolator_ptr_ = std::make_shared<interpolater_t>();
        ar(*ghost_interpolator_ptr_);

        sclx::deserialize_array(ar, density_source_term_);
        for (auto& tmp : temporary_distributions_) {
            sclx::deserialize_array(ar, tmp);
        }
        ar(frame_number_);

        pml_absorption_operator_ = pml_absorption_operator<Lattice>{this};
        ar(pml_absorption_operator_);
    }

    problem_parameters<value_type> parameters_{};
    state_variables<lattice_type> solution_{};
    simulation_nodes<value_type> domain_{};

    using advection_operator_t
        = nonlocal_calculus::advection_operator<value_type, dimensions>;
    std::shared_ptr<advection_operator_t> advection_operator_ptr_{};

    using interpolater_t = interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolater_t> boundary_interpolator_ptr_{};
    std::shared_ptr<interpolater_t> ghost_interpolator_ptr_{};

    sclx::array<value_type, 1> density_source_term_{};
    sclx::array<value_type, 1> temporary_distributions_[lattice_size]{};
    uint frame_number_ = 0;

    pml_absorption_operator<Lattice> pml_absorption_operator_{};

    std::vector<density_source<Lattice>*> density_sources_{};

    std::vector<simulation_observer<Lattice>*> observers_{};

    uint max_concurrency_ = lattice_size;
};

template<class Lattice>
void unregister_density_source(
    simulation_engine<Lattice>& engine,
    density_source<Lattice>* source
) {
    auto& sources = engine.density_sources_;
    auto it       = std::find(sources.begin(), sources.end(), source);
    if (it != sources.end()) {
        sources.erase(it);
    }
    source->registered_engine_ = nullptr;
}

}  // namespace naga::fluids::nonlocal_lbm::detail

#include "equilibrium_distribution.inl"
#include "pml_absorption.inl"
