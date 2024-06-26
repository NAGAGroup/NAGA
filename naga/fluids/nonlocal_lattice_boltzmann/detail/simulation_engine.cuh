
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
#include "pde_right_hand_side.cuh"
#include "pml_absorption.cuh"
#include "rk4_solver.cuh"
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
        value_type speed_of_sound,
        value_type characteristic_frequency = 0.f
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

        characteristic_frequency = (characteristic_frequency == 0)
                                     ? 1.f / time_step
                                     : characteristic_frequency;
        auto lattice_characteristic_frequency
            = characteristic_frequency * lattice_to_physical_time_ratio;
        parameters_.lattice_characteristic_frequency
            = lattice_characteristic_frequency;

        value_type signal_travel_per_frame = speed_of_sound * time_step;
        frames_for_node2node_travel_       = static_cast<std::uint32_t>(
            std::round(1.f / signal_travel_per_frame)
        );
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
        divergence_op_ = div_op_t::create(
            domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            ),
            domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            ),
            /*override_query_size*/ domain_.points.shape()[1]
        );
        // {
        //     auto boundary_div_operator = div_op_t::create(
        //         domain_.points.get_range(
        //             {0},
        //             {domain_.points.shape()[1] - domain_.num_ghost_nodes
        //              - domain_.num_boundary_points}
        //         ),
        //         domain_.points.get_range(
        //             {domain_.points.shape()[1] - domain_.num_ghost_nodes
        //              - domain_.num_boundary_points},
        //             {domain_.points.shape()[1] - domain_.num_ghost_nodes}
        //         )
        //     );
        //
        //     divergence_op_.set_subspace_to(
        //         boundary_div_operator,
        //         domain_.points.shape()[1] - domain_.num_ghost_nodes
        //             - domain_.num_boundary_points,
        //         domain_.points.shape()[1] - domain_.num_ghost_nodes
        //     );
        // }

        for (auto& f_alpha : solution_.lattice_distributions) {
            f_alpha = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        for (auto& tmp_alpha : scratchpad1) {
            tmp_alpha = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        for (auto& tmp_alpha : scratchpad2) {
            tmp_alpha = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }

        solution_.macroscopic_values.fluid_velocity
            = sclx::zeros<value_type, 2>(domain_.points.shape());
        solution_.macroscopic_values.fluid_density
            = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        sclx::fill(
            solution_.macroscopic_values.fluid_density,
            parameters_.nominal_density
        );

        {
            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            sclx::array<value_type, 2> bulk_points = domain_.points.get_range(
                {0},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes
                 - domain_.num_boundary_points}
            );
            auto boundary_points = domain_.points.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
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

        density_source_term_
            = sclx::zeros<value_type, 1>({domain_.points.shape()[1]});

        velocity_term_ = sclx::zeros<value_type, 2>(domain_.points.shape());

        auto absorption_coeffs    = domain_.layer_absorption;
        auto new_num_layer_points = domain_.num_layer_points
                                  + domain_.num_boundary_points
                                  + domain_.num_ghost_nodes;
        auto boundary_absorption = boundary_absorption_coefficient;
        sclx::array<value_type, 1> new_layer_absorption{{new_num_layer_points}};
        std::copy(
            absorption_coeffs.begin(),
            absorption_coeffs.begin() + domain_.num_layer_points,
            new_layer_absorption.begin()
        );
        std::fill(
            new_layer_absorption.begin() + domain_.num_layer_points,
            new_layer_absorption.begin() + domain_.num_layer_points
                + domain_.num_boundary_points,
            0.f
        );
        std::fill(
            new_layer_absorption.begin() + domain_.num_layer_points
                + domain_.num_boundary_points,
            new_layer_absorption.end(),
            0.f
        );

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

        auto rhs = std::make_shared<pde_right_hand_side<lattice_type>>(
            pde_right_hand_side<lattice_type>{
                domain_,
                parameters_,
                solution_,
                divergence_op_,
                *boundary_interpolator_ptr_,
                *this
            }
        );
        rk4_solver_ = std::make_shared<
            rk4_solver<pde_right_hand_side<lattice_type>, lattice_size>>(rhs);

        reset();
    }

    void init_domain(const node_provider<Lattice>& nodes) {
        this->init_domain(nodes.get());
    }

    value_type speed_of_sound() const { return parameters_.speed_of_sound; }

    void compute_density_source_terms(value_type t) {
        sclx::fill(
            density_source_term_,
            std::numeric_limits<value_type>::max()
        );

        for (density_source<Lattice>* source : density_sources_) {
            source
                ->add_density_source(
                    domain_,
                    parameters_,
                    solution_,
                    t,
                    density_source_term_
                )
                .get();
        }
    }

    static void compute_macroscopic_values(
        std::array<sclx::array<value_type, 1>, lattice_size> f,
        const state_variables<lattice_type>& solution,
        const problem_parameters<value_type>& parameters
    ) {
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
                    solution.macroscopic_values.fluid_density,
                    solution.macroscopic_values.fluid_velocity
                };

            handler.launch(
                sclx::md_range_t<1>(
                    solution.macroscopic_values.fluid_density.shape()
                ),
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
                            = f[alpha][idx[0]];
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
    void interpolate_boundaries(
        std::array<sclx::array<value_type, 1>, lattice_size> f
    ) {
        std::vector<std::future<void>> interpolation_futures;
        interpolation_futures.reserve(lattice_size * 3);
        auto lambda =
            [&, this](uint alpha, sclx::array<value_type, 1> lattice_quant) {
                sclx::array<value_type, 1> boundary_f_alpha
                    = lattice_quant.get_range(
                        {domain_.points.shape()[1] - domain_.num_boundary_points
                         - domain_.num_ghost_nodes},
                        {domain_.points.shape()[1] - domain_.num_ghost_nodes}
                    );
                const sclx::array<value_type, 1>& bulk_f_alpha
                    = lattice_quant.get_range(
                        {0},
                        {domain_.points.shape()[1] - domain_.num_boundary_points
                         - domain_.num_ghost_nodes}
                    );
                return boundary_interpolator_ptr_->interpolate(
                    bulk_f_alpha,
                    boundary_f_alpha,
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
                );
            };
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& f_alpha = f[alpha];
            interpolation_futures.emplace_back(lambda(alpha, f_alpha));
        }

        for (auto& fut : interpolation_futures) {
            fut.get();
        }
    }

    void collision_step(
        std::array<sclx::array<value_type, 1>, lattice_size> f0,
        std::array<sclx::array<value_type, 1>, lattice_size> f
    ) {
        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            sclx::array_list<value_type, 1, lattice_size> f_result(f);

            auto& parameters = parameters_;
            auto& solution   = solution_;

            handler.launch(
                sclx::md_range_t<1>{f[0].shape()[0]},
                f_result,
                [=] __device__(
                    const sclx::md_index_t<1>& idx,
                    const sclx::kernel_info<>& info
                ) {
                    value_type k[lattice_size];

                    value_type f0_i[lattice_size];
                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        f0_i[alpha] = f0[alpha][idx[0]];
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
                        f0_i,
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
                        f_result[l][idx[0]] += df_dt * lat_dt;
                    }
                }
            );
        }).get();
    }

    void apply_velocity_terms() {

        sclx::array_list<value_type, 1, lattice_size> result_f_boundary(
            solution_.lattice_distributions
        );

        auto bounce_back_lambda =
            [&](sclx::array_list<value_type, 1, lattice_size>& result_arrays) {
                return sclx::execute_kernel([&, result_arrays](
                                                sclx::kernel_handler& handler
                                            ) {
                    auto rho0 = parameters_.nominal_density;
                    auto velocity_scale
                        = parameters_.speed_of_sound
                        / lattice_traits<Lattice>::lattice_speed_of_sound;

                    auto boundary_velocities
                        = solution_.macroscopic_values.fluid_velocity;
                    auto boundary_densities
                        = solution_.macroscopic_values.fluid_density;

                    auto velocity_terms = velocity_term_;
                    sclx::local_array<value_type, 2> lattice_velocities(
                        handler,
                        {dimensions, lattice_size}
                    );
                    sclx::local_array<value_type, 1> lattice_weights(
                        handler,
                        {lattice_size}
                    );

                    auto lattice_time_step = parameters_.lattice_time_step;
                    auto lattice_viscosity = parameters_.lattice_viscosity;

                    auto lattice_speed_of_sound
                        = lattice_traits<Lattice>::lattice_speed_of_sound;

                    handler.launch(
                        sclx::md_range_t<1>{domain_.num_bulk_points},
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

                            const auto* velocity = &velocity_terms(0, idx[0]);
                            if (velocity[0]
                                == std::numeric_limits<value_type>::max()) {
                                return;
                            }
                            auto mag_velocity
                                = naga::math::loopless::norm<dimensions>(
                                    velocity
                                );

                            naga::point_t<value_type, dimensions> u_wall;
                            for (uint d = 0; d < dimensions; ++d) {
                                u_wall[d] = (boundary_velocities(d, idx[0])
                                             - velocity[d])
                                          / velocity_scale;
                            }
                            auto rho = boundary_densities[idx[0]] / rho0;

                            for (uint alpha = 0; alpha < lattice_size;
                                 ++alpha) {
                                const auto c_alpha
                                    = &lattice_velocities(0, alpha);
                                result_arrays[alpha][idx[0]]
                                    -= 2.f * rho
                                     * naga::math::loopless::dot<dimensions>(
                                           u_wall,
                                           c_alpha
                                     )
                                     / naga::math::loopless::pow<2>(
                                           lattice_speed_of_sound
                                     )
                                     * lattice_weights[alpha];
                            }
                        }
                    );
                });
            };

        auto f_fut = bounce_back_lambda(result_f_boundary);

        f_fut.get();
    }

    void compute_velocity_terms(value_type t) {
        sclx::fill(velocity_term_, std::numeric_limits<value_type>::max());
        for (auto& velocity_source : velocity_sources_) {
            velocity_source
                ->add_velocity_source(
                    domain_,
                    parameters_,
                    solution_,
                    t,
                    velocity_term_
                )
                .get();
        }
    }

    enum class bounce_back_config { half_step, full_step };

    void bounce_back_step_rates(
        std::array<sclx::array<value_type, 1>, lattice_size> rates
    ) {
        sclx::array<value_type, 1> f_boundary[lattice_size];
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_boundary[alpha] = rates[alpha].get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );
        }

        sclx::array_list<value_type, 1, lattice_size> result_f_boundary(
            f_boundary
        );
        auto boundary_normals = domain_.boundary_normals;
        auto rho0             = parameters_.nominal_density;
        auto velocity_scale   = parameters_.speed_of_sound
                            / lattice_traits<Lattice>::lattice_speed_of_sound;
        auto lattice_time_step = parameters_.lattice_time_step;

        auto boundary_velocities
            = solution_.macroscopic_values.fluid_velocity.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );
        auto boundary_densities
            = solution_.macroscopic_values.fluid_density.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );

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

                    auto lattice_time_step = parameters_.lattice_time_step;
                    auto lattice_viscosity = parameters_.lattice_viscosity;

                    auto lattice_speed_of_sound
                        = lattice_traits<Lattice>::lattice_speed_of_sound;

                    handler.launch(
                        sclx::md_range_t<1>{domain_.num_boundary_points},
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

                            naga::point_t<value_type, dimensions> u_wall;
                            naga::point_t<value_type, dimensions>
                                boundary_normal;
                            for (uint d = 0; d < dimensions; ++d) {
                                u_wall[d] = boundary_velocities(d, idx[0])
                                          / velocity_scale;
                                boundary_velocities(d, idx[0])
                                    = 0.f;  // reset the velocity
                                boundary_normal[d]
                                    = boundary_normals(d, idx[0]);
                            }
                            auto rho = boundary_densities[idx[0]] / rho0;

                            for (uint alpha = 0; alpha < lattice_size;
                                 ++alpha) {
                                const auto c_alpha
                                    = &lattice_velocities(0, alpha);
                                auto c_alpha_dot_n
                                    = naga::math::loopless::dot<dimensions>(
                                        c_alpha,
                                        boundary_normal
                                    );
                                if (c_alpha_dot_n <= 0) {
                                    continue;
                                }
                                result_arrays[alpha][idx[0]]
                                    = result_arrays[lattice_interface<
                                          lattice_t>::get_bounce_back_idx(alpha
                                      )][idx[0]]
                                    + 2.f * rho
                                          * naga::math::loopless::dot<
                                              dimensions>(u_wall, c_alpha)
                                          / naga::math::loopless::pow<2>(
                                              lattice_speed_of_sound
                                          )
                                          * lattice_weights[alpha];
                            }
                        }
                    );
                });
            };

        auto f_fut = bounce_back_lambda(result_f_boundary);

        f_fut.get();
    }

    void bounce_back_step(std::array<sclx::array<value_type, 1>, lattice_size> f
    ) {
        sclx::array<value_type, 1> f_boundary[lattice_size];
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_boundary[alpha] = f[alpha].get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );
        }

        sclx::array_list<value_type, 1, lattice_size> result_f_boundary(
            f_boundary
        );
        auto boundary_normals = domain_.boundary_normals;
        auto rho0             = parameters_.nominal_density;
        auto velocity_scale   = parameters_.speed_of_sound
                            / lattice_traits<Lattice>::lattice_speed_of_sound;

        auto boundary_velocities
            = solution_.macroscopic_values.fluid_velocity.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );
        auto boundary_densities
            = solution_.macroscopic_values.fluid_density.get_range(
                {domain_.points.shape()[1] - domain_.num_boundary_points
                 - domain_.num_ghost_nodes},
                {domain_.points.shape()[1] - domain_.num_ghost_nodes}
            );

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

                    auto lattice_time_step = parameters_.lattice_time_step;
                    auto lattice_viscosity = parameters_.lattice_viscosity;

                    auto lattice_speed_of_sound
                        = lattice_traits<Lattice>::lattice_speed_of_sound;

                    handler.launch(
                        sclx::md_range_t<1>{domain_.num_boundary_points},
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

                            naga::point_t<value_type, dimensions> u_wall;
                            naga::point_t<value_type, dimensions>
                                boundary_normal;
                            for (uint d = 0; d < dimensions; ++d) {
                                u_wall[d] = boundary_velocities(d, idx[0])
                                          / velocity_scale;
                                boundary_velocities(d, idx[0])
                                    = 0.f;  // reset the velocity
                                boundary_normal[d]
                                    = boundary_normals(d, idx[0]);
                            }
                            auto rho = boundary_densities[idx[0]] / rho0;

                            for (uint alpha = 0; alpha < lattice_size;
                                 ++alpha) {
                                const auto c_alpha
                                    = &lattice_velocities(0, alpha);
                                auto c_alpha_dot_n
                                    = naga::math::loopless::dot<dimensions>(
                                        c_alpha,
                                        boundary_normal
                                    );
                                if (c_alpha_dot_n <= 0) {
                                    continue;
                                }
                                result_arrays[alpha][idx[0]]
                                    = result_arrays[lattice_interface<
                                        lattice_t>::get_bounce_back_idx(alpha)]
                                                   [idx[0]];
                            }
                            // constexpr value_type tau = 2.f;
                            // naga::point_t<value_type, dimensions> u_wall;
                            // for (int d = 0; d < dimensions; ++d) {
                            //     u_wall[d] = boundary_velocities(d, idx[0])
                            //               / velocity_scale;
                            // }
                            // const auto rho  = boundary_densities[idx[0]] /
                            // rho0; decltype(u_wall) force_term; for (int d =
                            // 0; d < dimensions; ++d) {
                            //     force_term[d] = -rho * force_term[d];
                            //     boundary_velocities(d, idx[0]) /= 2.f;
                            // }
                            // for (int alpha = 0; alpha < lattice_size;
                            // ++alpha) {
                            //     value_type force_source = 0.f;
                            //     const auto c_alpha
                            //         = &lattice_velocities(0, alpha);
                            //     for (int d = 0; d < dimensions; ++d) {
                            //         force_source
                            //             += force_term[d] * (1. - 1. / tau /
                            //             2)
                            //              * lattice_weights[alpha]
                            //              * ((c_alpha[d] - u_wall[d])
                            //                     /
                            //                     naga::math::loopless::pow<2>(
                            //                         lattice_speed_of_sound
                            //                     )
                            //                 + naga::math::loopless::dot<
                            //                       dimensions>(c_alpha,
                            //                       u_wall) /
                            //                       naga::math::loopless::pow<
                            //                           4>(lattice_speed_of_sound)
                            //                       * c_alpha[d]);
                            //     }
                            //     result_arrays[alpha][idx[0]] += force_source;
                            // }
                        }
                    );
                });
            };

        auto f_fut = bounce_back_lambda(result_f_boundary);

        f_fut.get();
    }

    void apply_density_source_termsv3() {
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
            auto velocity_scale
                = parameters_.speed_of_sound
                / lattice_traits<lattice_type>::lattice_speed_of_sound;
            auto velocities = solution_.macroscopic_values.fluid_velocity;
            auto lattice_time_step = parameters_.lattice_time_step;

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

                    auto source_term = density_source_term[idx[1]];
                    if (source_term == std::numeric_limits<value_type>::max()) {
                        return;
                    }
                    auto imposed_density = 1.f + source_term / density_scale;
                    naga::point_t<value_type, dimensions> velocity_i;
                    naga::point_t<value_type, dimensions> c_alpha;
                    for (int d = 0; d < dimensions; ++d) {
                        velocity_i[d] = velocities(d, idx[1]) / velocity_scale;
                        c_alpha[d]    = lattice_interface<
                                            lattice_type>::lattice_velocities()
                                         .vals[idx[0]][d];
                    }
                    auto feq_alpha
                        = compute_equilibrium_distribution<lattice_type>(
                            imposed_density,
                            &velocity_i[0],
                            &c_alpha[0],
                            lattice_weights[idx[0]]
                        );

                    auto& f_alpha = result_arrays[idx[0]][idx[1]];
                    f_alpha       = feq_alpha;

                    densities[idx[1]] = imposed_density * density_scale;
                }
            );
        });
    }

    void apply_density_source_termsv4() {
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
            auto velocity_scale
                = parameters_.speed_of_sound
                / lattice_traits<lattice_type>::lattice_speed_of_sound;
            auto velocities = solution_.macroscopic_values.fluid_velocity;
            auto lattice_time_step = parameters_.lattice_time_step;

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

                    auto source_term = density_source_term[idx[1]];
                    if (source_term == std::numeric_limits<value_type>::max()) {
                        return;
                    }
                    auto imposed_density = 1.f + source_term / density_scale;
                    densities[idx[1]]    = imposed_density * density_scale;
                }
            );
        });
    }

    void apply_density_source_termsv5() {
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
            auto velocity_scale
                = parameters_.speed_of_sound
                / lattice_traits<lattice_type>::lattice_speed_of_sound;
            auto velocities = solution_.macroscopic_values.fluid_velocity;
            auto lattice_time_step = parameters_.lattice_time_step;

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

                    auto source_term = density_source_term[idx[1]];
                    if (source_term == std::numeric_limits<value_type>::max()) {
                        return;
                    }
                    auto imposed_density
                        = (densities[idx[1]] + source_term) / density_scale;
                    auto delta_rho = source_term / density_scale;
                    result_arrays[idx[0]][idx[1]]
                        += delta_rho * lattice_weights[idx[0]];

                    densities[idx[1]] += imposed_density * density_scale;
                    densities[idx[1]] /= 2;
                }
            );
        });
    }

    void apply_density_source_termsv2() {
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
            auto& densities        = solution_.macroscopic_values.fluid_density;
            auto lattice_time_step = parameters_.lattice_time_step;

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

                    auto source_term = density_source_term[idx[1]];
                    if (source_term == std::numeric_limits<value_type>::max()
                        || idx[0] == 0) {
                        return;
                    }

                    naga::point_t<value_type, dimensions> dipole_dir;
                    naga::point_t<value_type, dimensions> c_alpha;
                    for (int d = 0; d < dimensions; ++d) {
                        c_alpha[d] = lattice_interface<
                                         lattice_type>::lattice_velocities()
                                         .vals[idx[0]][d];
                        dipole_dir[d] = 0.f;
                    }
                    dipole_dir[0] = 1;
                    naga::math::loopless::normalize<dimensions>(c_alpha);
                    auto c_alpha_dot_dipole
                        = naga::math::loopless::dot<dimensions>(
                            c_alpha,
                            dipole_dir
                        );

                    auto& f_alpha = result_arrays[idx[0]];
                    f_alpha[idx[1]] += c_alpha_dot_dipole
                                     * lattice_weights[idx[0]] * source_term
                                     / density_scale;
                }
            );
        });
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
            auto& densities        = solution_.macroscopic_values.fluid_density;
            auto lattice_time_step = parameters_.lattice_time_step;

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

                    auto source_term = density_source_term[idx[1]];
                    if (source_term == std::numeric_limits<value_type>::max()) {
                        return;
                    }

                    // naga::point_t<value_type, dimensions> dipole_dir;
                    // naga::point_t<value_type, dimensions> c_alpha;
                    // for (int d = 0; d < dimensions; ++d) {
                    //     c_alpha[d] = lattice_interface<
                    //                      lattice_type>::lattice_velocities()
                    //                      .vals[idx[0]][d];
                    //     dipole_dir[d] = 0.f;
                    // }
                    // dipole_dir[0] = 1;
                    // if (idx[0] != 0) {
                    //     naga::math::loopless::normalize<dimensions>(c_alpha);
                    // }
                    // auto c_alpha_dot_dipole
                    //     = naga::math::loopless::dot<dimensions>(
                    //         c_alpha,
                    //         dipole_dir
                    //     );

                    auto& f_alpha = result_arrays[idx[0]];
                    f_alpha[idx[1]] += lattice_weights[idx[0]] * source_term
                                     / density_scale;
                }
            );
        });
    }

    void selective_filter_step(
        std::array<sclx::array<value_type, 1>, lattice_size> f0,
        std::array<sclx::array<value_type, 1>, lattice_size> f
    ) {
        auto support_indices = divergence_op_.support_indices();
        auto support_size    = support_indices.shape()[0];
        auto domain_points   = domain_.points;
        auto& delta_x        = domain_.nodal_spacing;
        auto gaussian_sigma  = delta_x / 2.5f;
        auto sigma           = .75f;

        bool compute_weights = false;
        if (filter_weights_.elements() == 0) {
            filter_weights_ = sclx::zeros<value_type, 2>(
                {support_size, domain_.points.shape()[1]}
            );
            compute_weights = true;
        }

        std::promise<void> capture_promise;
        auto capture_fut = capture_promise.get_future();

        auto bulk_end = domain_.num_bulk_points + domain_.num_layer_points;
        // bulk_end = domain_.num_bulk_points;

        auto fut = sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto f_old = f0;
            sclx::array_list<value_type, 1, lattice_size> f_new(f);
            auto filter_weights = filter_weights_;
            capture_promise.set_value();

            handler.launch(
                sclx::md_range_t<2>{lattice_size, domain_.points.shape()[1]},
                f_new,
                [=] __device__(const sclx::md_index_t<2>& idx, const sclx::kernel_info<2>&) mutable {
                    auto alpha = idx[0];
                    auto pidx  = idx[1];

                    if (pidx >= bulk_end) {
                        f_new[alpha][pidx] = f_old[alpha][pidx];
                        return;
                    }

                    const auto& x  = &domain_points(0, pidx);
                    auto f_alpha_i = f_old[alpha][pidx];
                    value_type sum = 0;
                    // uint num_summed = 0;
                    value_type sum_weights = 0;
                    for (int s = 0; s < support_size; ++s) {
                        const auto& x_s
                            = &domain_points(0, support_indices(s, pidx));
                        auto x2x_s_distance
                            = distance_functions::loopless::euclidean<
                                dimensions>{}(x, x_s);
                        value_type weight;
                        if (compute_weights) {
                            auto scale = naga::math::loopless::pow<dimensions>(
                                delta_x
                                / naga::math::
                                    sqrt(2 * naga::math::pi<value_type>)
                                / gaussian_sigma
                            );
                            weight = scale
                                   * naga::math::exp(
                                         -0.5
                                         * naga::math::loopless::pow<2>(
                                             x2x_s_distance / gaussian_sigma
                                         )
                                   );
                            filter_weights(s, pidx) = weight;
                            sum_weights += weight;
                        } else {
                            weight = filter_weights(s, pidx);
                        }
                        auto f_alpha_s = f_old[alpha][support_indices(s, pidx)];
                        sum += weight * (f_alpha_s - f_alpha_i);
                    }
                    for (int s = 0; compute_weights && s < support_size; ++s) {
                        filter_weights(s, pidx) /= sum_weights;
                        if (s == 0) {
                            sum /= sum_weights;
                        }
                    }
                    f_new[alpha][pidx] = f_alpha_i + sigma * sum;
                }
            );
        });

        capture_fut.get();
        fut.get();
    }

    void step_forward() {
        compute_macroscopic_values(
            solution_.lattice_distributions,
            solution_,
            parameters_
        );

        update_observers(time());

        compute_density_source_terms(time());
        // compute_velocity_terms(time());
        apply_density_source_termsv5();
        //         apply_velocity_terms();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            sclx::assign_array(
                solution_.lattice_distributions[alpha],
                scratchpad1[alpha]
            );
        }

        pml_absorption_operator_.apply();
        interpolate_boundaries(scratchpad1);

        // for (int alpha = 0; alpha < lattice_size; ++alpha) {
        //     sclx::assign_array(scratchpad1[alpha], scratchpad2[alpha]);
        // }

        collision_step(solution_.lattice_distributions, scratchpad1);
        selective_filter_step(scratchpad1, scratchpad2);

        auto f_old                      = solution_.lattice_distributions;
        auto f_new                      = scratchpad2;
        solution_.lattice_distributions = f_new;
        scratchpad2                     = f_old;

        // apply_density_source_termsv5();

        bounce_back_step(solution_.lattice_distributions);

        rk4_solver_->step_forward(
            solution_.lattice_distributions,
            time_ * parameters_.lattice_time_step / parameters_.time_step,
            parameters_.lattice_time_step
        );
        interpolate_boundaries(solution_.lattice_distributions);

        time_ += parameters_.time_step;

        ++frame_number_;
    }

    void reset() {
        frame_number_ = 0;
        time_         = 0;
        init_distribution();
        compute_macroscopic_values(
            solution_.lattice_distributions,
            solution_,
            parameters_
        );
    }

    value_type time() const { return time_; }

    void register_density_source(density_source<Lattice>& source) {
        density_sources_.push_back(&source);
        source.notify_registered(this);
    }

    void register_velocity_source(velocity_source<Lattice>& source) {
        velocity_sources_.push_back(&source);
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

    problem_parameters<value_type> parameters_{};
    state_variables<lattice_type> solution_{};
    simulation_nodes<value_type> domain_{};

    using interpolater_t = interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolater_t> boundary_interpolator_ptr_{};

    sclx::array<value_type, 1> density_source_term_{};
    sclx::array<value_type, 2> velocity_term_{};
    size_t frame_number_ = 0;
    value_type time_{0};
    size_t frames_for_node2node_travel_;
    std::array<sclx::array<value_type, 1>, lattice_size> scratchpad1{};
    std::array<sclx::array<value_type, 1>, lattice_size> scratchpad2{};
    sclx::array<value_type, 2> filter_weights_{};

    pml_absorption_operator<Lattice> pml_absorption_operator_{};

    std::vector<density_source<Lattice>*> density_sources_{};
    std::vector<velocity_source<Lattice>*> velocity_sources_{};

    std::vector<simulation_observer<Lattice>*> observers_{};

    nonlocal_calculus::divergence_operator<value_type, dimensions>
        divergence_op_{};

    std::shared_ptr<rk4_solver<pde_right_hand_side<lattice_type>, lattice_size>>
        rk4_solver_;

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

template<class Lattice>
void unregister_velocity_source(
    simulation_engine<Lattice>& engine,
    velocity_source<Lattice>* source
) {
    auto& sources = engine.velocity_sources_;
    auto it       = std::find(sources.begin(), sources.end(), source);
    if (it != sources.end()) {
        sources.erase(it);
    }
    source->registered_engine_ = nullptr;
}

}  // namespace naga::fluids::nonlocal_lbm::detail

#include "equilibrium_distribution.inl"
#include "pml_absorption.inl"
