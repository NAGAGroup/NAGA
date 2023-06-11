
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
#include "../simulation_domain.cuh"
#include "../simulation_variables.cuh"
#include <scalix/fill.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
__device__ void compute_unitary_macroscopic_quants(
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
        value_type characteristic_length,
        value_type characteristic_velocity,
        value_type lattice_characteristic_velocity
    ) {
        parameters_.nondim_factors.density_scale = nominal_density;
        parameters_.nondim_factors.length_scale  = characteristic_length;
        parameters_.nondim_factors.velocity_scale
            = characteristic_velocity / lattice_characteristic_velocity;
        parameters_.nondim_factors.time_scale
            = parameters_.nondim_factors.length_scale
            / parameters_.nondim_factors.velocity_scale;
        parameters_.nondim_factors.viscosity_scale
            = math::loopless::pow<2>(parameters_.nondim_factors.length_scale)
            / parameters_.nondim_factors.time_scale;

        parameters_.fluid_viscosity = fluid_viscosity;
        parameters_.time_step       = time_step;
    }

    void init_domain(simulation_domain<value_type> domain) {
        domain_ = domain;
        if (domain_.points.shape()[0] != dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the simulation domain does not "
                "match the number of dimensions in the lattice.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        size_t total_points = domain.num_bulk_points + domain.num_layer_points
                            + domain.num_boundary_points;
        if (domain.points.shape()[1] != total_points) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of points in the simulation domain does not match "
                "the sum of the number of bulk, layer, and boundary points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain.boundary_normals.shape()[0] != dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the boundary normals does not "
                "match the number of dimensions in the lattice.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain.boundary_normals.shape()[1] != domain.num_boundary_points) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of boundary normals does not match the number of "
                "boundary points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }
        if (domain.layer_absorption.elements() != domain.num_layer_points) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of layer absorption coefficients does not match "
                "the number of layer points.",
                "naga::fluids::nonlocal_lbm::simulation_engine::"
            );
        }

        advection_operator_ptr_ = std::make_shared<advection_operator_t>(
            advection_operator_t::create(domain.points)
        );

        {
            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            sclx::array<value_type, 2> bulk_points = domain.points.get_range(
                {0},
                {domain.num_bulk_points + domain.num_layer_points}
            );
            sclx::array<value_type, 2> boundary_points
                = domain.points.get_range(
                    {domain.num_bulk_points + domain.num_layer_points},
                    {domain.points.shape()[1]}
                );

            uint num_interp_points = 32;
            segmentation::nd_cubic_segmentation<float, 2> source_segmentation(
                bulk_points,
                num_interp_points
            );
            naga::default_point_map<float, 2> boundary_point_map{
                boundary_points};
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
                    domain.nodal_spacing
                )
            );
        }

        for (auto& f_alpha : solution_.lattice_distributions) {
            f_alpha = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        for (auto& f_alpha_tmp : temporary_distributions_) {
            f_alpha_tmp = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        init_distribution();

        solution_.macroscopic_values.fluid_velocity
            = sclx::zeros<value_type, 2>(domain_.points.shape());
        solution_.macroscopic_values.fluid_density
            = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        sclx::fill(
            solution_.macroscopic_values.fluid_density,
            parameters_.nondim_factors.density_scale
        );

        density_source_term_
            = sclx::zeros<value_type, 1>({domain_.points.shape()[1]});
    }

    std::future<void> compute_density_source_terms() {
        sclx::fill(density_source_term_, value_type{0});
        std::future<void> fut = std::async(std::launch::async, []() {});

        for (density_source<value_type>* source : density_sources_) {
            fut.get();
            auto fut_new = source->add_density_source(
                domain_,
                parameters_,
                static_cast<value_type>(frame_number_) * parameters_.time_step,
                density_source_term_
            );
            fut = std::move(fut_new);
        }

        return fut;
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
                    solution_.macroscopic_values.fluid_velocity};
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
                            lattice_velocities(i % dimensions, i / dimensions)
                                = lattice_interface<
                                    Lattice>::lattice_velocities()[i];
                        }
                    }
                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        f_shared(alpha, info.local_thread_linear_id())
                            = lattice_distributions[alpha][idx[0]];
                    }
                    handler.syncthreads();

                    value_type rho{};
                    value_type u[dimensions]{};
                    compute_unitary_macroscopic_quants<Lattice>(
                        rho,
                        u,
                        f_shared,
                        info.local_thread_linear_id(),
                        lattice_velocities
                    );
                    thrust::get<0>(result_arrays)[idx[0]]
                        = rho * parameters.nondim_factors.density_scale;
                    for (int d = 0; d < dimensions; ++d) {
                        thrust::get<1>(result_arrays)(d, idx[0])
                            = u[d] * parameters.nondim_factors.velocity_scale;
                    }
                }
            );
        }).get();
    }

    void collision_step() {}

    void bounce_back_step() {
        std::vector<std::future<void>> interpolation_futures;
        interpolation_futures.reserve(lattice_size);
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& f_alpha = solution_.lattice_distributions[alpha];
            sclx::array<value_type, 1> boundary_f_alpha = f_alpha.get_range(
                {domain_.num_bulk_points + domain_.num_layer_points},
                {domain_.points.shape()[1]}
            );
            const sclx::array<value_type, 1>& bulk_f_alpha = f_alpha.get_range(
                {0},
                {domain_.num_bulk_points + domain_.num_layer_points}
            );
            interpolation_futures.emplace_back(
                boundary_interpolator_ptr_->interpolate(
                    bulk_f_alpha,
                    boundary_f_alpha,
                    lattice_interface<Lattice>::lattice_weights()[alpha]
                )
            );
        }

        for (auto& fut : interpolation_futures) {
            fut.get();
        }

        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 2> lattice_velocities(
                handler,
                {dimensions, lattice_size}
            );
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );

            sclx::array<value_type, 1> f_boundary[lattice_size];
            for (int alpha = 0; alpha < lattice_size; ++alpha) {
                f_boundary[alpha]
                    = solution_.lattice_distributions[alpha].get_range(
                        {domain_.num_bulk_points + domain_.num_layer_points},
                        {domain_.points.shape()[1]}
                    );
            }

            sclx::array_list<value_type, 1, lattice_size> result_arrays(
                f_boundary
            );
            auto boundary_normals = domain_.boundary_normals;

            handler.launch(
                sclx::md_range_t<1>{domain_.num_boundary_points},
                result_arrays,
                [=] __device__(
                    const sclx::md_index_t<1>& idx,
                    const sclx::kernel_info<>& info
                ) mutable {
                    if (info.local_thread_linear_id() == 0) {
                        for (int i = 0; i < dimensions * lattice_size; ++i) {
                            lattice_velocities(i % dimensions, i / dimensions)
                                = lattice_interface<
                                    Lattice>::lattice_velocities()[i];
                            if (i % dimensions == 0) {
                                continue;
                            }
                            lattice_weights(i / dimensions)
                                = lattice_interface<Lattice>::lattice_weights(
                                )[i / dimensions];
                        }
                    }
                    handler.syncthreads();

                    value_type normal[dimensions]{};
                    for (int d = 0; d < dimensions; ++d) {
                        normal[d] = boundary_normals(d, idx[0]);
                    }

                    for (int alpha = 0; alpha < lattice_size; ++alpha) {
                        if (math::loopless::dot<dimensions>(
                                normal,
                                &lattice_velocities(0, alpha)
                            )
                            < 0) {
                            continue;
                        } else if (math::loopless::dot<dimensions>(normal, &lattice_velocities(0, alpha)) == 0) {
                            result_arrays[alpha][idx[0]]
                                = lattice_weights(alpha);
                            continue;
                        }
                        result_arrays[alpha][idx[0]]
                            = result_arrays[lattice_interface<
                                Lattice>::get_bounce_back_idx(alpha)][idx[0]];
                    }
                }
            );
        }).get();
    }

    void apply_density_source_terms(std::future<void>&& source_future) {
        source_future.get();

        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );
            sclx::array_list<value_type, 1, lattice_size> result_arrays(
                solution_.lattice_distributions
            );
            auto& density_source_term = density_source_term_;

            handler.launch(
                sclx::md_range_t<2>{lattice_size, domain_.points.shape()[1]},
                result_arrays,
                [=] __device__(
                    const sclx::md_index_t<2>& idx,
                    const sclx::kernel_info<2>& info
                ) mutable {
                    if (info.local_thread_linear_id() == 0) {
                        for (int alpha = 0; alpha < lattice_size; ++alpha) {
                            lattice_weights[alpha] = lattice_interface<
                                lattice_type>::lattice_weights()[alpha];
                        }
                    }
                    handler.syncthreads();

                    auto& f_alpha = result_arrays[idx[0]];
                    f_alpha[idx[1]] += lattice_weights[idx[0]]
                                     * density_source_term[idx[1]];
                }
            );
        });
    }

    void streaming_step() {
        std::vector<std::future<void>> advection_futures;
        advection_futures.reserve(lattice_size);

        static value_type lattice_velocities[dimensions * lattice_size];
        static bool lattice_velocities_initialized = false;
        if (!lattice_velocities_initialized) {
            for (int alpha = 0; alpha < lattice_size; ++alpha) {
                for (int d = 0; d < dimensions; ++d) {
                    lattice_velocities[alpha * dimensions + d]
                        = lattice_interface<Lattice>::lattice_velocities(
                        )[alpha * dimensions + d];
                }
            }
            lattice_velocities_initialized = true;
        }

        using velocity_map = ::naga::nonlocal_calculus::
            constant_velocity_field<value_type, dimensions>;

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& time_scale   = parameters_.nondim_factors.time_scale;
            auto& length_scale = parameters_.nondim_factors.length_scale;
            value_type time_step
                = parameters_.time_step / time_scale * length_scale;
            const value_type* lat_vel = &lattice_velocities[alpha * dimensions];
            auto velocity_map         = velocity_map::create(lat_vel);
            auto& f_alpha0            = solution_.lattice_distributions[alpha];
            auto& f_alpha             = temporary_distributions_[alpha];
            value_type centering_offset
                = lattice_interface<Lattice>::lattice_weights()[alpha];
            advection_futures.emplace_back(std::async(
                std::launch::async,
                [this,
                 &velocity_map,
                 &f_alpha0,
                 &f_alpha,
                 time_step,
                 centering_offset,
                 alpha]() {
                    advection_operator_ptr_->step_forward(
                        velocity_map,
                        f_alpha0,
                        f_alpha,
                        time_step,
                        centering_offset
                    );
                    sclx::assign_array(
                        temporary_distributions_[alpha],
                        solution_.lattice_distributions[alpha]
                    );
                }
            ));
        }

        for (auto& advection_future : advection_futures) {
            advection_future.get();
        }
    }

    void step_forward() {
        auto source_future = compute_density_source_terms();
        compute_macroscopic_values();

        collision_step();
        bounce_back_step();

        apply_density_source_terms(std::move(source_future));

        streaming_step();

        ++frame_number_;
    }

    void reset() {
        frame_number_ = 0;
        init_distribution();
    }

    void add_density_source(density_source<value_type>& source) {
        density_sources_.push_back(&source);
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
                            lattice_weights[alpha] = lattice_interface<
                                lattice_type>::lattice_weights()[alpha];
                        }
                    }
                    handler.syncthreads();

                    auto& f_alpha   = result_arrays[idx[0]];
                    f_alpha[idx[1]] = lattice_weights[idx[0]];
                }
            );
        });
    }

    problem_parameters<value_type> parameters_{};
    state_variables<lattice_type> solution_{};
    simulation_domain<value_type> domain_{};

    using advection_operator_t
        = nonlocal_calculus::advection_operator<value_type, dimensions>;
    std::shared_ptr<advection_operator_t> advection_operator_ptr_{};

    using interpolater_t = interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolater_t> boundary_interpolator_ptr_{};

    sclx::array<value_type, 1> density_source_term_{};
    sclx::array<value_type, 1> temporary_distributions_[lattice_size]{};
    uint frame_number_ = 0;

    std::vector<density_source<value_type>*> density_sources_{};
};

}  // namespace naga::fluids::nonlocal_lbm::detail
