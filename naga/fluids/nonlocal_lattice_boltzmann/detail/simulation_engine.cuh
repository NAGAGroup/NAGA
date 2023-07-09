
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
#include "../simulation_variables.cuh"
#include "naga/fluids/nonlocal_lattice_boltzmann/simulation_variables.cuh"
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
__device__ typename lattice_traits<Lattice>::value_type
compute_equilibrium_distribution(
    const typename lattice_traits<Lattice>::value_type& unitary_density,
    const typename lattice_traits<Lattice>::value_type* unitary_velocity,
    const typename lattice_traits<Lattice>::value_type* lattice_velocity,
    const typename lattice_traits<Lattice>::value_type& lattice_weight
) {
    using value_type          = typename lattice_traits<Lattice>::value_type;
    constexpr uint dimensions = lattice_traits<Lattice>::dimensions;

    value_type u_dot_u
        = math::loopless::dot<dimensions>(unitary_velocity, unitary_velocity);
    value_type u_dot_c
        = math::loopless::dot<dimensions>(unitary_velocity, lattice_velocity);

    constexpr value_type c_s = lattice_traits<Lattice>::lattice_speed_of_sound;

    constexpr auto pow2 = math::loopless::pow<2, value_type>;
    constexpr auto pow4 = math::loopless::pow<4, value_type>;

    return lattice_weight * unitary_density
         * (1 + u_dot_c / (pow2(c_s)) + pow2(u_dot_c) / (2 * pow4(c_s))
            - u_dot_u);
}

template<class Lattice>
struct equilibrium_distribution_kernel_body {
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

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_shared(alpha, info.local_thread_linear_id())
                = lattice_distributions[alpha][idx[0]];
        }

        value_type unitary_density = fluid_density[idx] / density_scale;
        value_type unitary_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitary_velocity[d] = fluid_velocity(d, idx[0]) / velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<Lattice>(
                unitary_density,
                unitary_velocity,
                &lattice_velocities(0, alpha),
                lattice_weights(alpha)
            );

            lattice_equilibrium_distributions[alpha][idx] = f_tilde_eq;
        }
    }
    sclx::kernel_handler handler;

    sclx::array_list<value_type, 1, lattice_size>
        lattice_equilibrium_distributions;

    sclx::array_list<const value_type, 1, lattice_size> lattice_distributions;

    sclx::array<const value_type, 1> fluid_density;
    sclx::array<const value_type, 2> fluid_velocity;

    sclx::local_array<value_type, 2> f_shared;
    sclx::local_array<value_type, 2> lattice_velocities;
    sclx::local_array<value_type, 1> lattice_weights;
    value_type density_scale;
    value_type velocity_scale;
};

template<class T, uint Dimensions>
class pml_emulated_velocity_t {
  public:
    using point_type = point_t<T, Dimensions>;

    __host__ __device__ pml_emulated_velocity_t(
        const T* c0,
        const sclx::array<T, 1>& absorption_coeff,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
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
                c[d] = c0[d] * coeff;
            }
        }

        return c;
    }

  private:
    T c0[Dimensions];
    sclx::array<T, 1> absorption_coeff;
    size_t pml_start_index;
    size_t pml_end_index;
};

template<class Lattice>
struct apply_pml_absorption_no_divQ {
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

        value_type unitary_density = fluid_density[idx] / density_scale;
        value_type unitary_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitary_velocity[d] = fluid_velocity(d, idx[0]) / velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<Lattice>(
                                        unitary_density,
                                        unitary_velocity,
                                        &lattice_velocities(0, alpha),
                                        lattice_weights(alpha)
                                    )
                                  - lattice_weights(alpha);

            // note that lattice_pml_Q_values is also used to store the
            // previous value of f_tilde_eq
            const value_type& f_tilde_eq_prev
                = lattice_pml_Q_values[alpha][idx]
                - lattice_weights(alpha);

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

    sclx::index_t absorption_layer_start;
    sclx::index_t absorption_layer_end;

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

template<class Lattice>
class simulation_engine {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    using lattice_type                 = Lattice;

    __host__ simulation_engine() = default;

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

    void init_domain(const simulation_nodes<value_type>& domain) {
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
            segmentation::nd_cubic_segmentation<value_type, dimensions>
                source_segmentation(bulk_points, num_interp_points);
            naga::default_point_map<value_type, dimensions> boundary_point_map{
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
        for (auto& Q_values : lattice_pml_Q_values_) {
            Q_values = sclx::array<value_type, 1>{domain_.points.shape()[1]};
        }
        temporary_distribution_
            = sclx::array<value_type, 1>{domain_.points.shape()[1]};
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

    void init_domain(const node_provider<Lattice>& nodes) {
        this->init_domain(nodes.get());
    }

    std::future<void> compute_density_source_terms() {
        sclx::fill(density_source_term_, value_type{0});
        std::future<void> fut = std::async(std::launch::async, []() {});

        for (density_source<Lattice>* source : density_sources_) {
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
                        / parameters.nondim_factors.density_scale;
                    value_type u[dimensions];
                    sclx::cexpr_memcpy<dimensions>(
                        u,
                        &solution.macroscopic_values.fluid_velocity(0, idx[0])
                    );
                    for (int d = 0; d < dimensions; ++d) {
                        u[d] /= parameters.nondim_factors.velocity_scale;
                    }
                    value_type lat_nu
                        = parameters.fluid_viscosity
                        / parameters.nondim_factors.viscosity_scale;
                    value_type lat_dt = parameters.time_step
                                      / parameters.nondim_factors.time_scale;

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

        if (domain_.num_layer_points == 0) {
            return;
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

            sclx::array_list<value_type, 1, lattice_size> lattice_distributions(
                solution_.lattice_distributions
            );
            sclx::array_list<value_type, 1, lattice_size> lattice_pml_Q_values(
                lattice_pml_Q_values_
            );

            sclx::array<value_type, 1> result_arrays_raw[2 * lattice_size];
            for (int alpha = 0; alpha < lattice_size; ++alpha) {
                result_arrays_raw[alpha]
                    = solution_.lattice_distributions[alpha];
                result_arrays_raw[alpha + lattice_size]
                    = lattice_pml_Q_values_[alpha];
            }
            sclx::array_list<value_type, 1, lattice_size> result_arrays(
                result_arrays_raw
            );

            apply_pml_absorption_no_divQ<Lattice> pml_kernel_body{
                handler,
                domain_.num_bulk_points,
                domain_.num_bulk_points + domain_.num_layer_points,
                lattice_distributions,
                lattice_pml_Q_values,
                domain_.layer_absorption,
                solution_.macroscopic_values.fluid_density,
                solution_.macroscopic_values.fluid_velocity,
                lattice_velocities,
                lattice_weights,
                parameters_.nondim_factors.density_scale,
                parameters_.nondim_factors.velocity_scale,
                parameters_.time_step / parameters_.nondim_factors.time_scale};

            handler.launch(
                sclx::md_range_t<1>{domain_.points.shape()[1]},
                result_arrays,
                pml_kernel_body
            );
        }).get();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            pml_emulated_velocity_t<value_type, dimensions> c_pml{
                lattice_interface<Lattice>::lattice_velocities().vals[alpha],
                domain_.layer_absorption,
                domain_.num_bulk_points,
                domain_.num_bulk_points + domain_.num_layer_points};

            sclx::assign_array(
                solution_.lattice_distributions[alpha],
                temporary_distribution_
            );

            advection_operator_ptr_->step_forward(
                c_pml,
                lattice_pml_Q_values_[alpha],
                temporary_distribution_,
                1.f
            );

            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::minus<>{},
                temporary_distribution_,
                temporary_distribution_,
                lattice_pml_Q_values_[alpha]
            );

            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>{},
                solution_.lattice_distributions[alpha],
                solution_.lattice_distributions[alpha],
                temporary_distribution_
            );
        }

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
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );

            std::vector<sclx::array<const value_type, 1>> lat_dist_vec(
                solution_.lattice_distributions,
                solution_.lattice_distributions + lattice_size
            );
            sclx::array_list<const value_type, 1, lattice_size>
                lattice_distributions(lat_dist_vec);
            sclx::array_list<value_type, 1, lattice_size>
                lattice_equilibrium_distributions(lattice_pml_Q_values_);

            equilibrium_distribution_kernel_body<Lattice> equilib_kernel_body{
                handler,
                lattice_equilibrium_distributions,
                lattice_distributions,
                solution_.macroscopic_values.fluid_density,
                solution_.macroscopic_values.fluid_velocity,
                f_shared,
                lattice_velocities,
                lattice_weights,
                parameters_.nondim_factors.density_scale,
                parameters_.nondim_factors.velocity_scale};

            handler.launch(
                sclx::md_range_t<1>{domain_.points.shape()[1]},
                lattice_equilibrium_distributions,
                equilib_kernel_body
            );
        }).get();
    }

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
                    lattice_interface<Lattice>::lattice_weights().vals[alpha]
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
                            lattice_velocities(
                                i % dimensions,
                                i / dimensions
                            ) = lattice_interface<Lattice>::lattice_velocities()
                                    .vals[i / dimensions][i % dimensions];
                            if (i % dimensions == 0) {
                                continue;
                            }
                            lattice_weights(i / dimensions)
                                = lattice_interface<Lattice>::lattice_weights()
                                      .vals[i / dimensions];
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
                        } else if (math::abs(math::loopless::dot<dimensions>(normal, &lattice_velocities(0, alpha))) < 1e-4) {
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
            auto& density_scale = parameters_.nondim_factors.density_scale;

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

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& time_scale   = parameters_.nondim_factors.time_scale;
            auto& length_scale = parameters_.nondim_factors.length_scale;
            value_type time_step
                = parameters_.time_step / time_scale * length_scale;

            auto velocity_map
                = velocity_map::create(&(lattice_velocities.vals[alpha][0]));

            auto& f_alpha0 = solution_.lattice_distributions[alpha];
            auto& f_alpha  = temporary_distribution_;

            value_type centering_offset = lattice_weights.vals[alpha];

            advection_operator_ptr_->step_forward(
                velocity_map,
                f_alpha0,
                f_alpha,
                time_step,
                centering_offset
            );
            sclx::assign_array(f_alpha, f_alpha0);
        }
    }

    void step_forward() {
        auto source_future = compute_density_source_terms();
        source_future.wait();
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

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            sclx::assign_array(
                solution_.lattice_distributions[alpha],
                lattice_pml_Q_values_[alpha]
            );
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

    using advection_operator_t
        = nonlocal_calculus::advection_operator<value_type, dimensions>;
    std::shared_ptr<advection_operator_t> advection_operator_ptr_{};

    using interpolater_t = interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolater_t> boundary_interpolator_ptr_{};

    sclx::array<value_type, 1> density_source_term_{};
    sclx::array<value_type, 1> temporary_distribution_{};
    sclx::array<value_type, 1> lattice_pml_Q_values_[lattice_size]{};
    uint frame_number_ = 0;

    std::vector<density_source<Lattice>*> density_sources_{};
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

// template class simulation_engine<d2q9_lattice<float>>;

}  // namespace naga::fluids::nonlocal_lbm::detail
