
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
#include "collision_term.cuh"
#include "pml_term.cuh"
#include <naga/nonlocal_calculus/advection.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

template<class>
struct pde_right_hand_side;

template<class T, uint Dimensions>
class pde_field_map {
  public:
    using point_type = point_t<T, Dimensions>;
    friend class advection_operator;

    pde_field_map(
        std::array<sclx::array<T, 1>, Dimensions + 1> rho_velocity,
        int quantity_index,
        T centering_offset = T(0)
    )
        : rho_velocity_(rho_velocity),
          quantity_index_(quantity_index),
          centering_offset_(centering_offset) {}

    __host__ __device__ point_type operator[](const sclx::index_t& index
    ) const {
        point_type field_value;
        T scalar = rho_velocity_[quantity_index_][index];
        for (uint i = 0; i < Dimensions; ++i) {
            field_value[i]
                = -rho_velocity_[i + 1][index] * (scalar - centering_offset_);
        }

        return field_value;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return rho_velocity_[0].elements();
    }

  private:
    std::array<sclx::array<T, 1>, Dimensions + 1> rho_velocity_;
    int quantity_index_;
    T centering_offset_;
};

template<class T>
struct pde_right_hand_side<d3q27_lattice<T>> {
    static constexpr auto dimensions
        = lattice_traits<d3q27_lattice<T>>::dimensions;
    static constexpr auto lattice_size = lattice_traits<d3q27_lattice<T>>::size;
    using value_type                   = T;
    using lattice                      = d3q27_lattice<T>;
    using interpolater_t = interpolation::radial_point_method<value_type>;

    pde_right_hand_side(
        simulation_nodes<T>& domain,
        problem_parameters<T>& parameters,
        state_variables<d3q27_lattice<T>>& state,
        naga::nonlocal_calculus::divergence_operator<T, dimensions>&
            divergence_op,
        interpolater_t& boundary_interpolater,
        simulation_engine<lattice>& engine
    )
        : domain_(&domain),
          parameters_(&parameters),
          state_(&state),
          divergence_op_(&divergence_op),
          boundary_interpolater_(&boundary_interpolater),
          engine_(&engine) {
        auto domain_size = state.lattice_distributions[0].elements();
        for (int i = 0; i < 1 + dimensions; ++i) {
            rates_[i] = sclx::array<T, 1>{domain_size};
        }
    }

    pde_right_hand_side()                                      = default;
    pde_right_hand_side(const pde_right_hand_side&)            = default;
    pde_right_hand_side(pde_right_hand_side&&)                 = default;
    pde_right_hand_side& operator=(const pde_right_hand_side&) = default;
    pde_right_hand_side& operator=(pde_right_hand_side&&)      = default;

    std::array<sclx::array<T, 1>, 1 + dimensions>& operator()(
        std::array<sclx::array<T, 1>, 1 + dimensions> rho_velocity,
        value_type t0,
        value_type dt
    ) {
        // simulation_engine<lattice>::compute_macroscopic_values(
        //     f,
        //     *state_,
        //     *parameters_
        // );

        auto& divergence_op      = divergence_op_;
        auto& rho_velocity_rates = rates_;

        auto boundary_start
            = domain_->num_bulk_points + domain_->num_layer_points;
        auto ghost_start
            = domain_->points.shape()[1] - domain_->num_ghost_nodes;
        auto boundary_end = domain_->points.shape()[1];

        auto engine   = engine_;
        auto rk4_step = rk4_step_;
        if (rk4_step == 4) {
            rk4_step  = 0;
            rk4_step_ = 0;
        }
        rk4_step_++;

        auto sum_weight   = runge_kutta_4::summation_weights[rk4_step];
        auto df_dt_weight = runge_kutta_4::df_dt_weights[rk4_step];

        auto homogeneous_future = std::async([engine,
                                              rho_velocity_rates,
                                              rho_velocity,
                                              t0,
                                              dt,
                                              divergence_op,
                                              boundary_start,
                                              ghost_start,
                                              boundary_end,
                                              sum_weight,
                                              df_dt_weight]() {
            std::vector<std::future<void>> futures;
            // engine->interpolate_boundaries(f);
            // engine->bounce_back_step(f);

            for (int d = 0; d < dimensions; ++d) {
                value_type c[3]{};
                c[d]             = 1;
                auto& rho_grad_d = engine->density_source_gradient_[d];
                if (rho_grad_d.elements() == 0) {
                    rho_grad_d
                        = sclx::array<T, 1>{engine->solution_.macroscopic_values
                                                .fluid_density.elements()};
                }
                using velocity_field_t = naga::nonlocal_calculus::
                    constant_velocity_field<value_type, dimensions>;
                auto velocity_field  = velocity_field_t::create(c);
                using field_map_type = typename naga::nonlocal_calculus::
                    advection_operator<value_type, dimensions>::
                        template divergence_field_map<velocity_field_t>;
                auto vector_field = field_map_type{
                    velocity_field,
                    engine->solution_.macroscopic_values.fluid_density,
                    engine->parameters_.nominal_density
                };
                // auto vector_field
                //     = field_map_type{velocity_field, density_source_term_};
                engine->divergence_op_.apply(vector_field, rho_grad_d);
            }

            sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
                sclx::array_list<value_type, 1, lattice_size> dist_result(
                    engine->solution_.lattice_distributions
                );

                auto f0 = engine->scratchpad1;
                auto f  = engine->scratchpad2;

                auto velocities
                    = engine->solution_.macroscopic_values.fluid_velocity;
                auto densities
                    = engine->solution_.macroscopic_values.fluid_density;

                auto density_scale = engine->parameters_.nominal_density;
                auto velocity_scale
                    = engine->parameters_.speed_of_sound
                    / lattice_traits<lattice_t>::lattice_speed_of_sound;
                auto lattice_speed_of_sound
                    = lattice_traits<lattice_t>::lattice_speed_of_sound;

                auto rho_grad          = engine->density_source_gradient_;
                auto lattice_time_step = engine->parameters_.lattice_time_step;

                handler.launch(
                    sclx::md_range_t<1>{dist_result[0].elements()},
                    dist_result,
                    [=] __device__(
                        const sclx::md_index_t<1>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= boundary_start) {
                            return;
                        }
                        value_type rho_grad_i[3]{
                            rho_grad[0][idx[0]] / density_scale,
                            rho_grad[1][idx[0]] / density_scale,
                            rho_grad[2][idx[0]] / density_scale
                        };
                        auto rho0 = densities[idx[0]] / density_scale;

                        auto rho = 0;
                        for (int alpha = 0; alpha < lattice_size; ++alpha) {
                            value_type c_alpha[3]{
                                lattice_interface<
                                    lattice_t>::lattice_velocities()
                                    .vals[alpha][0],
                                lattice_interface<
                                    lattice_t>::lattice_velocities()
                                    .vals[alpha][1],
                                lattice_interface<
                                    lattice_t>::lattice_velocities()
                                    .vals[alpha][2]
                            };
                            auto f0_alpha = f0[alpha][idx[0]];
                            auto f_alpha  = f[alpha][idx[0]];
                            auto dfalpha_dt
                                = naga::math::loopless::dot<3>(
                                      c_alpha,
                                      rho_grad_i
                                  )
                                * lattice_interface<lattice_t>::lattice_weights()
                                      .vals[alpha];
//                            if (idx[0] == 500) {
//                                printf(
//                                    "f0_alpha: %f, f_alpha: %f, dfalpha_dt: %f\n",
//                                    f0_alpha,
//                                    f_alpha,
//                                    dfalpha_dt
//                                );
//                            }
                            dist_result[alpha][idx[0]]
                                += dfalpha_dt * lattice_time_step * sum_weight;
                            f_alpha
                                = f0_alpha
                                + dfalpha_dt * lattice_time_step * df_dt_weight;
                            f[alpha][idx[0]] = f_alpha;
                            rho += f_alpha;
                        }

                        densities[idx[0]] = rho * density_scale;
                    }
                );
            }).get();

            // engine->interpolate_boundaries(rates);
        });

        // auto collision_future = (*collision_)();

        //        auto pml_future = (*pml_)(f, dt);

        homogeneous_future.get();
        // auto collision_terms = collision_future.get();
        //        auto pml_terms       = pml_future.get();

        // std::vector<std::future<void>> futures;
        // for (int i = 0; i < lattice_size; ++i) {
        // futures.push_back(sclx::algorithm::elementwise_reduce(
        // sclx::algorithm::plus<>(),
        // rates_[i],
        // rates_[i],
        // collision_terms[i]
        // ));
        // }

        return rates_;
    }

    void finalize(
        std::array<sclx::array<T, 1>, 1 + dimensions> /*unused*/,
        std::array<sclx::array<T, 1>, 1 + dimensions> f,
        value_type /*unused*/,
        value_type dt
    ) {
        //        simulation_engine<lattice>::compute_macroscopic_values(
        //            f,
        //            *state_,
        //            *parameters_
        //        );
        //        (*pml_)(f, dt, true).get();
    }

    // provided by engine
    simulation_nodes<T>* domain_;
    problem_parameters<T>* parameters_;
    state_variables<d3q27_lattice<T>>* state_;
    naga::nonlocal_calculus::divergence_operator<T, dimensions>* divergence_op_;
    interpolater_t* boundary_interpolater_;

    std::array<sclx::array<T, 1>, 1 + dimensions> rates_{};

    simulation_engine<lattice>* engine_;

    int rk4_step_ = 0;

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[4] = {1. / 2., 1. / 2., 1., 0.f};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };
};

}  // namespace naga::fluids::nonlocal_lbm::detail
