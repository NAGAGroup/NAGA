
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
        interpolater_t& boundary_interpolater
    )
        : domain_(&domain),
          parameters_(&parameters),
          state_(&state),
          divergence_op_(&divergence_op),
          boundary_interpolater_(&boundary_interpolater),
          collision_(
              std::make_shared<collision_term<lattice>>(parameters, state)
          ) {
        auto domain_size = state.lattice_distributions[0].elements();
        for (int i = 0; i < lattice_size; ++i) {
            rates_[i] = sclx::array<T, 1>{domain_size};
        }
    }

    pde_right_hand_side()                                      = default;
    pde_right_hand_side(const pde_right_hand_side&)            = default;
    pde_right_hand_side(pde_right_hand_side&&)                 = default;
    pde_right_hand_side& operator=(const pde_right_hand_side&) = default;
    pde_right_hand_side& operator=(pde_right_hand_side&&)      = default;

    std::array<sclx::array<T, 1>, lattice_size>& operator()(
        const std::array<sclx::array<T, 1>, lattice_size>& f,
        value_type t0,
        value_type dt
    ) {
        simulation_engine<lattice>::compute_macroscopic_values(
            f,
            *state_,
            *parameters_
        );

        auto& divergence_op = divergence_op_;
        auto& rates         = rates_;

        auto boundary_start
            = domain_->num_bulk_points + domain_->num_layer_points;
        auto ghost_start
            = domain_->points.shape()[1] - domain_->num_ghost_nodes;
        auto boundary_end = domain_->points.shape()[1];

        auto homogeneous_future = std::async([rates,
                                              f,
                                              divergence_op,
                                              boundary_start,
                                              ghost_start,
                                              boundary_end]() {
            for (int i = 0; i < lattice_size; ++i) {
                sclx::fill(rates[i], T{0});
                using velocity_field_t = naga::nonlocal_calculus::
                    constant_velocity_field<T, dimensions>;
                auto velocity_field = velocity_field_t::create(
                    lattice_interface<d3q27_lattice<T>>::lattice_velocities()
                        .vals[i]
                );
                using field_map_type = typename naga::nonlocal_calculus::
                    advection_operator<value_type, dimensions>::
                        template divergence_field_map<velocity_field_t>;
                auto vector_field = field_map_type{
                    velocity_field,
                    f[i],
                    lattice_interface<d3q27_lattice<T>>::lattice_weights()
                        .vals[i]
                };
                divergence_op->apply(vector_field, rates[i]);
            }
        });

        auto collision_future = (*collision_)();

//        auto pml_future = (*pml_)(f, dt);

        homogeneous_future.get();
        auto collision_terms = collision_future.get();
//        auto pml_terms       = pml_future.get();

        std::vector<std::future<void>> futures;
        for (int i = 0; i < lattice_size; ++i) {
            futures.push_back(sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>(),
                rates_[i],
                rates_[i],
                collision_terms[i]
            ));
        }

        for (int i = 0; i < lattice_size; ++i) {
            futures[i].get();
            auto ghost_rates
                = rates_[i].get_range({ghost_start}, {boundary_end});
            sclx::fill(ghost_rates, T{0});
        }

        return rates_;
    }

    void finalize(
        std::array<sclx::array<T, 1>, lattice_size> /*unused*/,
        std::array<sclx::array<T, 1>, lattice_size> f,
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

    // provided by this class
    std::shared_ptr<collision_term<lattice>> collision_;
    std::shared_ptr<pml_term<lattice>> pml_;

    std::array<sclx::array<T, 1>, lattice_size> rates_{};
};

}  // namespace naga::fluids::nonlocal_lbm::detail
