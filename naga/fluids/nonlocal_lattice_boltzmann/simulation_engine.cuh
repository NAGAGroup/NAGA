
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

#include "../../interpolation/radial_point_method.cuh"
#include "../../nonlocal_calculus/advection.cuh"
#include "lattices.cuh"
#include "simulation_domain.cuh"
#include <scalix/fill.cuh>

namespace naga::fluids::nonlocal_lbm {

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

        parameters_.nondim_factors.length_scale = characteristic_length;
        parameters_.nondim_factors.velocity_scale
            = characteristic_velocity / lattice_characteristic_velocity;
        parameters_.nondim_factors.time_scale
            = parameters_.nondim_factors.length_scale
            / parameters_.nondim_factors.velocity_scale;
        parameters_.nondim_factors.viscosity_scale
            = math::loopless::pow<2>(parameters_.nondim_factors.length_scale)
            / parameters_.nondim_factors.time_scale;

        parameters_.fluid_viscosity = fluid_viscosity;
        parameters_.nominal_density = nominal_density;
        parameters_.time_step       = time_step;
    }

    void init_domain(simulation_domain<value_type> domain) {
        domain_                 = domain;
        advection_operator_ptr_ = std::make_shared<advection_operator_t>(
            advection_operator_t::create(domain.points)
        );

        {
            uint num_interp_points = 32;
            segmentation::nd_cubic_segmentation<float, 2> source_segmentation(
                domain_.points,
                num_interp_points
            );

            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            const sclx::array<value_type, 2>& boundary_points
                = domain.points.get_range(
                    {domain.num_bulk_points + domain.num_layer_points},
                    {domain.points.shape()[1]}
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
                    domain.points,
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
            = sclx::array<value_type, 2>(domain_.points.shape());
        solution_.macroscopic_values.fluid_density
            = sclx::array<value_type, 1>{domain_.points.shape()[1]};
    }

  private:
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

    void init_distribution() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            sclx::local_array<value_type, 1> lattice_weights(
                handler,
                {lattice_size}
            );
            sclx::array_list<value_type, 1, lattice_size> result_arrays{
                solution_.lattice_distributions};
            handler.launch(
                sclx::md_range_t<2>{lattice_size, domain_.points.shape()[1]},
                result_arrays,
                [=](const sclx::md_index_t<2>& idx,
                    const sclx::kernel_info<2>& info) mutable {
                    if (info.local_thread_id().as_linear(
                            info.thread_block_shape()
                        )
                        == 0) {
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
};

template class simulation_engine<d2q9_lattice<float>>;

}  // namespace naga::fluids::nonlocal_lbm
