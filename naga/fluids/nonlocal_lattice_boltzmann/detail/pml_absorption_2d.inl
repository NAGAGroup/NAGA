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

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
class partial_pml_2d_subtask {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;

    partial_pml_2d_subtask(
        const simulation_engine<Lattice>& engine,
        sclx::kernel_handler& handler
    ) {
        params_local_ = sclx::local_array<parameters, 1>(handler, {1});
        params_       = sclx::detail::make_unified_ptr(parameters{});
        *params_      = parameters(engine, handler);
    }

    __device__ void operator()(
        const sclx::md_index_t<1>& idx,
        const sclx::kernel_info<>& info
    ) {
        auto& params = params_local_[0];
// the following if/else macro prevents linting errors in IDEs
// since the return type is different for host and device
#ifdef __CUDA_ARCH__
        sclx::kernel_handler& handler = params_->handler;
#else
        sclx::kernel_handler handler;
#endif
        if (info.local_thread_linear_id() == 0) {
            params = *params_;
            for (int i = 0; i < dimensions * lattice_size; ++i) {
                params.lattice_velocities(i % dimensions, i / dimensions)
                    = lattice_interface<Lattice>::lattice_velocities()
                          .vals[i / dimensions][i % dimensions];

                if (i % dimensions == 0) {
                    params.lattice_weights(i / dimensions)
                        = lattice_interface<Lattice>::lattice_weights()
                              .vals[i / dimensions];
                }
            }
        }
        handler.syncthreads();

        value_type unitless_density
            = params.fluid_density[idx] / params.density_scale;
        value_type unitless_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitless_velocity[d]
                = params.fluid_velocity(d, idx[0]) / params.velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<Lattice>(
                                        unitless_density,
                                        unitless_velocity,
                                        &params.lattice_velocities(0, alpha),
                                        params.lattice_weights(alpha)
                                    )
                                  - params.lattice_weights(alpha);

            // note that lattice_Q1_values_ is also used to store the
            // previous value of f_tilde_eq
            const value_type& f_tilde_eq_prev
                = params.lattice_Q1_values[alpha][idx]
                - params.lattice_weights(alpha);

            value_type Q_value
                = (f_tilde_eq + f_tilde_eq_prev) * params.lattice_dt / 2.f;

            params.lattice_Q1_values[alpha][idx] = Q_value;

            if (idx[0] < params.absorption_layer_start
                || idx[0] >= params.absorption_layer_end) {
                continue;
            }

            const value_type& sigma
                = params.absorption_coefficients
                      [idx[0] - params.absorption_layer_start];

            params.lattice_distributions[alpha][idx[0]]
                -= sigma * (2.f * f_tilde_eq + sigma * Q_value);
        }
    }

    sclx::array_list<value_type, 1, 2 * lattice_size> result() const {
        sclx::array<value_type, 1> result_arrays_raw[2 * lattice_size];

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            result_arrays_raw[alpha] = params_->lattice_distributions[alpha];
            result_arrays_raw[alpha + lattice_size]
                = params_->lattice_Q1_values[alpha];
        }

        sclx::array_list<value_type, 1, 2 * lattice_size> result_arrays(
            result_arrays_raw
        );
        return result_arrays;
    }

    struct parameters {
        parameters() = default;

        parameters(
            const simulation_engine<Lattice>& engine,
            sclx::kernel_handler& handler
        ) {
            absorption_layer_start = engine.domain_.num_bulk_points;
            absorption_layer_end   = engine.domain_.num_bulk_points
                                 + engine.domain_.num_layer_points;

            lattice_distributions
                = sclx::array_list<value_type, 1, lattice_size>(
                    engine.solution_.lattice_distributions
                );

            lattice_Q1_values = sclx::array_list<value_type, 1, lattice_size>(
                engine
                    .lattice_equilibrium_values_  // temporarily use this array
                                                  // as Q1 values to save memory
            );

            absorption_coefficients = engine.domain_.layer_absorption;
            fluid_density  = engine.solution_.macroscopic_values.fluid_density;
            fluid_velocity = engine.solution_.macroscopic_values.fluid_velocity;

            lattice_velocities = sclx::local_array<value_type, 2>(
                handler,
                {dimensions, lattice_size}
            );
            lattice_weights
                = sclx::local_array<value_type, 1>(handler, {lattice_size});

            density_scale  = engine.parameters_.nondim_factors.density_scale;
            velocity_scale = engine.parameters_.nondim_factors.velocity_scale;
            lattice_dt     = engine.parameters_.time_step
                       / engine.parameters_.nondim_factors.time_scale;

            this->handler = handler;
        }

        sclx::kernel_handler handler;

        sclx::index_t absorption_layer_start{};
        sclx::index_t absorption_layer_end{};

        sclx::array_list<value_type, 1, lattice_size> lattice_distributions;
        sclx::array_list<value_type, 1, lattice_size> lattice_Q1_values;

        sclx::array<const value_type, 1> absorption_coefficients;
        sclx::array<const value_type, 1> fluid_density;
        sclx::array<const value_type, 2> fluid_velocity;

        sclx::local_array<value_type, 2> lattice_velocities;
        sclx::local_array<value_type, 1> lattice_weights;
        value_type density_scale;
        value_type velocity_scale;
        value_type lattice_dt;
    };

  private:
    sclx::detail::unified_ptr<parameters> params_;
    sclx::local_array<parameters, 1> params_local_;
};

template<class Lattice>
partial_pml_2d_subtask<Lattice>
subtask_factory<partial_pml_2d_subtask<Lattice>>::create(
    const simulation_engine<Lattice>& engine,
    sclx::kernel_handler& handler
) {
    return {engine, handler};
}

}  // namespace naga::fluids::nonlocal_lbm::detail