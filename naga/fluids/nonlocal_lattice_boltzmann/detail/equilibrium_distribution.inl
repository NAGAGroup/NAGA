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
class compute_equilibrium_subtask {

  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;

    __host__ compute_equilibrium_subtask(
        const simulation_engine<Lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::array_list<typename Lattice::value_type, 1, Lattice::size>&
            lattice_equilibrium_values
    )
        : lattice_equilibrium_distributions_(lattice_equilibrium_values) {

        lattice_distributions_ = sclx::array_list<value_type, 1, lattice_size>(
            engine.solution_.lattice_distributions
        );

        fluid_density_  = engine.solution_.macroscopic_values.fluid_density;
        fluid_velocity_ = engine.solution_.macroscopic_values.fluid_velocity;

        f_shared_ = sclx::local_array<value_type, 2>(
            handler,
            {lattice_size, sclx::cuda::traits::kernel::default_block_shape[0]}
        );
        lattice_velocities_ = sclx::local_array<value_type, 2>(
            handler,
            {dimensions, lattice_size}
        );
        lattice_weights_
            = sclx::local_array<value_type, 1>(handler, {lattice_size});

        density_scale_  = engine.parameters_.nominal_density;
        velocity_scale_ = engine.parameters_.speed_of_sound / lattice_traits<
            Lattice>::lattice_speed_of_sound;

        this->handler_ = handler;
    }

    __device__ void operator()(
        const sclx::md_index_t<1>& idx,
        const sclx::kernel_info<>& info
    ) {
        if (info.local_thread_linear_id() == 0) {
            for (int i = 0; i < dimensions * lattice_size; ++i) {
                lattice_velocities_(i % dimensions, i / dimensions)
                    = lattice_interface<Lattice>::lattice_velocities()
                          .vals[i / dimensions][i % dimensions];

                if (i % dimensions == 0) {
                    lattice_weights_(i / dimensions)
                        = lattice_interface<Lattice>::lattice_weights()
                              .vals[i / dimensions];
                }
            }
        }
        handler_.syncthreads();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            f_shared_(alpha, info.local_thread_linear_id())
                = lattice_distributions_[alpha][idx[0]];
        }

        value_type unitless_density = fluid_density_[idx] / density_scale_;
        value_type unitless_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitless_velocity[d] = fluid_velocity_(d, idx[0]) / velocity_scale_;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<Lattice>(
                unitless_density,
                unitless_velocity,
                &lattice_velocities_(0, alpha),
                lattice_weights_(alpha)
            );

            lattice_equilibrium_distributions_[alpha][idx] = f_tilde_eq;
        }
    }

    __host__ sclx::array_list<value_type, 1, lattice_size>& result() {
        return lattice_equilibrium_distributions_;
    }

  private:
    sclx::kernel_handler handler_;

    sclx::array_list<value_type, 1, lattice_size>
        lattice_equilibrium_distributions_;

    sclx::array_list<const value_type, 1, lattice_size> lattice_distributions_;

    sclx::array<const value_type, 1> fluid_density_;
    sclx::array<const value_type, 2> fluid_velocity_;

    sclx::local_array<value_type, 2> f_shared_;
    sclx::local_array<value_type, 2> lattice_velocities_;
    sclx::local_array<value_type, 1> lattice_weights_;
    value_type density_scale_;
    value_type velocity_scale_;
};

template<class Lattice>
compute_equilibrium_subtask<Lattice>
subtask_factory<compute_equilibrium_subtask<Lattice>>::create(
    const simulation_engine<Lattice>& engine,
    sclx::kernel_handler& handler,
    const sclx::array_list<typename Lattice::value_type, 1, Lattice::size>&
        lattice_equilibrium_values
) {
    return {engine, handler, lattice_equilibrium_values};
}

}  // namespace naga::fluids::nonlocal_lbm::detail