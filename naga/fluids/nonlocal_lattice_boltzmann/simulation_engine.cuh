
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

#include "detail/simulation_engine.cuh"

namespace naga::fluids::nonlocal_lbm {

template<class Lattice>
class simulation_engine {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    using lattice_type                 = Lattice;
    using simulation_domain_t          = simulation_nodes<const value_type>;
    using problem_parameters_t         = problem_parameters<value_type>;
    using solution_t                   = state_variables<Lattice>;

    simulation_engine() = default;

    void set_problem_parameters(
        value_type fluid_viscosity,
        value_type nominal_density,
        value_type time_step,
        value_type characteristic_length,
        value_type characteristic_velocity,
        value_type lattice_characteristic_velocity
    ) {
        engine_ptr_->set_problem_parameters(
            fluid_viscosity,
            nominal_density,
            time_step,
            characteristic_length,
            characteristic_velocity,
            lattice_characteristic_velocity
        );
    }

    void init_domain(const simulation_nodes<value_type>& domain) {
        engine_ptr_->init_domain(domain);
    }

    void init_domain(const node_provider<Lattice>& nodes) {
        engine_ptr_->init_domain(nodes.get());
    }

    void step_forward() { engine_ptr_->step_forward(); }

    void reset() { engine_ptr_->reset(); }

    void register_density_source(density_source<Lattice>& source) {
        engine_ptr_->register_density_source(source);
    }

    value_type time() const { return engine_ptr_->time(); }

    void attach_observer(simulation_observer<Lattice>& observer) {
        engine_ptr_->attach_observer(observer);
    }

    void detach_observer(simulation_observer<Lattice>& observer) {
        engine_ptr_->detach_observer(observer);
    }

    simulation_domain_t domain() const { return engine_ptr_->domain_; }

    const solution_t& solution() const { return engine_ptr_->solution_; }

    const problem_parameters_t& problem_parameters() const {
        return engine_ptr_->parameters_;
    }

    value_type speed_of_sound() const { return engine_ptr_->speed_of_sound(); }

    static value_type speed_of_sound(
        const value_type& characteristic_velocity,
        const value_type& lattice_characteristic_velocity
    ) {
        return detail::simulation_engine<Lattice>::speed_of_sound(
            characteristic_velocity,
            lattice_characteristic_velocity
        );
    }

  private:
    std::shared_ptr<detail::simulation_engine<Lattice>> engine_ptr_
        = std::make_shared<detail::simulation_engine<Lattice>>();
};

}  // namespace naga::fluids::nonlocal_lbm
