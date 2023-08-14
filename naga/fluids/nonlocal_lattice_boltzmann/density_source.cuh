
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

#include "detail/density_source.cuh"
#include "simulation_nodes.cuh"
#include "simulation_variables.cuh"

namespace naga::fluids::nonlocal_lbm {

template<class Lattice>
class density_source {
  public:
    using value_type = typename Lattice::value_type;

    virtual std::future<void> add_density_source(
        const simulation_nodes<const value_type>& domain,
        const problem_parameters<value_type>& params,
        const state_variables<Lattice>& state,
        const value_type& time,
        sclx::array<value_type, 1>& source_terms
    ) = 0;

    void unregister() {
        if (!registered_engine_) {
            return;
        }
        detail::unregister_density_source(*registered_engine_, this);
    }

    virtual ~density_source() { unregister(); }

  protected:
    friend class detail::simulation_engine<Lattice>;

    template<class Lattice_>
    friend void detail::unregister_density_source(
        detail::simulation_engine<Lattice_>& engine,
        density_source<Lattice_>* source
    );

    void notify_registered(detail::simulation_engine<Lattice>* engine) {
        if (registered_engine_) {
            sclx::throw_exception<std::runtime_error>(
                "Density source already registered with an engine.",
                "naga::fluids::nonlocal_lbm::density_source::"
            );
        }
        registered_engine_ = engine;
    }
    detail::simulation_engine<Lattice>* registered_engine_ = nullptr;
};

}  // namespace naga::fluids::nonlocal_lbm
