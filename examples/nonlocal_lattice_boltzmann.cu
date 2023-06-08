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

#include "utils.hpp"
#include <naga/fluids/nonlocal_lattice_boltzmann.cuh>
#include <naga/regions/hypersphere.cuh>

using lattice_t = naga::fluids::nonlocal_lbm::d2q9_lattice<float>;
using sim_engine_t
    = naga::fluids::nonlocal_lbm::detail::simulation_engine<lattice_t>;

int main() {
    auto examples_path = get_examples_dir();
    auto results_path
        = get_examples_results_dir() / "nonlocal_lattice_boltzmann_results";

    sclx::filesystem::create_directories(results_path);

    sclx::filesystem::path domain_dir
        = examples_path
        / "../resources/lbm_example_domains/circles_in_rectangle";

    naga::fluids::nonlocal_lbm::boundary_specification<float> outer_boundary{
        domain_dir / "domain.obj",
        8,
        4,
        .01f,
    };

    std::vector<naga::fluids::nonlocal_lbm::boundary_specification<float>>
        inner_boundaries{
            {
                domain_dir / "circle1.obj",
                4,
                2,
                .01f,
            },
            {
                domain_dir / "circle2.obj",
                8,
                4,
                .01f,
            },
        };

    auto domain
        = naga::fluids::nonlocal_lbm::simulation_domain<float>::import <2>(
            outer_boundary,
            inner_boundaries,
            .05f
        );

    sim_engine_t engine;
    engine.set_problem_parameters(
        0.0f,
        1.0f,
        domain.nodal_spacing * domain.nodal_spacing,
        2.f,
        0.1f,
        0.1f
    );
    engine.init_domain(domain);
    engine.step_forward();
    engine.step_forward();

    std::ofstream domain_check_file(results_path / "domain_check.csv");
    domain_check_file << "x,y,nx,ny,absorption,ux,uy,rho";
    for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
        domain_check_file << ",f" << alpha;
    }
    domain_check_file << "\n";
    auto& f                = engine.solution_.lattice_distributions;
    auto& rho              = engine.solution_.macroscopic_values.fluid_density;
    auto& velocity         = engine.solution_.macroscopic_values.fluid_velocity;
    auto& layer_absorption = engine.domain_.layer_absorption;
    auto& normals          = engine.domain_.boundary_normals;
    auto& points           = engine.domain_.points;
    size_t num_bulk_points = engine.domain_.num_bulk_points;
    size_t num_layer_points    = engine.domain_.num_layer_points;
    size_t num_boundary_points = engine.domain_.num_boundary_points;
    for (size_t i = 0; i < domain.points.shape()[1]; ++i) {
        float absorption = 0;
        float normal[2]  = {0, 0};
        if (i >= num_bulk_points + num_layer_points) {
            normal[0] = normals(0, i - num_bulk_points - num_layer_points);
            normal[1] = normals(1, i - num_bulk_points - num_layer_points);
        } else if (i >= num_bulk_points) {
            absorption = layer_absorption(i - num_bulk_points);
        }
        domain_check_file << points(0, i) << "," << points(1, i) << ","
                          << normal[0] << "," << normal[1] << "," << absorption
                          << "," << velocity(0, i) << "," << velocity(1, i)
                          << "," << rho(i);
        for (const auto & f_alpha : f) {
            domain_check_file << "," << f_alpha(i);
        }
        domain_check_file << "\n";
    }
    domain_check_file.close();

    return 0;
}
