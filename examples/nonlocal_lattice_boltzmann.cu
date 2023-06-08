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

#include <naga/fluids/nonlocal_lattice_boltzmann.cuh>
#include <naga/regions/hypersphere.cuh>

int main() {
    auto examples_path = get_examples_results_dir();
    auto results_path  = examples_path / "nonlocal_lattice_boltzmann_results";

    naga::fluids::nonlocal_lbm::problem_parameters<float> metadata{};
    std::cout << metadata.nondim_factors.length_scale << std::endl;

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
            .01f
        );

    std::ofstream domain_file(results_path / "domain.csv");
    domain_file << "x,y,nx,ny,type,absorption\n";
    for (uint i = 0; i < domain.points.shape()[1]; ++i) {
        uint type;
        float absorption = 0;
        float normal[2]{0, 0};
        if (i >= domain.num_bulk_points + domain.num_layer_points) {
            type      = 3;
            normal[0] = domain.boundary_normals(
                0,
                i - domain.num_bulk_points - domain.num_layer_points
            );
            normal[1] = domain.boundary_normals(
                1,
                i - domain.num_bulk_points - domain.num_layer_points
            );
        } else if (i >= domain.num_bulk_points) {
            type       = 2;
            absorption = domain.layer_absorption[i - domain.num_bulk_points];
        } else {
            type = 1;
        }
        domain_file << domain.points(0, i) << "," << domain.points(1, i) << ","
                    << normal[0] << "," << normal[1] << "," << type << ","
                    << absorption << "\n";
    }

    auto domain_contour = naga::mesh::closed_contour_t<float>::import(
        outer_boundary.obj_file_path,
        true,
        0.01f
    );

    std::ofstream domain_contour_file(results_path / "domain_contour.csv");
    domain_contour_file << "x,y,nx,ny\n";
    for (uint i = 0; i < domain_contour.vertices.shape()[1]; ++i) {
        domain_contour_file << domain_contour.vertices(0, i) << ","
                            << domain_contour.vertices(1, i) << ","
                            << domain_contour.vertex_normals(0, i) << ","
                            << domain_contour.vertex_normals(1, i) << "\n";
    }
    domain_contour_file.close();

    return 0;
}
