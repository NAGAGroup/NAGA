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

#include <naga/fluids/nonlocal_lbm_solver.cuh>

int main() {
    auto results_path = sclx::filesystem::path(__FILE__).parent_path()
                      / "nonlocal_lattice_boltzmann_results";

    sclx::filesystem::path domain_dir
        = results_path / "../../resources/lbm_example_domains/3_cube";

    naga::fluids::nonlocal_lbm::boundary_specification<float> domain_spec{
        domain_dir / "cube.obj",
        6,
        6,
        .03f};

    naga::fluids::nonlocal_lbm::boundary_specification<float> inner_ball_spec{
        domain_dir / "ball.obj",
        3,
        3,
        .03f};

    auto domain
        = naga::fluids::nonlocal_lbm::simulation_domain<float>::import <3>(
            domain_spec,
            {inner_ball_spec}
        );

    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "nonlocal_lattice_boltzmann_results.csv");
    file << "x,y,z,nx,ny,nz,type\n";
    for (size_t i = 0; i < domain.points.shape()[1]; ++i) {
        int type;
        if (i >= domain.num_bulk_points + domain.num_layer_points) {
            type = 3;
        } else if (i >= domain.num_bulk_points) {
            type = 2;
        } else {
            type = 1;
        }
        float normal[3]{};
        if (type == 3) {
            for (size_t j = 0; j < 3; ++j) {
                normal[j] = domain.boundary_normals(
                    j,
                    i - domain.num_bulk_points - domain.num_layer_points
                );
            }
        }
        file << domain.points(0, i) << "," << domain.points(1, i) << ","
             << domain.points(2, i) << "," << normal[0] << "," << normal[1]
             << "," << normal[2] << "," << type << "\n";
    }
    file.close();

    return 0;
}
