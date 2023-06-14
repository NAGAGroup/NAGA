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

using value_type = float;

using lattice_t = naga::fluids::nonlocal_lbm::d2q9_lattice<value_type>;
using sim_engine_t
    = naga::fluids::nonlocal_lbm::detail::simulation_engine<lattice_t>;

using density_source_t = naga::fluids::nonlocal_lbm::density_source<value_type>;
using simulation_domain_t
    = naga::fluids::nonlocal_lbm::simulation_domain<const value_type>;
using problem_parameters_t
    = naga::fluids::nonlocal_lbm::problem_parameters<value_type>;
using region_t = naga::regions::hypersphere<value_type, 2>;

class circular_init_peak : public density_source_t {
  public:
    std::future<void> add_density_source(
        const simulation_domain_t& domain,
        const problem_parameters_t& params,
        const value_type& time,
        sclx::array<value_type, 1>& source_terms
    ) final {
        region_t source_region{0.06, {-2.4, 0.0}};

        float period = 0.2f;
        if (time > period) {
            return std::async(std::launch::deferred, []() {});
        }

        return sclx::execute_kernel([=](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{source_terms.shape()},
                source_terms,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    if (source_region.contains(&domain.points(0, idx[0]))) {
                        auto distance
                            = naga::distance_functions::loopless::euclidean<
                                2>{}(
                                &domain.points(0, idx[0]),
                                source_region.center()
                            );
                        // gaussian peak as a function of distance from the
                        // center w/ width radius
                        source_terms(idx[0])
                            += 0.005f
                             * (std::sin(
                                 naga::math::pi<value_type> * time / period
                             ))
                             * exp(-distance * distance
                                   / (2.0 * source_region.radius()
                                      * source_region.radius()));
                    }
                }
            );
        });
    }
};

int main() {
    auto examples_path = get_examples_dir();
    auto results_path
        = get_examples_results_dir() / "nonlocal_lattice_boltzmann_results";

    sclx::filesystem::create_directories(results_path);

    sclx::filesystem::path domain_dir
        = examples_path
        / "../resources/lbm_example_domains/circles_in_rectangle";

    naga::fluids::nonlocal_lbm::boundary_specification<value_type>
        outer_boundary{
            domain_dir / "domain.obj",
            8,
            4,
            .01f,
        };

    std::vector<naga::fluids::nonlocal_lbm::boundary_specification<value_type>>
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
        = naga::fluids::nonlocal_lbm::simulation_domain<value_type>::import <2>(
            outer_boundary,
            inner_boundaries,
            .005f
        );

    sim_engine_t engine;
    engine.set_problem_parameters(
        0.0f,
        1.0f,
        2 * domain.nodal_spacing * domain.nodal_spacing,
        2.f,
        0.1f,
        0.1f
    );
    engine.init_domain(domain);
    circular_init_peak source{};
    engine.register_density_source(source);

    int frames = 100;
    std::mutex frame_mutex;
    std::chrono::milliseconds total_time{0};
    for (int frame = 0; frame < frames; ++frame) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.step_forward();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        );
        std::thread([&] {
            std::lock_guard<std::mutex> lock(frame_mutex);
            std::cout << "Time: "
                      << engine.frame_number_ * engine.parameters_.time_step
                      << "\n";
        }).detach();

        if (frame % 100 != 0) {
            continue;
        }

        std::ofstream domain_check_file(
            results_path
            / (std::string("result.csv.") + std::to_string(frame / 100))
        );
        domain_check_file << "x,y,nx,ny,absorption,ux,uy,rho";
        for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
            domain_check_file << ",f" << alpha;
        }
        domain_check_file << "\n";
        auto& f        = engine.solution_.lattice_distributions;
        auto& rho      = engine.solution_.macroscopic_values.fluid_density;
        auto& velocity = engine.solution_.macroscopic_values.fluid_velocity;
        auto& layer_absorption     = engine.domain_.layer_absorption;
        auto& normals              = engine.domain_.boundary_normals;
        auto& points               = engine.domain_.points;
        size_t num_bulk_points     = engine.domain_.num_bulk_points;
        size_t num_layer_points    = engine.domain_.num_layer_points;
        size_t num_boundary_points = engine.domain_.num_boundary_points;
        for (size_t i = 0; i < domain.points.shape()[1]; ++i) {
            value_type absorption = 0;
            value_type normal[2]  = {0, 0};
            if (i >= num_bulk_points + num_layer_points) {
                normal[0] = normals(0, i - num_bulk_points - num_layer_points);
                normal[1] = normals(1, i - num_bulk_points - num_layer_points);
            } else if (i >= num_bulk_points) {
                absorption = layer_absorption(i - num_bulk_points);
            }
            domain_check_file << points(0, i) << "," << points(1, i) << ","
                              << normal[0] << "," << normal[1] << ","
                              << absorption << "," << velocity(0, i) << ","
                              << velocity(1, i) << "," << rho(i);
            for (const auto& f_alpha : f) {
                domain_check_file << "," << f_alpha(i);
            }
            domain_check_file << "\n";
        }
        domain_check_file.close();
    }

    std::cout << "Average time per frame: " << total_time.count() / frames
              << "ms\n";
    std::cout << "Problem size: " << domain.points.shape()[1] << "\n";

    return 0;
}
