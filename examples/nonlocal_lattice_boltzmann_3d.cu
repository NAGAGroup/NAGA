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

#include "naga/fluids/nonlocal_lattice_boltzmann/simulation_variables.cuh"
#include "utils.hpp"
#include <naga/fluids/nonlocal_lattice_boltzmann.cuh>
#include <naga/regions/hypersphere.cuh>

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>

using value_type = float;

using lattice_t = naga::fluids::nonlocal_lbm::d3q27_lattice<value_type>;
using sim_engine_t
    = naga::fluids::nonlocal_lbm::detail::simulation_engine<lattice_t>;

using density_source_t = naga::fluids::nonlocal_lbm::density_source<value_type>;
using simulation_domain_t
    = naga::fluids::nonlocal_lbm::simulation_domain<const value_type>;
using problem_parameters_t
    = naga::fluids::nonlocal_lbm::problem_parameters<value_type>;
using region_t   = naga::regions::hypersphere<value_type, 3>;
using solution_t = naga::fluids::nonlocal_lbm::state_variables<lattice_t>;

template<class T>
struct get_vtk_array_type {
    using type = void;
};

template<>
struct get_vtk_array_type<float> {
    using type = vtkFloatArray;
};

template<>
struct get_vtk_array_type<double> {
    using type = vtkDoubleArray;
};

class spherical_init_peak : public density_source_t {
  public:
    std::future<void> add_density_source(
        const simulation_domain_t& domain,
        const problem_parameters_t& params,
        const value_type& time,
        sclx::array<value_type, 1>& source_terms
    ) final {
        region_t source_region{0.1, {0.0, 0.0, 0.0}};
        static bool has_run = false;
        if (has_run) {
            return std::async(std::launch::deferred, []() {});
        }
        has_run = true;

        return sclx::execute_kernel([=](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{source_terms.shape()},
                source_terms,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    if (source_region.contains(&domain.points(0, idx[0]))) {
                        auto distance
                            = naga::distance_functions::loopless::euclidean<
                                3>{}(
                                &domain.points(0, idx[0]),
                                source_region.center()
                            );
                        // gaussian peak as a function of distance from the
                        // center w/ width radius
                        source_terms(idx[0])
                            += 0.001f
                             * naga::math::exp(
                                   -distance * distance
                                   / (2.0 * source_region.radius()
                                      * source_region.radius())
                             );
                    }
                }
            );
        });
    }
};

void save_solution(const sim_engine_t& engine, uint save_frame);

int main() {
    auto examples_path = get_examples_dir();

    sclx::filesystem::path domain_dir
        = examples_path / "../resources/lbm_example_domains/ball_in_cube";

    naga::fluids::nonlocal_lbm::boundary_specification<value_type>
        outer_boundary{
            domain_dir / "cube.obj",
            0,
            0,
            .01f,
        };

    std::vector<naga::fluids::nonlocal_lbm::boundary_specification<value_type>>
        inner_boundaries{};

    auto domain
        = naga::fluids::nonlocal_lbm::simulation_domain<value_type>::import <3>(
            outer_boundary,
            inner_boundaries
        );

    sim_engine_t engine;
    engine.set_problem_parameters(
        0.0f,
        1.0f,
        0.1f * domain.nodal_spacing * domain.nodal_spacing,
        2.f,
        0.4f,
        0.2f
    );
    engine.init_domain(domain);
    save_solution(engine, 0);

    std::cout << "Lattice time step: "
              << engine.parameters_.time_step
                     / engine.parameters_.nondim_factors.time_scale
              << "\n";
    std::cout << "Lattice approx nodal spacing: "
              << engine.domain_.nodal_spacing
                     / engine.parameters_.nondim_factors.length_scale
              << "\n";

    spherical_init_peak source{};
    engine.register_density_source(source);

    int frames = 1000;
    std::mutex frame_mutex;
    std::chrono::milliseconds total_time{0};
    uint save_frame = 0;
    value_type fps  = 60.0f;
    while (engine.frame_number_ * engine.parameters_.time_step < 1.f) {
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

        if (engine.frame_number_ * engine.parameters_.time_step * fps
            < save_frame) {
            continue;
        }

        save_solution(engine, save_frame);
        ++save_frame;
    }

    std::cout << "Average time per frame: " << total_time.count() / frames
              << "ms\n";
    std::cout << "Problem size: " << domain.points.shape()[1] << "\n";

    return 0;
}

void save_solution(const sim_engine_t& engine, uint save_frame) {
    const auto& domain   = engine.domain_;
    const auto& solution = engine.solution_;
    static auto results_path
        = get_examples_results_dir() / "nonlocal_lattice_boltzmann_3d_results";

    static bool first = true;

    if (first) {
        first = false;
        sclx::filesystem::remove_all(results_path);
        sclx::filesystem::create_directories(results_path);
    }

    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(static_cast<vtkIdType>(domain.points.shape()[1]));

    vtkNew<get_vtk_array_type<value_type>::type> f;
    f->SetNumberOfComponents(lattice_t::size);
    f->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1]));
    f->SetName("f");

    vtkNew<get_vtk_array_type<value_type>::type> density;
    density->SetNumberOfComponents(1);
    density->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    density->SetName("density");

    vtkNew<get_vtk_array_type<value_type>::type> velocity;
    velocity->SetNumberOfComponents(3);
    velocity->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    velocity->SetName("velocity");

    vtkNew<get_vtk_array_type<value_type>::type> absorption;
    absorption->SetNumberOfComponents(1);
    absorption->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape(
    )[1]));
    absorption->SetName("absorption");

    vtkNew<get_vtk_array_type<value_type>::type> normals;
    normals->SetNumberOfComponents(3);
    normals->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    normals->SetName("normals");

    vtkNew<vtkIntArray> type;
    type->SetNumberOfComponents(1);
    type->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1]));
    type->SetName("type");

    for (vtkIdType i = 0; i < domain.points.shape()[1]; ++i) {
        points->SetPoint(
            i,
            domain.points(0, i),
            domain.points(1, i),
            domain.points(2, i)
        );

        value_type f_i[lattice_t::size];
        auto lat_weights
            = naga::fluids::nonlocal_lbm::detail::lattice_interface<
                lattice_t>::lattice_weights();
        for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
            f_i[alpha] = (solution.lattice_distributions[alpha][i]
                          - lat_weights.vals[alpha])
                       / lat_weights.vals[alpha];
        }
        f->SetTuple(i, f_i);

        density->SetTuple1(i, solution.macroscopic_values.fluid_density(i));

        velocity->SetTuple3(
            i,
            solution.macroscopic_values.fluid_velocity(0, i),
            solution.macroscopic_values.fluid_velocity(1, i),
            solution.macroscopic_values.fluid_velocity(2, i)
        );

        int type_i = 0;
        value_type normal_i[3]{0, 0, 0};
        value_type absorption_i = 0;
        if (i >= domain.num_bulk_points + domain.num_layer_points) {
            type_i      = 2;
            normal_i[0] = domain.boundary_normals(
                0,
                i - domain.num_bulk_points - domain.num_layer_points
            );
            normal_i[1] = domain.boundary_normals(
                1,
                i - domain.num_bulk_points - domain.num_layer_points
            );
            normal_i[2] = domain.boundary_normals(
                2,
                i - domain.num_bulk_points - domain.num_layer_points
            );
        } else if (i >= domain.num_bulk_points) {
            type_i       = 1;
            absorption_i = domain.layer_absorption(i - domain.num_bulk_points);
        }
        normals->SetTuple3(i, normal_i[0], normal_i[1], normal_i[2]);
        type->SetTuple1(i, type_i);
        absorption->SetTuple1(i, absorption_i);
    }

    std::string filename
        = results_path
        / (std::string("result.") + std::to_string(save_frame) + ".vtp");
    vtkNew<vtkPolyData> polydata;
    polydata->SetPoints(points);
    polydata->GetPointData()->AddArray(f);
    polydata->GetPointData()->AddArray(density);
    polydata->GetPointData()->AddArray(velocity);
    polydata->GetPointData()->AddArray(absorption);
    polydata->GetPointData()->AddArray(normals);
    polydata->GetPointData()->AddArray(type);

    vtkNew<vtkXMLPolyDataWriter> writer;
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polydata);
    writer->Write();
}
