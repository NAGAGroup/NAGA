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
#include <naga/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/conforming_point_cloud_provider.cuh>

#include "naga/fluids/nonlocal_lattice_boltzmann/simulation_variables.cuh"
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

using value_type = double;

using lattice_t = naga::fluids::nonlocal_lbm::d3q27_lattice<value_type>;
using sim_engine_t
    = naga::fluids::nonlocal_lbm::detail::simulation_engine<lattice_t>;

using density_source_t = naga::fluids::nonlocal_lbm::density_source<lattice_t>;
using simulation_domain_t
    = naga::fluids::nonlocal_lbm::simulation_nodes<const value_type>;
using problem_parameters_t
    = naga::fluids::nonlocal_lbm::problem_parameters<value_type>;
using region_t   = naga::regions::hypersphere<value_type, 3>;
using solution_t = naga::fluids::nonlocal_lbm::state_variables<lattice_t>;

using node_provider_t
    = naga::experimental::fluids::nonlocal_lbm::conforming_point_cloud_provider<lattice_t>;

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
        region_t source_region{0.1, {-0.6, 0.0, 0.0}};
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

                        auto perturbation
                            = 0.01f
                            * (1
                               - naga::math::loopless::pow<2>(
                                     2 * distance / source_region.radius()
                                 ) / 4)
                            * naga::math::exp(-naga::math::loopless::pow<2>(
                                2 * distance / source_region.radius()
                            ));

                        source_terms(idx[0]
                        ) += perturbation * params.nondim_factors.density_scale;
                    }
                }
            );
        });
    }
};

std::future<void> save_solution(const sim_engine_t& engine, uint save_frame);

int main() {
    value_type nodal_spacing = 0.015;

    auto resources_dir
        = get_resources_dir() / "lbm_example_domains" / "ball_in_cube";

    sclx::filesystem::path domain_obj       = resources_dir / "cube.obj";
    double outer_absorption_layer_thickness = 0.0;
    double outer_absorption_coefficient     = 0.0;

    std::vector<sclx::filesystem::path> immersed_boundary_objs
        = {resources_dir / "ball.obj"};
    std::vector<double> immersed_absorption_layer_thicknesses = {0.0};
    std::vector<double> immersed_absorption_coefficients      = {0.0};

    node_provider_t node_provider(
        nodal_spacing,
        domain_obj,
        immersed_boundary_objs,
        outer_absorption_layer_thickness,
        outer_absorption_coefficient,
        immersed_absorption_layer_thicknesses,
        immersed_absorption_coefficients
    );

    sim_engine_t engine;
    engine.set_problem_parameters(
        1e-5f,
        1.0f,
        0.5 * nodal_spacing * nodal_spacing,
        2.f,
        0.4f,
        0.2f
    );

    engine.init_domain(node_provider);
    auto domain                   = engine.domain_;
    std::future<void> save_future = save_solution(engine, 0);

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

    int sim_frame = 0;
    std::mutex frame_mutex;
    std::chrono::milliseconds total_time{0};
    uint save_frame = 0;
    value_type fps  = 60.0f;
    while (engine.frame_number_ * engine.parameters_.time_step < 10.f) {
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

        ++sim_frame;

        if (engine.frame_number_ * engine.parameters_.time_step * fps
            < save_frame) {
            continue;
        }

        save_future.get();
        save_future = std::move(save_solution(engine, save_frame));
        ++save_frame;
    }

    std::cout << "Average time per frame: " << total_time.count() / sim_frame
              << "ms\n";
    std::cout << "Problem size: " << domain.points.shape()[1] << "\n";

    return 0;
}

std::future<void> save_solution(const sim_engine_t& engine, uint save_frame) {
    const auto& domain   = engine.domain_;
    const auto& solution = engine.solution_;
    static auto results_path
        = get_examples_results_dir() / "cgal_3d_mesh";

    static bool first = true;

    if (first) {
        first = false;
        sclx::filesystem::remove_all(results_path);
        sclx::filesystem::create_directories(results_path);
    }

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(static_cast<vtkIdType>(domain.points.shape()[1]));

    vtkSmartPointer<get_vtk_array_type<value_type>::type> f
        = vtkSmartPointer<get_vtk_array_type<value_type>::type>::New();
    f->SetNumberOfComponents(lattice_t::size);
    f->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1]));
    f->SetName("f");

    vtkSmartPointer<get_vtk_array_type<value_type>::type> density
        = vtkSmartPointer<get_vtk_array_type<value_type>::type>::New();
    density->SetNumberOfComponents(1);
    density->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    density->SetName("density");

    vtkSmartPointer<get_vtk_array_type<value_type>::type> velocity
        = vtkSmartPointer<get_vtk_array_type<value_type>::type>::New();
    velocity->SetNumberOfComponents(3);
    velocity->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    velocity->SetName("velocity");

    vtkSmartPointer<get_vtk_array_type<value_type>::type> absorption
        = vtkSmartPointer<get_vtk_array_type<value_type>::type>::New();
    absorption->SetNumberOfComponents(1);
    absorption->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape(
    )[1]));
    absorption->SetName("absorption");

    vtkSmartPointer<get_vtk_array_type<value_type>::type> normals
        = vtkSmartPointer<get_vtk_array_type<value_type>::type>::New();
    normals->SetNumberOfComponents(3);
    normals->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
    );
    normals->SetName("normals");

    vtkSmartPointer<vtkIntArray> type = vtkSmartPointer<vtkIntArray>::New();
    type->SetNumberOfComponents(1);
    type->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1]));
    type->SetName("type");

    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    cells->Allocate(domain.points.shape()[1]);

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
            f_i[alpha] = (solution.lattice_distributions[alpha][i])
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

        cells->InsertNextCell(1, &i);
    }

    return std::async(std::launch::async, [=]() {
        std::string filename
            = results_path
            / (std::string("result.") + std::to_string(save_frame) + ".vtp");
        vtkSmartPointer<vtkPolyData> polydata
            = vtkSmartPointer<vtkPolyData>::New();
        polydata->SetPoints(points);
        polydata->GetPointData()->AddArray(f);
        polydata->GetPointData()->AddArray(density);
        polydata->GetPointData()->AddArray(velocity);
        polydata->GetPointData()->AddArray(absorption);
        polydata->GetPointData()->AddArray(normals);
        polydata->GetPointData()->AddArray(type);
        polydata->SetVerts(cells);

        vtkSmartPointer<vtkXMLPolyDataWriter> writer
            = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(polydata);
        writer->SetCompressorTypeToNone();
        writer->Write();
    });
}