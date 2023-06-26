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
#include <naga/nonlocal_calculus/advection.cuh>
#include <naga/regions/hypersphere.cuh>
#include <scalix/filesystem.hpp>

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>

using namespace naga::fluids;

using value_type = float;
using region_t   = naga::regions::hypersphere<value_type, 3>;
using lattice_t  = nonlocal_lbm::d3q27_lattice<value_type>;

__managed__ auto lattice_velocities
    = nonlocal_lbm::detail::lattice_interface<lattice_t>::lattice_velocities();
__managed__ auto lattice_weights
    = nonlocal_lbm::detail::lattice_interface<lattice_t>::lattice_weights();

template<class PointType>
__host__ __device__ value_type
field_function(const PointType& x, const int& alpha) {
    region_t source_region(0.1, {0.0, 0.0, 0.0});

    // 3D gaussian pulse with radius 0.05 and center (0.5, 0.5)
    if (!source_region.contains(x)) {
        return lattice_weights.vals[alpha];
    }

    auto distance = naga::distance_functions::loopless::euclidean<3>{}(
        x,
        source_region.center()
    );

    auto perturbation
        = 0.01f
        * (1
           - naga::math::loopless::pow<2>(2 * distance / source_region.radius())
                 / 4)
        * naga::math::exp(-naga::math::loopless::pow<2>(
            2 * distance / source_region.radius()
        ));

    return (1.f + perturbation) * lattice_weights.vals[alpha];
}

int main() {
    value_type grid_length   = 2.f;
    value_type spacing_scale = 1.f;
    value_type grid_spacing  = value_type{0.03125} * spacing_scale;
    value_type time_step     = value_type{9.76563e-05} * spacing_scale;
    size_t grid_size         = std::floor(grid_length / grid_spacing + 1);

    size_t point_count = grid_size * grid_size * grid_size;
    sclx::array<value_type, 2> source_grid{3, point_count};
    sclx::array<value_type, 1> source_values[lattice_t::size];

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{point_count},
            source_grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                source_grid(0, idx[0])
                    = static_cast<value_type>(idx[0] % grid_size) * grid_spacing
                    - grid_length / 2.f;
                source_grid(1, idx[0])
                    = static_cast<value_type>((idx[0] / grid_size) % grid_size)
                        * grid_spacing
                    - grid_length / 2.f;
                source_grid(2, idx[0])
                    = static_cast<value_type>(idx[0] / grid_size / grid_size)
                        * grid_spacing
                    - grid_length / 2.f;
            }
        );
    });

    for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
        auto& f_alpha = source_values[alpha];
        using array_t = std::decay_t<decltype(f_alpha)>;
        f_alpha       = array_t{point_count};
        sclx::execute_kernel([&](sclx::kernel_handler& handle) {
            handle.launch(
                sclx::md_range_t<1>{point_count},
                f_alpha,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    const auto& field_value
                        = field_function(&source_grid(0, idx[0]), alpha);
                    f_alpha[idx] = field_value;
                }
            );
        });
    }

    auto results_path = get_examples_results_dir() / "lbm_streaming_tests_3d";
    sclx::filesystem::remove_all(results_path);
    sclx::filesystem::create_directories(results_path);

    sclx::array<value_type, 1> advection_result[lattice_t::size];
    for (auto& f_alpha : advection_result) {
        f_alpha = sclx::array<value_type, 1>{point_count};
    }
    auto advection_op
        = naga::nonlocal_calculus::advection_operator<value_type, 3>::create(
            source_grid
        );

    value_type time = 0.0f;
    uint save_frame = 0;
    value_type fps  = 60.0f;
    while (time < 0.4f) {
        std::cout << "Time: " << time << "\n";
        for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
            value_type velocity[3];
            for (int i = 0; i < 3; ++i) {
                velocity[i] = lattice_velocities.vals[alpha][i];
            }
            auto velocity_field = naga::nonlocal_calculus::
                constant_velocity_field<value_type, 3>::create(velocity);

            auto& f_alpha        = source_values[alpha];
            auto& f_alpha_result = advection_result[alpha];
            advection_op.step_forward(
                velocity_field,
                f_alpha,
                f_alpha_result,
                time_step,
                lattice_weights.vals[alpha]
            );
        }
        if (time * fps >= static_cast<value_type>(save_frame)) {
            vtkNew<vtkPoints> points;
            points->SetNumberOfPoints(point_count);

            vtkNew<vtkFloatArray> values;
            values->SetName("values");
            values->SetNumberOfComponents(lattice_t::size);
            values->SetNumberOfTuples(point_count);

            vtkNew<vtkCellArray> cells;
            cells->Allocate(point_count);

            for (vtkIdType i = 0; i < point_count; ++i) {
                value_type normalized_result[lattice_t::size];
                for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
                    normalized_result[alpha] = (advection_result[alpha][i]
                                                - lattice_weights.vals[alpha])
                                             / lattice_weights.vals[alpha];
                }
                points->SetPoint(
                    i,
                    source_grid(0, i),
                    source_grid(1, i),
                    source_grid(2, i)
                );
                values->SetTuple(i, normalized_result);
                cells->InsertNextCell(1, &i);
            }

            vtkNew<vtkPolyData> polydata;
            polydata->SetPoints(points);
            polydata->GetPointData()->AddArray(values);
            polydata->SetVerts(cells);

            vtkNew<vtkXMLPolyDataWriter> writer;
            writer->SetFileName(
                (results_path / ("frame_" + std::to_string(save_frame) + ".vtp")
                )
                    .string()
                    .c_str()
            );
            writer->SetInputData(polydata);
            writer->Write();

            ++save_frame;
        }
        time += time_step;
        for (int alpha = 0; alpha < lattice_t::size; ++alpha) {
            sclx::assign_array(advection_result[alpha], source_values[alpha]);
        }
    }

    return 0;
}
