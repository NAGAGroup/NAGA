
// BSD 3-Clause License
//
// Copyright (c) 2023
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

#include "../simulation_observer.cuh"

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>

namespace naga::fluids::nonlocal_lbm {

template<class T>
struct get_vtk_array_type;

template<>
struct get_vtk_array_type<float> {
    using type = vtkFloatArray;
};

template<>
struct get_vtk_array_type<double> {
    using type = vtkDoubleArray;
};

template<class Lattice>
class vtk_observer : public simulation_observer<Lattice> {
  public:
    using base                 = simulation_observer<Lattice>;
    using value_type           = typename base::value_type;
    using simulation_domain_t  = typename base::simulation_domain_t;
    using problem_parameters_t = typename base::problem_parameters_t;
    using solution_t           = typename base::solution_t;

    vtk_observer(
        sclx::filesystem::path output_directory,
        const value_type& time_multiplier = 1,
        const value_type& frame_rate      = 60
    )
        : output_directory_(std::move(output_directory)),
          time_multiplier_(time_multiplier),
          frame_rate_(frame_rate) {}

    void update(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t&,
        const solution_t& solution
    ) {
        const auto& results_path = output_directory_;
        auto save_frame
            = static_cast<size_t>(time_multiplier_ * time * frame_rate_);
        if (save_frame < current_frame_) {
            return;
        }
        current_frame_ = save_frame + 1;

        previous_frame_future_.get();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->SetNumberOfPoints(static_cast<vtkIdType>(domain.points.shape(
        )[1]));

        using vtk_array_type = typename get_vtk_array_type<value_type>::type;

        vtkSmartPointer<vtk_array_type> f
            = vtkSmartPointer<vtk_array_type>::New();
        f->SetNumberOfComponents(Lattice::size);
        f->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1]));
        f->SetName("f");

        vtkSmartPointer<vtk_array_type> density
            = vtkSmartPointer<vtk_array_type>::New();
        density->SetNumberOfComponents(1);
        density->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape(
        )[1]));
        density->SetName("density");

        vtkSmartPointer<vtk_array_type> velocity
            = vtkSmartPointer<vtk_array_type>::New();
        velocity->SetNumberOfComponents(3);
        velocity->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape(
        )[1]));
        velocity->SetName("velocity");

        vtkSmartPointer<vtk_array_type> absorption
            = vtkSmartPointer<vtk_array_type>::New();
        absorption->SetNumberOfComponents(1);
        absorption->SetNumberOfTuples(
            static_cast<vtkIdType>(domain.points.shape()[1])
        );
        absorption->SetName("absorption");

        vtkSmartPointer<vtk_array_type> normals
            = vtkSmartPointer<vtk_array_type>::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape(
        )[1]));
        normals->SetName("normals");

        vtkSmartPointer<vtkIntArray> type = vtkSmartPointer<vtkIntArray>::New();
        type->SetNumberOfComponents(1);
        type->SetNumberOfTuples(static_cast<vtkIdType>(domain.points.shape()[1])
        );
        type->SetName("type");

        vtkSmartPointer<vtkCellArray> cells
            = vtkSmartPointer<vtkCellArray>::New();
        cells->Allocate(domain.points.shape()[1]);

        for (vtkIdType i = 0; i < domain.points.shape()[1]; ++i) {
            value_type point[3]{0, 0, 0};
            for (int d = 0; d < Lattice::dimensions; ++d) {
                point[d] = domain.points(d, i);
            }
            points->SetPoint(
                i,
                point[0],
                point[1],
                point[2]
            );

            value_type f_i[Lattice::size];
            auto lat_weights
                = detail::lattice_interface<Lattice>::lattice_weights();
            for (int alpha = 0; alpha < Lattice::size; ++alpha) {
                f_i[alpha] = (solution.lattice_distributions[alpha][i])
                           / lat_weights.vals[alpha];
            }
            f->SetTuple(i, f_i);

            density->SetTuple1(i, solution.macroscopic_values.fluid_density(i));


            value_type fluid_velocity[3]{0, 0, 0};
            for (int d = 0; d < Lattice::dimensions; ++d) {
                point[d] = solution.macroscopic_values.fluid_velocity(d, i);
            }
            velocity->SetTuple3(
                i,
                fluid_velocity[0],
                fluid_velocity[1],
                fluid_velocity[2]
            );

            int type_i = 0;
            value_type normal_i[3]{0, 0, 0};
            value_type absorption_i = 0;
            if (i >= domain.num_bulk_points + domain.num_layer_points) {
                type_i      = 2;
                for (int d = 0; d < Lattice::dimensions; ++d) {
                    normal_i[d] = domain.boundary_normals(
                        d,
                        i - domain.num_bulk_points - domain.num_layer_points
                    );
                }
            } else if (i >= domain.num_bulk_points) {
                type_i = 1;
                absorption_i
                    = domain.layer_absorption(i - domain.num_bulk_points);
            }
            normals->SetTuple3(i, normal_i[0], normal_i[1], normal_i[2]);
            type->SetTuple1(i, type_i);
            absorption->SetTuple1(i, absorption_i);

            cells->InsertNextCell(1, &i);
        }

        auto fut               = std::async(std::launch::async, [=]() {
            std::string filename = results_path
                                 / (std::string("result.")
                                    + std::to_string(save_frame) + ".vtp");
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
        previous_frame_future_ = std::move(fut);
    }

  private:
    sclx::filesystem::path output_directory_;
    value_type time_multiplier_;
    value_type frame_rate_;
    std::future<void> previous_frame_future_
        = std::async(std::launch::deferred, []() {});
    size_t current_frame_ = 0;
};

}  // namespace naga::fluids::nonlocal_lbm
