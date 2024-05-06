
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

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <random>
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
        const value_type& frame_rate      = 60,
        const size_t approx_save_points   = 0,
        bool include_ghost_nodes          = false
    )
        : output_directory_(std::move(output_directory)),
          time_multiplier_(time_multiplier),
          frame_rate_(frame_rate),
          approx_save_points_(approx_save_points),
          include_ghost_nodes_(include_ghost_nodes) {}

    void update(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t& parameters,
        const solution_t& solution
    ) {
        if (approx_save_points_ == 0) {
            update_all(time, domain, parameters, solution);
        } else {
            update_approx(time, domain, parameters, solution);
        }
    }

    void update_all(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t& parameters,
        const solution_t& solution
    ) {
        const auto& results_path = output_directory_;
        auto save_frame
            = static_cast<std::uint32_t>(time_multiplier_ * time * frame_rate_);
        save_frame = frame_rate_ != 0.
                       ? save_frame
                       : static_cast<std::uint32_t>(
                             std::ceil(time / parameters.time_step)
                         );
        if (frame_rate_ != 0 && save_frame < current_frame_) {
            return;
        }
        current_frame_ = save_frame + 1;

        if (previous_frame_future_.valid()) {
            previous_frame_future_.get();
        }

        auto number_of_points = domain.points.shape()[1];
        if (!include_ghost_nodes_) {
            number_of_points -= domain.num_ghost_nodes;
        }

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->SetNumberOfPoints(static_cast<vtkIdType>(number_of_points));

        using vtk_array_type = typename get_vtk_array_type<value_type>::type;

        vtkSmartPointer<vtk_array_type> f
            = vtkSmartPointer<vtk_array_type>::New();
        f->SetNumberOfComponents(Lattice::size);
        f->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        f->SetName("f");

        vtkSmartPointer<vtk_array_type> density
            = vtkSmartPointer<vtk_array_type>::New();
        density->SetNumberOfComponents(1);
        density->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        density->SetName("density");

        vtkSmartPointer<vtk_array_type> velocity
            = vtkSmartPointer<vtk_array_type>::New();
        velocity->SetNumberOfComponents(3);
        velocity->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        velocity->SetName("velocity");

        vtkSmartPointer<vtk_array_type> absorption
            = vtkSmartPointer<vtk_array_type>::New();
        absorption->SetNumberOfComponents(1);
        absorption->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        absorption->SetName("absorption");

        vtkSmartPointer<vtk_array_type> normals
            = vtkSmartPointer<vtk_array_type>::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        normals->SetName("normals");

        vtkSmartPointer<vtkIntArray> type = vtkSmartPointer<vtkIntArray>::New();
        type->SetNumberOfComponents(1);
        type->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        type->SetName("type");

        vtkSmartPointer<vtkCellArray> cells
            = vtkSmartPointer<vtkCellArray>::New();
        cells->Allocate(number_of_points);

        for (vtkIdType i = 0; i < number_of_points; ++i) {
            value_type point[3]{0, 0, 0};
            for (int d = 0; d < Lattice::dimensions; ++d) {
                point[d] = domain.points(d, i);
            }
            points->SetPoint(i, point[0], point[1], point[2]);

            value_type f_i[Lattice::size];
            auto lat_weights
                = detail::lattice_interface<Lattice>::lattice_weights();
            for (int alpha = 0; alpha < Lattice::size; ++alpha) {
                f_i[alpha] = (solution.lattice_distributions[alpha][i])
                           / lat_weights.vals[alpha];
            }
            f->SetTuple(i, f_i);

            density->SetTuple1(i, solution.macroscopic_values.fluid_density(i));

            value_type fluid_velocity[3];
            for (int d = 0; d < Lattice::dimensions; ++d) {
                fluid_velocity[d]
                    = solution.macroscopic_values.fluid_velocity(d, i);
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
            if (i >= domain.num_bulk_points + domain.num_layer_points
                         + domain.num_boundary_points) {
                type_i = 3;
                absorption_i
                    = domain.layer_absorption(i - domain.num_bulk_points);
            } else if (i >= domain.num_bulk_points + domain.num_layer_points) {
                type_i = 2;
                for (int d = 0; d < Lattice::dimensions; ++d) {
                    normal_i[d] = domain.boundary_normals(
                        d,
                        i - domain.num_bulk_points - domain.num_layer_points
                    );
                }
                absorption_i
                    = domain.layer_absorption(i - domain.num_bulk_points);
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

    void update_approx(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t& parameters,
        const solution_t& solution
    ) {
        if (save_points_.empty()) {
            auto number_of_points = domain.points.shape()[1];
            if (!include_ghost_nodes_) {
                number_of_points -= domain.num_ghost_nodes;
            }

            // randomly choose save points given the approximate number of save
            // points
            save_points_.reserve(approx_save_points_);
            std::uniform_int_distribution<std::uint32_t> dist(
                0,
                number_of_points - 1
            );
            std::mt19937 gen;
            for (size_t i = 0; i < approx_save_points_; ++i) {
                save_points_.push_back(dist(gen));
            }
            std::sort(save_points_.begin(), save_points_.end());
            auto last = std::unique(save_points_.begin(), save_points_.end());
            save_points_.erase(last, save_points_.end());
        }
        const auto& results_path = output_directory_;
        auto save_frame
            = static_cast<std::uint32_t>(time_multiplier_ * time * frame_rate_);
        save_frame = frame_rate_ != 0.
                       ? save_frame
                       : static_cast<std::uint32_t>(
                             std::ceil(time / parameters.time_step)
                         );
        if (frame_rate_ != 0 && save_frame < current_frame_) {
            return;
        }
        current_frame_ = save_frame + 1;

        if (previous_frame_future_.valid()) {
            previous_frame_future_.get();
        }

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->SetNumberOfPoints(static_cast<vtkIdType>(save_points_.size()));

        using vtk_array_type = typename get_vtk_array_type<value_type>::type;

        vtkSmartPointer<vtk_array_type> f
            = vtkSmartPointer<vtk_array_type>::New();
        f->SetNumberOfComponents(Lattice::size);
        f->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size()));
        f->SetName("f");

        vtkSmartPointer<vtk_array_type> density
            = vtkSmartPointer<vtk_array_type>::New();
        density->SetNumberOfComponents(1);
        density->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size()));
        density->SetName("density");

        vtkSmartPointer<vtk_array_type> velocity
            = vtkSmartPointer<vtk_array_type>::New();
        velocity->SetNumberOfComponents(3);
        velocity->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size())
        );
        velocity->SetName("velocity");

        vtkSmartPointer<vtk_array_type> absorption
            = vtkSmartPointer<vtk_array_type>::New();
        absorption->SetNumberOfComponents(1);
        absorption->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size()
        ));
        absorption->SetName("absorption");

        vtkSmartPointer<vtk_array_type> normals
            = vtkSmartPointer<vtk_array_type>::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size()));
        normals->SetName("normals");

        vtkSmartPointer<vtkIntArray> type = vtkSmartPointer<vtkIntArray>::New();
        type->SetNumberOfComponents(1);
        type->SetNumberOfTuples(static_cast<vtkIdType>(save_points_.size()));
        type->SetName("type");

        vtkSmartPointer<vtkCellArray> cells
            = vtkSmartPointer<vtkCellArray>::New();
        cells->Allocate(save_points_.size());

        for (auto& p : save_points_) {
            auto i
                = static_cast<vtkIdType>(std::distance(&save_points_[0], &p));
            value_type point[3]{0, 0, 0};
            for (int d = 0; d < Lattice::dimensions; ++d) {
                point[d] = domain.points(d, p);
            }
            points->SetPoint(i, point[0], point[1], point[2]);

            value_type f_i[Lattice::size];
            auto lat_weights
                = detail::lattice_interface<Lattice>::lattice_weights();
            for (int alpha = 0; alpha < Lattice::size; ++alpha) {
                f_i[alpha] = (solution.lattice_distributions[alpha][p])
                           / lat_weights.vals[alpha];
            }
            f->SetTuple(i, f_i);

            density->SetTuple1(i, solution.macroscopic_values.fluid_density(p));

            value_type fluid_velocity[3];
            for (int d = 0; d < Lattice::dimensions; ++d) {
                fluid_velocity[d]
                    = solution.macroscopic_values.fluid_velocity(d, p);
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
            if (p >= domain.num_bulk_points + domain.num_layer_points
                         + domain.num_boundary_points) {
                type_i = 3;
                absorption_i
                    = domain.layer_absorption(p - domain.num_bulk_points);
            } else if (p >= domain.num_bulk_points + domain.num_layer_points) {
                type_i = 2;
                for (int d = 0; d < Lattice::dimensions; ++d) {
                    normal_i[d] = domain.boundary_normals(
                        d,
                        p - domain.num_bulk_points - domain.num_layer_points
                    );
                }
                absorption_i
                    = domain.layer_absorption(p - domain.num_bulk_points);
            } else if (p >= domain.num_bulk_points) {
                type_i = 1;
                absorption_i
                    = domain.layer_absorption(p - domain.num_bulk_points);
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

    template<class Archive>
    void save_state(Archive& ar) const {
        if (previous_frame_future_.valid()) {
            previous_frame_future_.wait();
        }
        ar(output_directory_.string());
        ar(time_multiplier_, frame_rate_, current_frame_);
    }

    template<class Archive>
    void load_state(Archive& ar) {
        if (previous_frame_future_.valid()) {
            previous_frame_future_.wait();
        }
        std::string output_directory_str;
        ar(output_directory_str);
        output_directory_ = output_directory_str;
        ar(time_multiplier_, frame_rate_, current_frame_);
    }

  private:
    sclx::filesystem::path output_directory_;
    value_type time_multiplier_;
    value_type frame_rate_;
    std::future<void> previous_frame_future_;
    size_t current_frame_ = 0;
    size_t approx_save_points_;
    std::vector<std::uint32_t> save_points_;
    bool include_ghost_nodes_;
};

template<class Lattice>
class vtk_observerv2 : public simulation_observer<Lattice> {
  public:
    using base                       = simulation_observer<Lattice>;
    using value_type                 = typename base::value_type;
    using simulation_domain_t        = typename base::simulation_domain_t;
    using problem_parameters_t       = typename base::problem_parameters_t;
    using solution_t                 = typename base::solution_t;
    static constexpr auto dimensions = Lattice::dimensions;

    vtk_observerv2(
        const simulation_domain_t& domain,
        const node_provider<Lattice>& nodes,
        sclx::filesystem::path output_directory,
        const value_type& frame_rate      = 60,
        const value_type& time_multiplier = 1
    )
        : output_directory_(std::move(output_directory)),
          frame_rate_(frame_rate),
          time_multiplier_(time_multiplier) {
        auto visualization_nodes  = nodes.get();
        auto visualization_points = visualization_nodes.points.get_range(
            {0},
            {visualization_nodes.num_bulk_points
             + visualization_nodes.num_layer_points
             + visualization_nodes.num_boundary_points}
        );
        {
            // We use the nearest neighbors algorithm to provide the
            // interpolation indices to the radial point method.
            auto points = domain.points.get_range(
                {0},
                {domain.num_bulk_points + domain.num_layer_points
                 + domain.num_boundary_points}
            );

            uint num_interp_points = 32;
            segmentation::nd_cubic_segmentation<value_type, dimensions>
                source_segmentation(points, num_interp_points);
            naga::default_point_map<value_type, dimensions> boundary_point_map{
                visualization_points
            };
            auto [distances_squared, indices]
                = naga::segmentation::batched_nearest_neighbors(
                    num_interp_points,
                    boundary_point_map,
                    source_segmentation
                );

            visualization_interpolater_ptr_ = std::make_shared<interpolater_t>(
                interpolater_t::create_interpolator(
                    points,
                    indices,
                    boundary_point_map,
                    domain.nodal_spacing
                )
            );
        }

        points_ = sclx::array<value_type, 2>{
            dimensions,
            visualization_points.shape()[1]
        };
        sclx::assign_array(visualization_points, points_);
        fluid_density_  = sclx::array<value_type, 1>{points_.shape()[1]};
        fluid_velocity_ = sclx::array<value_type, 2>{points_.shape()};
    }

    void update(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t& parameters,
        const solution_t& simulation_solution
    ) {
        const auto& results_path = output_directory_;
        auto save_frame
            = static_cast<std::uint32_t>(time_multiplier_ * time * frame_rate_);
        save_frame = frame_rate_ != 0.
                       ? save_frame
                       : static_cast<std::uint32_t>(
                             std::ceil(time / parameters.time_step)
                         );
        if (frame_rate_ != 0 && save_frame < current_frame_) {
            return;
        }
        current_frame_ = save_frame + 1;

        if (previous_frame_future_.valid()) {
            previous_frame_future_.get();
        }

        visualization_interpolater_ptr_->interpolate(
            simulation_solution.macroscopic_values.fluid_density.get_range(
                {0},
                {domain.num_bulk_points + domain.num_layer_points
                 + domain.num_boundary_points}
            ),
            fluid_density_
        );

        visualization_interpolater_ptr_->interpolate(
            simulation_solution.macroscopic_values.fluid_velocity.get_range(
                {0},
                {domain.num_bulk_points + domain.num_layer_points
                 + domain.num_boundary_points}
            ),
            fluid_velocity_
        );

        auto number_of_points = points_.shape()[1];

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->SetNumberOfPoints(static_cast<vtkIdType>(number_of_points));

        using vtk_array_type = typename get_vtk_array_type<value_type>::type;

        vtkSmartPointer<vtk_array_type> density
            = vtkSmartPointer<vtk_array_type>::New();
        density->SetNumberOfComponents(1);
        density->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        density->SetName("density");

        vtkSmartPointer<vtk_array_type> velocity
            = vtkSmartPointer<vtk_array_type>::New();
        velocity->SetNumberOfComponents(3);
        velocity->SetNumberOfTuples(static_cast<vtkIdType>(number_of_points));
        velocity->SetName("velocity");

        vtkSmartPointer<vtkCellArray> cells
            = vtkSmartPointer<vtkCellArray>::New();
        cells->Allocate(number_of_points);

        for (vtkIdType i = 0; i < number_of_points; ++i) {
            value_type point[3]{0, 0, 0};
            for (int d = 0; d < Lattice::dimensions; ++d) {
                point[d] = points_(d, i);
            }
            points->SetPoint(i, point[0], point[1], point[2]);

            density->SetTuple1(i, fluid_density_(i));

            value_type fluid_velocity[3];
            for (int d = 0; d < Lattice::dimensions; ++d) {
                fluid_velocity[d] = fluid_velocity_(d, i);
            }
            velocity->SetTuple3(
                i,
                fluid_velocity[0],
                fluid_velocity[1],
                fluid_velocity[2]
            );

            cells->InsertNextCell(1, &i);
        }

        auto fut               = std::async(std::launch::async, [=]() {
            std::string filename = results_path
                                 / (std::string("result.")
                                    + std::to_string(save_frame) + ".vtp");
            vtkSmartPointer<vtkPolyData> polydata
                = vtkSmartPointer<vtkPolyData>::New();
            polydata->SetPoints(points);
            polydata->GetPointData()->AddArray(density);
            polydata->GetPointData()->AddArray(velocity);
            polydata->SetVerts(cells);

            vtkSmartPointer<vtkXMLPolyDataWriter> writer
                = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
            writer->SetFileName(filename.c_str());
            writer->SetInputData(polydata);
            writer->Write();
        });
        previous_frame_future_ = std::move(fut);
    }

    template<class Archive>
    void save_state(Archive& ar) const {}

    template<class Archive>
    void load_state(Archive& ar) {}

  private:
    size_t current_frame_ = 0;
    sclx::filesystem::path output_directory_;
    value_type frame_rate_;
    value_type time_multiplier_;
    sclx::array<value_type, 2> points_;
    sclx::array<value_type, 1> fluid_density_;
    sclx::array<value_type, 2> fluid_velocity_;
    using interpolater_t = interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolater_t> visualization_interpolater_ptr_;
    std::future<void> previous_frame_future_;
};

}  // namespace naga::fluids::nonlocal_lbm
