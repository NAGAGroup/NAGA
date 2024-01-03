
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
#include "naga/interpolation/radial_point_method.cuh"
#include "naga/segmentation/nd_cubic_segmentation.cuh"
#include "naga/segmentation/nearest_neighbors.cuh"
#include <naga/point.hpp>
#include <sdf/sdf.hpp>
#include <tiny_obj_loader.h>

namespace naga::fluids::nonlocal_lbm::detail {

struct edge_t {
    edge_t(size_t i, size_t j) {
        if (i > j) {
            std::swap(i, j);
        }
        this->i = i;
        this->j = j;
    }

    edge_t()                               = default;
    edge_t(const edge_t& other)            = default;
    edge_t(edge_t&& other)                 = default;
    edge_t& operator=(const edge_t& other) = default;
    edge_t& operator=(edge_t&& other)      = default;

    size_t i;
    size_t j;
};

auto operator==(const edge_t& e1, const edge_t& e2) -> bool {
    return e1.i == e2.i && e1.j == e2.j;
}

}  // namespace naga::fluids::nonlocal_lbm::detail

template<>
struct std::hash<naga::fluids::nonlocal_lbm::detail::edge_t> {
    using argument_type = naga::fluids::nonlocal_lbm::detail::edge_t;
    auto operator()(const argument_type& edge) const noexcept -> size_t {
        return std::hash<size_t>()(edge.i) ^ std::hash<size_t>()(edge.j);
    }
};

namespace naga::fluids::nonlocal_lbm::detail {

struct sdf_metadata {
    std::shared_ptr<sdf::SDF> sdf;
    std::shared_ptr<sdf::Points> points;
    std::shared_ptr<sdf::Triangles> faces;
};

template<class T>
struct manifold_mesh_t {
    using value_type = T;

    static constexpr auto no_face   = std::numeric_limits<size_t>::max();
    static constexpr auto no_edge   = std::numeric_limits<size_t>::max();
    static constexpr auto no_vertex = std::numeric_limits<size_t>::max();

    using edge_t       = detail::edge_t;
    using edge_count_t = int;

    static auto import_from_obj(const sclx::filesystem::path& obj_path)
        -> manifold_mesh_t {

        using namespace detail;
        manifold_mesh_t mesh;

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(obj_path)) {
            if (!reader.Error().empty()) {
                throw std::runtime_error("TinyObjReader: " + reader.Error());
            }
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        // check if all faces are triangles
        for (const auto& shape : reader.GetShapes()) {
            for (const auto& num_face_vertices : shape.mesh.num_face_vertices) {
                if (num_face_vertices != 3) {
                    throw std::runtime_error(
                        "TinyObjReader: non-triangular face detected"
                    );
                }
            }
        }

        std::vector<size_t> face_vertex_indices;
        std::vector<size_t> face_normal_indices;
        std::unordered_map<edge_t, std::vector<size_t>> edge_opposite_vertices;
        std::unordered_map<edge_t, std::vector<size_t>> edge_face_neighbors;
        std::unordered_map<size_t, std::vector<size_t>> vertex_face_neighbors;
        std::unordered_map<size_t, std::vector<size_t>> vertex_opposite_edges;
        std::unordered_map<size_t, std::vector<size_t>> vertex_edge_neighbors;
        std::unordered_map<edge_t, edge_count_t> edge_counts;
        size_t total_face_count = 0;
        size_t total_vertex_count;
        size_t total_normal_count;

        {
            auto attrib = reader.GetAttrib();
            auto shapes = reader.GetShapes();

            total_vertex_count = attrib.vertices.size() / 3;
            mesh.vertices = sclx::array<value_type, 2>{3, total_vertex_count};
            std::copy(
                attrib.vertices.begin(),
                attrib.vertices.end(),
                &mesh.vertices(0, 0)
            );
            attrib.vertices.clear();

            total_normal_count = attrib.normals.size() / 3;
            mesh.normals = sclx::array<value_type, 2>{3, total_normal_count};
            std::copy(
                attrib.normals.begin(),
                attrib.normals.end(),
                &mesh.normals(0, 0)
            );

            // get metadata for constructing mesh
            for (const auto& shape : shapes) {
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size();
                     ++f) {
                    for (size_t i = 0; i < 3; ++i) {
                        // face data
                        face_vertex_indices.push_back(
                            shape.mesh.indices[f * 3 + i].vertex_index
                        );
                        face_normal_indices.push_back(
                            shape.mesh.indices[f * 3 + i].normal_index
                        );

                        // edge data
                        edge_t edge{
                            static_cast<size_t>(
                                shape.mesh.indices[f * 3 + i].vertex_index
                            ),
                            static_cast<size_t>(
                                shape.mesh.indices[f * 3 + (i + 1) % 3]
                                    .vertex_index
                            )
                        };
                        auto& edge_count = edge_counts[edge];
                        ++edge_count;
                        if (edge_count > 2) {
                            throw std::runtime_error(
                                "manifold_mesh_t: edge is "
                                "shared by more than two triangles"
                            );
                        }
                        edge_opposite_vertices[edge].push_back(
                            shape.mesh.indices[f * 3 + (i + 2) % 3].vertex_index
                        );
                        const auto& edge_opposite_vertex
                            = edge_opposite_vertices[edge].back();
                        edge_face_neighbors[edge].push_back(total_face_count);

                        auto edge_it  = edge_counts.find({edge.i, edge.j});
                        auto edge_idx = static_cast<size_t>(
                            std::distance(edge_counts.begin(), edge_it)
                        );

                        // vertex data
                        vertex_face_neighbors[shape.mesh.indices[f * 3 + i]
                                                  .vertex_index]
                            .push_back(total_face_count);
                        vertex_edge_neighbors[shape.mesh.indices[f * 3 + i]
                                                  .vertex_index]
                            .push_back(edge_idx);
                        vertex_opposite_edges[edge_opposite_vertex].push_back(
                            edge_idx
                        );
                    }
                    total_face_count += 1;
                }
            }
        }
        size_t max_vertex_face_neighbors = 0;
        for (const auto& [vertex, face_neighbors] : vertex_face_neighbors) {
            max_vertex_face_neighbors
                = std::max(max_vertex_face_neighbors, face_neighbors.size());
        }
        size_t max_vertex_edge_neighbors = 0;
        for (const auto& [vertex, edge_neighbors] : vertex_edge_neighbors) {
            max_vertex_edge_neighbors
                = std::max(max_vertex_edge_neighbors, edge_neighbors.size());
        }
        auto num_unique_edges = edge_counts.size();

        mesh.triangle_vert_indices
            = sclx::array<size_t, 2>{3, total_face_count};
        std::copy(
            face_vertex_indices.begin(),
            face_vertex_indices.end(),
            &mesh.triangle_vert_indices(0, 0)
        );
        face_vertex_indices.clear();
        face_vertex_indices.shrink_to_fit();

        mesh.unique_edges = sclx::array<size_t, 2>{2, num_unique_edges};
        mesh.double_edges = sclx::array<bool, 1>{num_unique_edges};
        sclx::fill(mesh.double_edges, false);
        mesh.edge_opposite_vertices
            = sclx::array<size_t, 2>{2, num_unique_edges};
        sclx::fill(mesh.edge_opposite_vertices, no_vertex);
        mesh.edge_face_neighbors = sclx::array<size_t, 2>{2, num_unique_edges};
        sclx::fill(mesh.edge_face_neighbors, no_face);
        for (const auto& [edge, count] : edge_counts) {
            auto edge_it  = edge_counts.find(edge);
            auto edge_idx = static_cast<size_t>(
                std::distance(edge_counts.begin(), edge_it)
            );
            if (count == 2) {
                mesh.double_edges(edge_idx) = true;
            }
            mesh.unique_edges(0, edge_idx) = edge.i;
            mesh.unique_edges(1, edge_idx) = edge.j;
            const auto& opposite_vertices  = edge_opposite_vertices[edge];
            if (opposite_vertices.size() != count) {
                throw std::runtime_error(
                    "manifold_mesh_t: edge opposite "
                    "vertices size does not match edge count, "
                    "this likely indicates bug in naga"
                );
            }
            const auto& face_neighbors = edge_face_neighbors[edge];
            if (face_neighbors.size() != count) {
                throw std::runtime_error("manifold_mesh_t: edge face neighbors "
                                         "size does not match edge count, "
                                         "this likely indicates bug in naga");
            }
            for (size_t i = 0; i < count; ++i) {
                mesh.edge_opposite_vertices(i, edge_idx) = opposite_vertices[i];
                mesh.edge_face_neighbors(i, edge_idx)    = face_neighbors[i];
            }
        }
        edge_counts.clear();
        edge_counts.rehash(0);
        edge_opposite_vertices.clear();
        edge_opposite_vertices.rehash(0);
        edge_face_neighbors.clear();
        edge_face_neighbors.rehash(0);

        mesh.vertex_face_neighbors = sclx::array<size_t, 2>{
            max_vertex_face_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_face_neighbors, no_face);
        mesh.vertex_edge_neighbors = sclx::array<size_t, 2>{
            max_vertex_edge_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_edge_neighbors, no_edge);
        mesh.vertex_opposite_edges = sclx::array<size_t, 2>{
            max_vertex_face_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_opposite_edges, no_edge);
        for (size_t i = 0; i < total_vertex_count; ++i) {
            if (vertex_face_neighbors[i].size()
                != vertex_opposite_edges[i].size()) {
                throw std::runtime_error(
                    "manifold_mesh_t: vertex face neighbor "
                    "and vertex opposite edge sizes do not match"
                );
            }
            for (size_t j = 0; j < vertex_face_neighbors[i].size(); ++j) {
                mesh.vertex_face_neighbors(j, i) = vertex_face_neighbors[i][j];
                mesh.vertex_opposite_edges(j, i) = vertex_opposite_edges[i][j];
            }
            for (size_t j = 0; j < vertex_edge_neighbors[i].size(); ++j) {
                mesh.vertex_edge_neighbors(j, i) = vertex_edge_neighbors[i][j];
            }
        }

        return mesh;
    }

    sclx::array<value_type, 2> vertices;
    sclx::array<value_type, 2> normals;
    sclx::array<size_t, 2> triangle_vert_indices;
    sclx::array<size_t, 1> triangle_normal_indices;
    sclx::array<size_t, 2> unique_edges;
    sclx::array<bool, 1> double_edges;
    sclx::array<size_t, 2> vertex_face_neighbors;
    sclx::array<size_t, 2> vertex_opposite_edges;
    sclx::array<size_t, 2> vertex_edge_neighbors;
    sclx::array<size_t, 2> edge_opposite_vertices;
    sclx::array<size_t, 2> edge_face_neighbors;
};

template<class T>
sdf_metadata build_sdf(manifold_mesh_t<T> mesh) {
    std::shared_ptr<sdf::Points> sdf_points
        = std::make_shared<sdf::Points>(mesh.vertices.shape()[1], 3);
    for (u_int32_t p = 0; p < mesh.vertices.shape()[1]; p++) {
        sdf_points.get()[0](p, 0) = mesh.vertices(0, p);
        sdf_points.get()[0](p, 1) = mesh.vertices(1, p);
        sdf_points.get()[0](p, 2) = mesh.vertices(2, p);
    }
    std::shared_ptr<sdf::Triangles> sdf_faces
        = std::make_shared<sdf::Triangles>(
            mesh.triangle_vert_indices.shape()[1],
            3
        );
    for (u_int32_t f = 0; f < mesh.triangle_vert_indices.shape()[1]; f++) {
        sdf_faces.get()[0](f, 0) = mesh.triangle_vert_indices(0, f);
        sdf_faces.get()[0](f, 1) = mesh.triangle_vert_indices(1, f);
        sdf_faces.get()[0](f, 2) = mesh.triangle_vert_indices(2, f);
    }
    Eigen::Ref<const sdf::Points> sdf_points_ref(*sdf_points);
    Eigen::Ref<const sdf::Triangles> sdf_faces_ref(*sdf_faces);
    auto sdf = std::make_shared<sdf::SDF>(sdf_points_ref, sdf_faces_ref);
    return {sdf, sdf_points, sdf_faces};
}

template<class T>
std::vector<T>
get_sdf_to_points(const sdf::Points& points, const sdf::SDF& surface) {
    auto sdfs = surface(points);
    return {sdfs.begin(), sdfs.end()};
}

template<class T>
struct vdb_write_data {
    const T* data;
    const bool* out_of_bounds_data;
    T grid_spacing;
    const T* domain_lb;
    const std::size_t* grid_size;
    T acoustic_normalization;
};

template<class T>
struct vdb_writer {
    static void
    write(const std::string& filename, const vdb_write_data<T>& write_data);
};

extern template struct vdb_writer<float>;
extern template struct vdb_writer<double>;

}  // namespace naga::fluids::nonlocal_lbm::detail

namespace naga::fluids::nonlocal_lbm {

template<class Lattice>
class vdb_grid_observer;

template<class T, template<class> class Lattice>
class vdb_grid_observer<Lattice<T>> {
    static_assert(
        !std::is_same_v<d2q9_lattice<T>, Lattice<T>>,
        "d2q9_lattice is not supported by vdb_grid_observer"
    );
};

template<class T>
class vdb_grid_observer<d3q27_lattice<T>>
    : public simulation_observer<d3q27_lattice<T>> {
  public:
    using lattice_t            = d3q27_lattice<T>;
    using base                 = simulation_observer<lattice_t>;
    using value_type           = typename base::value_type;
    using simulation_domain_t  = typename base::simulation_domain_t;
    using problem_parameters_t = typename base::problem_parameters_t;
    using solution_t           = typename base::solution_t;

    vdb_grid_observer(
        sclx::filesystem::path output_directory,
        value_type grid_spacing,
        const simulation_domain_t& input_domain,
        const sclx::filesystem::path input_mesh_obj_file,
        const value_type& acoustic_normalization = 1.,
        const value_type& time_multiplier        = 1,
        const value_type& frame_rate             = 60
    )
        : output_directory_(std::move(output_directory)),
          time_multiplier_(time_multiplier),
          frame_rate_(frame_rate),
          grid_spacing_(grid_spacing),
          acoustic_normalization_(acoustic_normalization) {
        ::naga::segmentation::detail::compute_bounds(
            static_cast<naga::point_view_t<value_type, 3>>(domain_lb_),
            static_cast<naga::point_view_t<value_type, 3>>(domain_ub_),
            input_domain.points
        );

        auto input_mesh = detail::manifold_mesh_t<value_type>::import_from_obj(
            input_mesh_obj_file
        );

        sclx::algorithm::transform(
            input_mesh.normals,
            input_mesh.normals,
            -1,
            sclx::algorithm::multiplies<>{}
        )
            .get();

        auto grid_size_x = static_cast<std::size_t>(
            std::ceil((domain_ub_[0] - domain_lb_[0]) / grid_spacing_) + 1
        );
        domain_ub_[0]    = domain_lb_[0] + grid_size_x * grid_spacing_;
        auto grid_size_y = static_cast<std::size_t>(
            std::ceil((domain_ub_[1] - domain_lb_[1]) / grid_spacing_) + 1
        );
        domain_ub_[1]    = domain_lb_[1] + grid_size_y * grid_spacing_;
        auto grid_size_z = static_cast<std::size_t>(
            std::ceil((domain_ub_[2] - domain_lb_[2]) / grid_spacing_) + 1
        );
        domain_ub_[2] = domain_lb_[2] + grid_size_z * grid_spacing_;
        grid_size_    = naga::point_t<std::size_t, 3>{
            {grid_size_x, grid_size_y, grid_size_z}
        };

        sclx::array<value_type, 2> grid_world_points{
            3,
            static_cast<size_t>(grid_size_x * grid_size_y * grid_size_z)
        };
        grid_world_points
            .prefetch_async(std::vector<int>{sclx::cuda::traits::cpu_device_id})
            .get();
        for (std::int64_t flat_idx = 0; flat_idx < grid_world_points.shape()[1];
             ++flat_idx) {
            auto i = flat_idx % grid_size_x;
            auto j = (flat_idx / grid_size_x) % grid_size_y;
            auto k = flat_idx / (grid_size_x * grid_size_y);
            auto shifted_i
                = static_cast<std::int64_t>((domain_lb_[0] + i * grid_spacing_) / grid_spacing_);
            auto shifted_j
                = static_cast<std::int64_t>((domain_lb_[1] + j * grid_spacing_) / grid_spacing_);
            auto shifted_k
                = static_cast<std::int64_t>((domain_lb_[2] + k * grid_spacing_) / grid_spacing_);
            grid_world_points(0, flat_idx) = shifted_i * grid_spacing_;
            grid_world_points(1, flat_idx) = shifted_j * grid_spacing_;
            grid_world_points(2, flat_idx) = shifted_k * grid_spacing_;
        }

        {
            uint num_interp_points = 32;
            segmentation::
                nd_cubic_segmentation<value_type, lattice_t::dimensions>
                    source_segmentation(input_domain.points, num_interp_points);
            naga::default_point_map<value_type, lattice_t::dimensions>
                grid_point_map{grid_world_points};
            auto [distances_squared, indices]
                = segmentation::batched_nearest_neighbors(
                    num_interp_points,
                    grid_point_map,
                    source_segmentation
                );

            grid_interpolator_ = interpolater_t::create_interpolator(
                input_domain.points,
                indices,
                grid_point_map,
                input_domain.nodal_spacing
            );
        }

        grid_values_ = sclx::array<value_type, 1>{grid_world_points.shape()[1]};
        grid_values_
            .prefetch_async(std::vector<int>{sclx::cuda::traits::cpu_device_id})
            .get();

        out_of_bounds_grid_values_
            = sclx::array<bool, 1>{grid_values_.shape()[0]};
        out_of_bounds_grid_values_.set_primary_devices(
            std::vector<int>{sclx::cuda::traits::cpu_device_id}
        );
        auto sdf_metadata = build_sdf(input_mesh);
        sdf::Points points(grid_world_points.shape()[1], 3);
        for (std::uint32_t i = 0; i < grid_world_points.shape()[1]; ++i) {
            points(i, 0) = grid_world_points(0, i);
            points(i, 1) = grid_world_points(1, i);
            points(i, 2) = grid_world_points(2, i);
        }
        auto sdf_to_points
            = detail::get_sdf_to_points<value_type>(points, *sdf_metadata.sdf);
        for (size_t i = 0; i < grid_values_.shape()[0]; ++i) {
            out_of_bounds_grid_values_(i) = sdf_to_points[i] < 0;
        }
    }

    void update(
        const value_type& time,
        const simulation_domain_t& domain,
        const problem_parameters_t& parameters,
        const solution_t& solution
    ) override {

        const auto& results_path = output_directory_;
        auto save_frame
            = static_cast<size_t>(time_multiplier_ * time * frame_rate_);
        save_frame
            = frame_rate_ != 0.
                ? save_frame
                : static_cast<size_t>(std::floor(time / parameters.time_step));
        if (frame_rate_ != 0 && save_frame < current_frame_) {
            return;
        }
        grid_interpolator_
            .interpolate(
                solution.macroscopic_values.fluid_density,
                grid_values_,
                parameters.nondim_factors.density_scale
            )
            .get();

        sclx::algorithm::transform(
            grid_values_,
            grid_values_,
            parameters.nondim_factors.density_scale,
            sclx::algorithm::minus<>{}
        )
            .get();

        sclx::algorithm::transform(
            grid_values_,
            grid_values_,
            acoustic_normalization_,
            sclx::algorithm::divides<>{}
        )
            .get();

        std::string filename
            = results_path
            / (std::string("result.") + std::to_string(current_frame_) + ".vdb");

        detail::vdb_write_data<value_type> write_data;
        write_data.acoustic_normalization = acoustic_normalization_;
        write_data.data                   = grid_values_.data().get();
        write_data.out_of_bounds_data = out_of_bounds_grid_values_.data().get();
        write_data.grid_spacing       = grid_spacing_;
        write_data.domain_lb          = &domain_lb_[0];
        write_data.grid_size          = &grid_size_[0];
        detail::vdb_writer<value_type>::write(filename, write_data);
        ++current_frame_;
    }

    template<class Archive>
    void save_state(Archive& ar) const {
        ar(current_frame_);
    }

    template<class Archive>
    void load_state(Archive& ar) {
        ar(current_frame_);
    }

  private:
    sclx::filesystem::path output_directory_;
    value_type time_multiplier_;
    value_type frame_rate_;
    value_type grid_spacing_;
    sclx::array<value_type, 1> grid_values_;
    sclx::array<bool, 1> out_of_bounds_grid_values_;
    value_type acoustic_normalization_;

    using interpolater_t = interpolation::radial_point_method<value_type>;
    interpolater_t grid_interpolator_;

    // default initialized
    naga::point_t<value_type, 3> domain_lb_{
        {std::numeric_limits<value_type>::max(),
         std::numeric_limits<value_type>::max(),
         std::numeric_limits<value_type>::max()}
    };
    naga::point_t<value_type, 3> domain_ub_{
        {std::numeric_limits<value_type>::lowest(),
         std::numeric_limits<value_type>::lowest(),
         std::numeric_limits<value_type>::lowest()}
    };
    size_t current_frame_ = 0;

    naga::point_t<std::size_t, 3> grid_size_{};
};

}  // namespace naga::fluids::nonlocal_lbm
