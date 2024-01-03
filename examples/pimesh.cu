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
#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>
#include <unordered_map>
#include <naga/point.hpp>

#define TINYOBJLOADER_IMPLEMENTATION  // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation.
// Requires C++11
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

#include <naga/distance_functions.hpp>
#include <scalix/fill.cuh>

namespace detail {

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

}  // namespace detail

template<>
struct std::hash<detail::edge_t> {
    using argument_type = detail::edge_t;
    auto operator()(const argument_type& edge) const noexcept -> size_t {
        return std::hash<size_t>()(edge.i) ^ std::hash<size_t>()(edge.j);
    }
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
        size_t total_face_count   = 0;
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
struct closed_contour_t {

    using manifold_mesh_t = manifold_mesh_t<T>;

    static auto import_from_obj(sclx::filesystem::path obj_path)
        -> closed_contour_t {
        closed_contour_t contour;

        contour.input_mesh = manifold_mesh_t::import_from_obj(obj_path);

        std::set<size_t> contour_edges;
        std::unordered_map<size_t, uint> vertices_and_counts;
        for (size_t i = 0; i < contour.input_mesh.unique_edges.shape()[1];
             ++i) {
            if (!contour.input_mesh.double_edges(i)) {
                contour_edges.insert(i);

                auto v1idx     = contour.input_mesh.unique_edges(0, i);
                auto& v1_count = vertices_and_counts[v1idx];
                ++v1_count;
                if (v1_count > 2) {
                    throw std::runtime_error("closed_contour_t: vertex is "
                                             "shared by more than two edges, "
                                             "this likely indicates bug in naga"
                    );
                }

                auto v2idx     = contour.input_mesh.unique_edges(1, i);
                auto& v2_count = vertices_and_counts[v2idx];
                ++v2_count;
                if (v2_count > 2) {
                    throw std::runtime_error("closed_contour_t: vertex is "
                                             "shared by more than two edges, "
                                             "this likely indicates bug in naga"
                    );
                }
            }
        }

        contour.contour_edges_in_input_mesh
            = sclx::array<size_t, 1>{contour_edges.size()};
        contour.contour_vertices_in_input_mesh
            = sclx::array<size_t, 1>{vertices_and_counts.size()};
        std::copy(
            contour_edges.begin(),
            contour_edges.end(),
            &contour.contour_edges_in_input_mesh(0)
        );
        std::transform(
            vertices_and_counts.begin(),
            vertices_and_counts.end(),
            &contour.contour_vertices_in_input_mesh(0),
            [](const auto& pair) { return pair.first; }
        );

        return contour;
    }

    manifold_mesh_t input_mesh;
    sclx::array<size_t, 1> contour_vertices_in_input_mesh;
    sclx::array<size_t, 1> contour_edges_in_input_mesh;
};

template<class T>
struct sdf2d_result {
    using closest_edge_t    = size_t;
    using signed_distance_t = T;

    closest_edge_t closest_edge;
    signed_distance_t signed_distance;
};

template <class T>
auto compute_contour_normals(const closed_contour_t<T>& contour) -> sclx::array<T, 2> {
    sclx::array<T, 2> edge_normals{2, contour.contour_edges_in_input_mesh.elements()};
    for (size_t e = 0;
         e < contour.contour_edges_in_input_mesh.elements();
         ++e) {
        auto edge_idx = contour.contour_edges_in_input_mesh(e);
        const T* x1 = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(0, edge_idx)
        );
        const T* x2 = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(1, edge_idx)
        );
        const T x12[2]{x2[0] - x1[0], x2[1] - x1[1]};
        T edge_normal[2]{-x12[1], x12[0]};

        naga::math::loopless::normalize<2>(edge_normal);

        const T* opposite_vertex = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.edge_opposite_vertices(0, edge_idx)
        );
        const T opp2x1[2]{x1[0] - opposite_vertex[0], x1[1] - opposite_vertex[1]};
        auto opp2x1_dot_edge_normal = naga::math::loopless::dot<2>(
            opp2x1,
            edge_normal
        );
        if (opp2x1_dot_edge_normal < 0) {
            edge_normal[0] *= -1;
            edge_normal[1] *= -1;
        }

        edge_normals(0, e) = edge_normal[0];
        edge_normals(1, e) = edge_normal[1];
    }

    return edge_normals;
}
template <class T>
struct contour_bounds {
    naga::point_t<T, 2> min;
    naga::point_t<T, 2> max;
};

template <class T>
auto compute_contour_bounds(const closed_contour_t<T> contour) -> contour_bounds<T> {
    contour_bounds<T> bounds;
    bounds.min[0] = std::numeric_limits<T>::max();
    bounds.min[1] = std::numeric_limits<T>::max();
    bounds.max[0] = std::numeric_limits<T>::lowest();
    bounds.max[1] = std::numeric_limits<T>::lowest();
    for (const auto& vidx: contour.contour_vertices_in_input_mesh) {
        if (contour.input_mesh.vertices(0, vidx) < bounds.min[0]) {
            bounds.min[0] = contour.input_mesh.vertices(0, vidx);
        }
        if (contour.input_mesh.vertices(1, vidx) < bounds.min[1]) {
            bounds.min[1] = contour.input_mesh.vertices(1, vidx);
        }
        if (contour.input_mesh.vertices(0, vidx) > bounds.max[0]) {
            bounds.max[0] = contour.input_mesh.vertices(0, vidx);
        }
        if (contour.input_mesh.vertices(1, vidx) > bounds.max[1]) {
            bounds.max[1] = contour.input_mesh.vertices(1, vidx);
        }
    }

    return bounds;
}

template<class PointType, class T>
__host__ __device__ auto get_sdf2d(
    const PointType& point,
    const closed_contour_t<T>& contour,
    const sclx::array<T, 2>& edge_normals
)

    -> sdf2d_result<T> {
    sdf2d_result<T> result{};

    result.signed_distance = std::numeric_limits<T>::max();
    for (size_t e = 0;
         e < contour.contour_edges_in_input_mesh.elements();
         ++e) {
        auto edge_idx = contour.contour_edges_in_input_mesh(e);
        const T* x1 = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(0, edge_idx)
        );
        const T* x2 = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(1, edge_idx)
        );
        const T* edge_normal = &edge_normals(0, e);
        T perpendicular2edge_intersection[2];
        if (naga::math::abs(edge_normal[0]) < 1e-6) {
            perpendicular2edge_intersection[0] = point[0];
            perpendicular2edge_intersection[1] = x1[1];
        } else if (naga::math::abs(edge_normal[1]) < 1e-6) {
            perpendicular2edge_intersection[0] = x1[0];
            perpendicular2edge_intersection[1] = point[1];
        } else {
            const T edge_slope = -edge_normal[0] / edge_normal[1];
            const T edge_b     = x1[1] - edge_slope * x1[0];

            const T perpendicular_slope = -1.0 / edge_slope;
            const T perpendicular_b     = point[1] - perpendicular_slope * point[0];

            perpendicular2edge_intersection[0] = (edge_b - perpendicular_b) / (perpendicular_slope - edge_slope);
            perpendicular2edge_intersection[1] = perpendicular_slope * (edge_b - perpendicular_b) / (perpendicular_slope - edge_slope) + perpendicular_b;
        }
        const T x12[2]{x2[0] - x1[0], x2[1] - x1[1]};

        auto edge_length = naga::distance_functions::loopless::euclidean<2>{
            }(x1, x2);
        const T x1_to_intersection_vector[2] {  perpendicular2edge_intersection[0] - x1[0],
                                                perpendicular2edge_intersection[1] - x1[1]};
        const T point_to_intersection_vector[2] {  perpendicular2edge_intersection[0] - point[0],
                                                   perpendicular2edge_intersection[1] - point[1]};

        T normalized_x12[2]{x12[0] / edge_length, x12[1] / edge_length};
        auto x1_to_intersection_dot_normalized_x12 = naga::math::loopless::dot<2>(
            x1_to_intersection_vector,
            normalized_x12
        );
        T signed_distance;
        if (x1_to_intersection_dot_normalized_x12 < 1e-6) {
            signed_distance = naga::distance_functions::loopless::euclidean<2>{}(x1, point);
        } else if (x1_to_intersection_dot_normalized_x12 < edge_length) {
            signed_distance = naga::distance_functions::loopless::euclidean<2>{
            }(point, perpendicular2edge_intersection);
        } else {
            signed_distance = naga::distance_functions::loopless::euclidean<2>{}(x2, point);
        }
        auto point2intersection_dot_edge_normal = naga::math::loopless::dot<2>(
            point_to_intersection_vector,
            edge_normal
        );

#ifndef __CUDA_ARCH__

        std::cout << "point: " << point[0] << ", " << point[1] << "\n";
        std::cout << "x1: " << x1[0] << ", " << x1[1] << "\n";
        std::cout << "x2: " << x2[0] << ", " << x2[1] << "\n";
        std::cout << "edge_normal: " << edge_normal[0] << ", " << edge_normal[1] << "\n";
        std::cout << "perpendicular2edge_intersection: " << perpendicular2edge_intersection[0] << ", " << perpendicular2edge_intersection[1] << "\n";
        std::cout << "edge_length: " << edge_length << "\n";
        std::cout << "x1_to_intersection_dot_normalized_x12: " << x1_to_intersection_dot_normalized_x12 << "\n";
        std::cout << "signed_distance: " << signed_distance << "\n\n";
#endif

        if (naga::math::abs(signed_distance) >= naga::math::abs(result.signed_distance)) {
            continue;
        }

        if (point2intersection_dot_edge_normal < 0) {
            signed_distance *= -1;
        }

        result.signed_distance = signed_distance;
        result.closest_edge     = contour.contour_edges_in_input_mesh(e);
    }

    return result;
}

auto main() -> int {
    auto mesh = manifold_mesh_t<float>::import_from_obj(sclx::filesystem::path(
        "naga/resources/lbm_example_domains/rectangle/domain.obj"
    ));

    // resave faces to domain_imported.obj
    std::ofstream obj_file(
        "naga/resources/lbm_example_domains/rectangle/domain_imported.obj"
    );
    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
        obj_file << "v " << mesh.vertices(0, i) << " " << mesh.vertices(1, i)
                 << " " << mesh.vertices(2, i) << "\n";
    }
    for (size_t i = 0; i < mesh.triangle_vert_indices.shape()[1]; ++i) {
        obj_file << "f " << mesh.triangle_vert_indices(0, i) + 1 << " "
                 << mesh.triangle_vert_indices(1, i) + 1 << " " << mesh.triangle_vert_indices(2, i) + 1
                 << "\n";
    }
    obj_file.close();

    // resave edges to domain_imported_edges.obj
    std::ofstream obj_file_edges(
        "naga/resources/lbm_example_domains/rectangle/domain_imported_edges.obj"
    );
    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
        obj_file_edges << "v " << mesh.vertices(0, i) << " "
                       << mesh.vertices(1, i) << " " << mesh.vertices(2, i)
                       << "\n";
    }
    for (size_t i = 0; i < mesh.unique_edges.shape()[1]; ++i) {
        obj_file_edges << "l " << mesh.unique_edges(0, i) + 1 << " "
                       << mesh.unique_edges(1, i) + 1 << "\n";
    }
    obj_file_edges.close();

    auto contour
        = closed_contour_t<float>::import_from_obj(sclx::filesystem::path(
            "naga/resources/lbm_example_domains/rectangle/domain.obj"
        ));

    // resave edges to domain_imported_contour.obj
    std::ofstream obj_file_contour("naga/resources/lbm_example_domains/"
                                   "rectangle/domain_imported_contour.obj");
    for (size_t i = 0; i < contour.input_mesh.vertices.shape()[1]; ++i) {
        obj_file_contour << "v " << mesh.vertices(0, i) << " "
                         << mesh.vertices(1, i) << " " << mesh.vertices(2, i)
                         << "\n";
    }
    for (size_t i = 0; i < contour.contour_edges_in_input_mesh.shape()[0];
         ++i) {
        obj_file_contour
            << "l "
            << mesh.unique_edges(0, contour.contour_edges_in_input_mesh(i)) + 1
            << " "
            << mesh.unique_edges(1, contour.contour_edges_in_input_mesh(i)) + 1
            << "\n";
    }
    obj_file_contour.close();

    auto contour_edge_normals = compute_contour_normals(contour);
    // save contour vertices w/ normasl to domain_imported_contour_normals.csv
    std::ofstream contour_normals_file("naga/resources/lbm_example_domains/"
                                       "rectangle/domain_imported_contour_normals.csv");
    contour_normals_file << "x,y,nx,ny\n";
    for (size_t e = 0; e < contour.contour_edges_in_input_mesh.elements(); ++e) {
        auto edge_idx = contour.contour_edges_in_input_mesh(e);
        float center_point[2] {
            (contour.input_mesh.vertices(0, contour.input_mesh.unique_edges(0, edge_idx))
             + contour.input_mesh.vertices(0, contour.input_mesh.unique_edges(1, edge_idx)))
                / 2,
            (contour.input_mesh.vertices(1, contour.input_mesh.unique_edges(0, edge_idx))
             + contour.input_mesh.vertices(1, contour.input_mesh.unique_edges(1, edge_idx)))
                / 2
        };
        contour_normals_file << center_point[0] << ","
                             << center_point[1] << ","
                             << contour_edge_normals(0, e) << ","
                             << contour_edge_normals(1, e) << "\n";
    }
    contour_normals_file.close();

    auto contour_bounds = compute_contour_bounds(contour);
    float node_separation = 0.2;
    size_t num_nodes_x = static_cast<size_t>(
        (contour_bounds.max[0] - contour_bounds.min[0]) / node_separation
    );
    size_t num_nodes_y = static_cast<size_t>(
        (contour_bounds.max[1] - contour_bounds.min[1]) / node_separation
    );
    sclx::array<float, 2> node_positions{2, num_nodes_x * num_nodes_y};
    for (size_t i = 0; i < num_nodes_x; ++i) {
        for (size_t j = 0; j < num_nodes_y; ++j) {
            node_positions(0, i * num_nodes_y + j) = contour_bounds.min[0] + i * node_separation;
            node_positions(1, i * num_nodes_y + j) = contour_bounds.min[1] + j * node_separation;
        }
    }
    sclx::array<sdf2d_result<float>, 1> sdf_results{node_positions.shape()[1]};

    sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
        ctx.launch(sclx::md_range_t<1>{sdf_results.shape()},
            sdf_results,
            [=] __device__ (const sclx::md_index_t<1>& idx, const auto&) {
            sdf_results[idx] = get_sdf2d(
                &node_positions(0, idx[0]),
                contour,
                contour_edge_normals
            );
        });
    });
//    for (size_t i = 0; i < sdf_results.shape()[0]; ++i) {
//        sdf_results(i) = get_sdf2d(
//            &node_positions(0, i),
//            contour,
//            contour_edge_normals
//        );
//    }

    // save node positions and sdf results to sdf2d_results.csv
    std::ofstream sdf2d_results_file("naga/resources/lbm_example_domains/"
                                     "rectangle/sdf2d_results.csv");
    sdf2d_results_file << "x,y,sdf\n";
    for (size_t i = 0; i < node_positions.shape()[1]; ++i) {
        sdf2d_results_file << node_positions(0, i) << ","
                           << node_positions(1, i) << ","
                           << sdf_results(i).signed_distance << "\n";
    }
    sdf2d_results_file.close();

    return 0;
}