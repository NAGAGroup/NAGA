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
#include <naga/distance_functions.hpp>
#include <naga/point.hpp>
#include <naga/segmentation/nd_cubic_segmentation.cuh>
#include <scalix/algorithm/reduce.cuh>
#include <scalix/algorithm/reduce_last_dim.cuh>
#include <scalix/algorithm/transform.cuh>
#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>
#include <scalix/fill.cuh>
#include <thrust/random.h>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION  // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation.
// Requires C++11
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

#include <naga/segmentation/nearest_neighbors.cuh>

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

class spinlock {
  public:
    void lock() {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                break;
            }
            while (lock_.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }
        }
    }

    void unlock() { lock_.store(false, std::memory_order_release); }

  private:
    std::atomic<bool> lock_{false};
};

template<class T>
struct manifold_mesh_t {
    using value_type = T;

    static constexpr auto no_face   = std::numeric_limits<size_t>::max();
    static constexpr auto no_edge   = std::numeric_limits<size_t>::max();
    static constexpr auto no_vertex = std::numeric_limits<size_t>::max();

    using edge_t       = detail::edge_t;
    using edge_count_t = int;

    static auto import_from_obj(
        const sclx::filesystem::path& obj_path,
        const bool flip_normals = false
    ) -> manifold_mesh_t {

        using namespace detail;
        manifold_mesh_t mesh{};

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

        spinlock lock;
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

            face_vertex_indices.reserve(3 * (attrib.vertices.size() - 2));
            face_normal_indices.reserve(attrib.vertices.size() - 2);
            edge_opposite_vertices.reserve(3 * (attrib.vertices.size() - 2));
            edge_face_neighbors.reserve(3 * (attrib.vertices.size() - 2));
            vertex_face_neighbors.reserve(attrib.vertices.size());
            vertex_opposite_edges.reserve(attrib.vertices.size());
            vertex_edge_neighbors.reserve(attrib.vertices.size());
            edge_counts.reserve(3 * (attrib.vertices.size() - 2));

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
#pragma omp parallel for
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size();
                     ++f) {
                    for (size_t i = 0; i < 3; ++i) {
                        // face data
                        {
                            std::lock_guard<spinlock> guard(lock);
                            face_vertex_indices.push_back(
                                shape.mesh.indices[f * 3 + i].vertex_index
                            );
                            if (i == 0) {
                                face_normal_indices.push_back(
                                    shape.mesh.indices[f * 3 + i].normal_index
                                );
                            }
                        }

                        // edge data
                        size_t edge_idx;
                        size_t edge_opposite_vertex;
                        {
                            std::lock_guard<spinlock> guard(lock);
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
                                shape.mesh.indices[f * 3 + (i + 2) % 3]
                                    .vertex_index
                            );
                            edge_opposite_vertex
                                = edge_opposite_vertices[edge].back();
                            edge_face_neighbors[edge].push_back(total_face_count
                            );

                            auto edge_it = edge_counts.find({edge.i, edge.j});
                            edge_idx     = static_cast<size_t>(
                                std::distance(edge_counts.begin(), edge_it)
                            );
                        }

                        // vertex data
                        {
                            std::lock_guard<spinlock> guard(lock);
                            vertex_face_neighbors[shape.mesh.indices[f * 3 + i]
                                                      .vertex_index]
                                .push_back(total_face_count);
                            vertex_edge_neighbors[shape.mesh.indices[f * 3 + i]
                                                      .vertex_index]
                                .push_back(edge_idx);
                            vertex_opposite_edges[edge_opposite_vertex]
                                .push_back(edge_idx);
                        }
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
        mesh.triangle_normal_indices = sclx::array<size_t, 1>{total_face_count};
        std::copy(
            face_normal_indices.begin(),
            face_normal_indices.end(),
            &mesh.triangle_normal_indices(0)
        );

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
            std::cout << "number of face neighbors for vertex " << i << " is "
                      << vertex_face_neighbors[i].size() << std::endl;
            for (size_t j = 0; j < vertex_face_neighbors[i].size(); ++j) {
                mesh.vertex_face_neighbors(j, i) = vertex_face_neighbors[i][j];
                mesh.vertex_opposite_edges(j, i) = vertex_opposite_edges[i][j];
            }
            for (size_t j = 0; j < vertex_edge_neighbors[i].size(); ++j) {
                mesh.vertex_edge_neighbors(j, i) = vertex_edge_neighbors[i][j];
            }
        }

        auto lower_bounds_reduce = sclx::algorithm::reduce_last_dim(
            mesh.vertices,
            std::numeric_limits<T>::max(),
            sclx::algorithm::min<>()
        );
        auto upper_bounds_reduce = sclx::algorithm::reduce_last_dim(
            mesh.vertices,
            std::numeric_limits<T>::lowest(),
            sclx::algorithm::max<>()
        );

        for (uint i = 0; i < 3; ++i) {
            mesh.lower_bound[i] = lower_bounds_reduce[i];
            mesh.upper_bound[i] = upper_bounds_reduce[i];
        }

        if (flip_normals) {
            sclx::algorithm::transform(
                mesh.normals,
                mesh.normals,
                -1,
                sclx::algorithm::multiplies<>{}
            )
                .get();
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

    naga::point_t<value_type, 3> lower_bound;
    naga::point_t<value_type, 3> upper_bound;
};

template<class T>
struct closed_contour_t;

template<class T>
auto compute_contour_normals(const closed_contour_t<T>& contour)
    -> sclx::array<T, 2>;

template<class T>
struct closed_contour_t {

    using manifold_mesh_t = manifold_mesh_t<T>;

    static auto import_from_obj(
        sclx::filesystem::path obj_path,
        const bool flip_normals = false
    ) -> closed_contour_t {
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

        contour.edge_normals = compute_contour_normals(contour);

        if (flip_normals) {
            sclx::algorithm::transform(
                contour.edge_normals,
                contour.edge_normals,
                -1,
                sclx::algorithm::multiplies<>{}
            )
                .get();
        }

        return contour;
    }

    manifold_mesh_t input_mesh;
    sclx::array<size_t, 1> contour_vertices_in_input_mesh;
    sclx::array<size_t, 1> contour_edges_in_input_mesh;
    sclx::array<T, 2> edge_normals;
};

template<class T>
struct sdf2d_result {
    using closest_edge_t    = size_t;
    using signed_distance_t = T;

    closest_edge_t closest_edge;
    signed_distance_t signed_distance;
};

template<class T>
auto compute_contour_normals(const closed_contour_t<T>& contour)
    -> sclx::array<T, 2> {
    sclx::array<T, 2> edge_normals{
        2,
        contour.contour_edges_in_input_mesh.elements()
    };
    for (size_t e = 0; e < contour.contour_edges_in_input_mesh.elements();
         ++e) {
        auto edge_idx = contour.contour_edges_in_input_mesh(e);
        const T* x1   = &contour.input_mesh.vertices(
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
        const T opp2x1[2]{
            x1[0] - opposite_vertex[0],
            x1[1] - opposite_vertex[1]
        };
        auto opp2x1_dot_edge_normal
            = naga::math::loopless::dot<2>(opp2x1, edge_normal);
        if (opp2x1_dot_edge_normal > 0) {
            edge_normal[0] *= -1;
            edge_normal[1] *= -1;
        }

        edge_normals(0, e) = edge_normal[0];
        edge_normals(1, e) = edge_normal[1];
    }

    return edge_normals;
}

template<class PointType, class T>
__host__ __device__ auto
get_sdf2d(const PointType& point, const closed_contour_t<T>& contour)

    -> sdf2d_result<T> {
    sdf2d_result<T> result{};

    result.signed_distance = std::numeric_limits<T>::max();
    for (size_t e = 0; e < contour.contour_edges_in_input_mesh.elements();
         ++e) {
        auto edge_idx = contour.contour_edges_in_input_mesh(e);
        const T* x1   = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(0, edge_idx)
        );
        const T* x2 = &contour.input_mesh.vertices(
            0,
            contour.input_mesh.unique_edges(1, edge_idx)
        );
        const T* edge_normal = &contour.edge_normals(0, e);
        T point2edge_intersection[2];
        if (naga::math::abs(edge_normal[0]) < 1e-6) {
            point2edge_intersection[0] = point[0];
            point2edge_intersection[1] = x1[1];
        } else if (naga::math::abs(edge_normal[1]) < 1e-6) {
            point2edge_intersection[0] = x1[0];
            point2edge_intersection[1] = point[1];
        } else {
            point2edge_intersection[0]
                = (edge_normal[1] * (point[0] - x1[0]) + edge_normal[0] * x1[1]
                   - edge_normal[0] * point[1])
                / edge_normal[1];
            point2edge_intersection[1]
                = (edge_normal[0] * (point[1] - x1[1]) + edge_normal[1] * x1[0]
                   - edge_normal[1] * point[0])
                / edge_normal[0];
        }
        const T x12[2]{x2[0] - x1[0], x2[1] - x1[1]};

        auto edge_length
            = naga::distance_functions::loopless::euclidean<2>{}(x1, x2);
        const T x1_to_intersection_vector[2]{
            point2edge_intersection[0] - x1[0],
            point2edge_intersection[1] - x1[1]
        };
        const T point_to_intersection_vector[2]{
            point2edge_intersection[0] - point[0],
            point2edge_intersection[1] - point[1]
        };

        T normalized_x12[2]{x12[0] / edge_length, x12[1] / edge_length};
        auto x1_to_intersection_dot_normalized_x12
            = naga::math::loopless::dot<2>(
                x1_to_intersection_vector,
                normalized_x12
            );
        T signed_distance;
        if (x1_to_intersection_dot_normalized_x12 < 1e-6) {
            signed_distance
                = naga::distance_functions::loopless::euclidean<2>{}(x1, point);
        } else if (x1_to_intersection_dot_normalized_x12 < edge_length) {
            signed_distance
                = naga::math::loopless::norm<2>(point_to_intersection_vector);
        } else {
            signed_distance
                = naga::distance_functions::loopless::euclidean<2>{}(x2, point);
        }
        auto point2intersection_dot_edge_normal = naga::math::loopless::dot<2>(
            point_to_intersection_vector,
            edge_normal
        );

        if (point2intersection_dot_edge_normal < 0) {
            signed_distance *= -1;
        }

        if (naga::math::abs(signed_distance)
            > naga::math::abs(result.signed_distance)) {
            continue;
        }
        result.signed_distance = signed_distance;
        result.closest_edge    = contour.contour_edges_in_input_mesh(e);
    }

    return result;
}

template<class T>
sclx::array<sdf2d_result<T>, 1> batched_sdf2d(
    const sclx::array<T, 2>& points,
    const closed_contour_t<T>& contour
)

{
    sclx::array<sdf2d_result<T>, 1> results{points.shape()[1]};

    sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
        ctx.launch(
            sclx::md_range_t<1>{results.shape()},
            results,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                results[idx] = get_sdf2d(&points(0, idx[0]), contour);
            }
        );
    });

    return results;
}

template<class T>
struct batched_sdf2d_result {
    sclx::array<sdf2d_result<T>, 1> results;
    sclx::array<size_t, 1> input_contour_idx;
};

template<class T>
batched_sdf2d_result<T> compute_combined_sdf2d(
    const sclx::array<T, 2>& query_points,
    const std::vector<closed_contour_t<T>>& all_contours
) {
    sdf2d_result<T> max_possible_sdf2d_result{};
    max_possible_sdf2d_result.signed_distance = std::numeric_limits<T>::max();
    sclx::array<sdf2d_result<T>, 1> min_sdf2d_results{query_points.shape()[1]};
    sclx::fill(min_sdf2d_results, max_possible_sdf2d_result);

    std::vector<sclx::array<sdf2d_result<T>, 1>> injection_sdf2d_all_contours;
    for (const auto& contour : all_contours) {
        injection_sdf2d_all_contours.push_back(
            batched_sdf2d(query_points, contour)
        );
    }

    sclx::array<size_t, 1> closest_contour_indices{query_points.shape()[1]};
    for (size_t ctr_idx = 0; ctr_idx < all_contours.size(); ++ctr_idx) {
        auto ctr_sdf2d_results = injection_sdf2d_all_contours[ctr_idx];
        sclx::
            array_tuple<sclx::array<sdf2d_result<T>, 1>, sclx::array<size_t, 1>>
                result_tuple{min_sdf2d_results, closest_contour_indices};
        sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
            ctx.launch(
                sclx::md_range_t<1>{query_points.shape()[1]},
                result_tuple,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    if (naga::math::abs(ctr_sdf2d_results[idx].signed_distance)
                        < naga::math::abs(min_sdf2d_results[idx].signed_distance
                        )) {
                        min_sdf2d_results[idx]       = ctr_sdf2d_results[idx];
                        closest_contour_indices[idx] = ctr_idx;
                    }
                }
            );
        }).get();
    }
    return {min_sdf2d_results, closest_contour_indices};
}

template<class T>
sclx::array<T, 2> fill_2d_domain_with_grid(
    const T& nodal_spacing,
    const closed_contour_t<T>& domain_contour,
    const std::vector<closed_contour_t<T>>& immersed_contours,
    T max_sdf2d_distance = std::numeric_limits<T>::infinity()
) {
    max_sdf2d_distance
        = max_sdf2d_distance == std::numeric_limits<T>::infinity()
            ? nodal_spacing
            : max_sdf2d_distance;
    auto num_nodes_x = static_cast<size_t>(
                           (domain_contour.input_mesh.upper_bound[0]
                            - domain_contour.input_mesh.lower_bound[0])
                           / nodal_spacing
                       )
                     + 1;
    auto num_nodes_y = static_cast<size_t>(
                           (domain_contour.input_mesh.upper_bound[1]
                            - domain_contour.input_mesh.lower_bound[1])
                           / nodal_spacing
                       )
                     + 1;
    sclx::array<T, 2> potential_points{2, num_nodes_x * num_nodes_y};
    for (size_t i = 0; i < num_nodes_x; ++i) {
        for (size_t j = 0; j < num_nodes_y; ++j) {
            potential_points(0, i * num_nodes_y + j)
                = domain_contour.input_mesh.lower_bound[0] + i * nodal_spacing;
            potential_points(1, i * num_nodes_y + j)
                = domain_contour.input_mesh.lower_bound[1] + j * nodal_spacing;
        }
    }

    std::vector<closed_contour_t<T>> all_contours{domain_contour};
    all_contours.insert(
        all_contours.end(),
        immersed_contours.begin(),
        immersed_contours.end()
    );
    auto combined_sdf2d_results
        = compute_combined_sdf2d(potential_points, all_contours);

    const auto& min_sdf2d_results = combined_sdf2d_results.results;

    sclx::array<bool, 1> out_of_bounds_points{potential_points.shape()[1]};
    sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
        ctx.launch(
            sclx::md_range_t<1>{out_of_bounds_points.shape()},
            out_of_bounds_points,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                out_of_bounds_points[idx]
                    = min_sdf2d_results[idx].signed_distance <= nodal_spacing;
            }
        );
    }).get();

    std::vector<T> in_bounds_points;
    in_bounds_points.reserve(2 * out_of_bounds_points.elements());

    out_of_bounds_points
        .prefetch_async(std::vector<int>{sclx::cuda::traits::cpu_device_id})
        .get();
    potential_points
        .prefetch_async(std::vector<int>{sclx::cuda::traits::cpu_device_id})
        .get();
    for (size_t i = 0; i < out_of_bounds_points.elements(); ++i) {
        if (!out_of_bounds_points(i)) {
            in_bounds_points.push_back(potential_points(0, i));
            in_bounds_points.push_back(potential_points(1, i));
        }
    }
    if (in_bounds_points.size() == 0) {
        throw std::runtime_error(
            "fill_2d_domain_with_grid: no injection points found"
        );
    }

    sclx::array<T, 2> points{2, in_bounds_points.size() / 2};
    std::copy(in_bounds_points.begin(), in_bounds_points.end(), points.begin());

    return points;
}

template<class T>
sclx::array<T, 2> create_pimesh2d_injection_sites(
    const T& nodal_spacing,
    const closed_contour_t<T>& domain_contour,
    const std::vector<closed_contour_t<T>>& immersed_contours
) {
    auto injection_spacing = nodal_spacing;

    return fill_2d_domain_with_grid(
        injection_spacing,
        domain_contour,
        immersed_contours,
        -.1f * nodal_spacing
    );
}

template<class T>
struct domain2d {
    sclx::array<T, 2> points;
    size_t num_bulk_points;
    size_t num_layer_points;
    size_t num_boundary_points;
};

template<class T>
__host__ __device__ T pimesh_kernel_function(const T& alpha, const T& q) {
    if (q >= 2.f) {
        return 0.f;
    } else if (q >= 1.f) {
        return alpha * naga::math::loopless::pow<3>(2 - q);
    } else {
        return alpha
             * (naga::math::loopless::pow<3>(2 - q)
                - 4 * naga::math::loopless::pow<3>(1 - q));
    }
}

template<class T, uint Dimensions>
struct pimesh_kernel_alpha;

template<class T>
struct pimesh_kernel_alpha<T, 2> {
    __host__ __device__ static constexpr auto value() -> T { return 1.f / 6.f; }
};

template<class T>
struct pimesh_kernel_alpha<T, 3> {
    __host__ __device__ static constexpr auto value() -> T { return 1. / 18.f; }
};

template<class T>
void pimesh2d_inject_points(
    sclx::array<T, 2> injection_sites,
    size_t num_injected_per_site,
    sclx::array<T, 2> points,
    sclx::array<T, 2> velocities
) {
    const auto& num_injection_sites = injection_sites.shape()[1];
    const auto& total_injections_per_site
        = points.shape()[1] / num_injection_sites;

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto velocities_fut
        = sclx::execute_kernel([&](sclx::kernel_handler& handle) {
              handle.launch(
                  sclx::md_range_t<1>{points.shape()[1]},
                  velocities,
                  [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                      auto injection_idx = idx[0] % total_injections_per_site;
                      if (injection_idx != num_injected_per_site) {
                          return;
                      }
                      thrust::default_random_engine rng(seed);
                      thrust::uniform_real_distribution<T>
                          dist_angle(0, 2 * naga::math::pi<T>);
                      rng.discard(idx[0]);
                      auto angle            = dist_angle(rng);
                      velocities(0, idx[0]) = 0.f * naga::math::cos(angle);
                      velocities(1, idx[0]) = 0.f * naga::math::sin(angle);
                  }
              );
          });
    auto points_fut = sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{points.shape()[1]},
            points,
            [total_injections_per_site,
             num_injected_per_site,
             points,
             injection_sites] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                auto injection_site_idx = idx[0] / total_injections_per_site;
                auto injection_idx      = idx[0] % total_injections_per_site;
                if (injection_idx != num_injected_per_site) {
                    return;
                }
                points(0, idx[0]) = injection_sites(0, injection_site_idx);
                points(1, idx[0]) = injection_sites(1, injection_site_idx);
            }
        );
    });

    points_fut.get();
    velocities_fut.get();
}

template<class T>
domain2d<T> pimesh2d_generate_domain(
    const T& nodal_spacing,
    const closed_contour_t<T>& domain_contour,
    const std::vector<closed_contour_t<T>>& immersed_contours
) {
    auto injection_sites = create_pimesh2d_injection_sites(
        nodal_spacing,
        domain_contour,
        immersed_contours
    );
    auto points = injection_sites;

    sclx::array<T, 2> node_velocities{points.shape()};
    sclx::fill(node_velocities, T{0});
    sclx::array<T, 2> node_forces{points.shape()};
    sclx::array<T, 1> distance_travelled{points.shape()[1]};

    std::vector<closed_contour_t<T>> all_contours{domain_contour};
    all_contours.insert(
        all_contours.end(),
        immersed_contours.begin(),
        immersed_contours.end()
    );

    const auto delta_t = 1.f;
    T drag_coefficient = .05f;
    // Define the following:
    // q_nom = 1.f;
    // force_mag_nom = pimesh_kernel_function(pimesh_kernel_alpha<T,
    // 2>::value(), q_nom); displacement_mag_nom = 0.5f * force_mag_nom *
    // delta_t * delta_t;
    //
    // Then:
    // We want a value for kf such that displacement_mag_nom * kf = .01f *
    // nodal_spacing
    auto q_nom = 1.f;
    auto force_mag_nom
        = pimesh_kernel_function(pimesh_kernel_alpha<T, 2>::value(), q_nom);
    auto displacement_mag_nom = 0.5f * force_mag_nom * delta_t * delta_t;
    auto kf                   = .0005f * nodal_spacing / displacement_mag_nom;

    size_t num_injected_per_site = 0;
    T current_time               = 0.f;

    while (true) {
        //        if (num_injected_per_site < total_injections_per_site) {
        //            pimesh2d_inject_points(
        //                injection_sites,
        //                num_injected_per_site,
        //                points,
        //                node_velocities
        //            );
        //            ++num_injected_per_site;
        //        }

        sclx::fill(node_forces, T{0});

        sclx::
            array_tuple<sclx::array<T, 2>, sclx::array<T, 2>, sclx::array<T, 1>>
                positions_velocities_and_distances{
                    points,
                    node_velocities,
                    distance_travelled
                };

        {
            naga::segmentation::nd_cubic_segmentation<T, 2> points_segmentation(
                points,
                nodal_spacing * 8.f
            );

            auto combined_sdf2d_results
                = compute_combined_sdf2d(points, all_contours);

            for (size_t i = 0; i < all_contours.size(); ++i) {
                sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
                    const auto& edge_normals  = all_contours[i].edge_normals;
                    const auto& sdf2d_results = combined_sdf2d_results.results;
                    const auto& closest_contours
                        = combined_sdf2d_results.input_contour_idx;
                    const auto& edge_indices
                        = all_contours[i].contour_edges_in_input_mesh;
                    const auto& mesh_vertices
                        = all_contours[i].input_mesh.vertices;
                    const auto& mesh_edges
                        = all_contours[i].input_mesh.unique_edges;
                    ctx.launch(
                        sclx::md_range_t<1>{points.shape()[1]},
                        positions_velocities_and_distances,
                        [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                            //                            auto injection_idx
                            //                                = idx[0] %
                            //                                total_injections_per_site;
                            //                            if (injection_idx >=
                            //                            num_injected_per_site)
                            //                            {
                            //                                return;
                            //                            }
                            auto closest_contour_idx = closest_contours[idx[0]];
                            if (closest_contour_idx != i) {
                                return;
                            }
                            auto sdf2d_result     = sdf2d_results[idx[0]];
                            auto closest_edge_idx = sdf2d_result.closest_edge;
                            auto closest_edge_normal
                                = &edge_normals(0, closest_edge_idx);
                            //                            if
                            //                            (sdf2d_result.signed_distance
                            //                            < -nodal_spacing) {
                            //                                auto edge_x1 =
                            //                                &mesh_vertices(
                            //                                    0,
                            //                                    mesh_edges(
                            //                                        0,
                            //                                        edge_indices(closest_edge_idx)
                            //                                    )
                            //                                );
                            //
                            //                                naga::point_t<T,
                            //                                2> x1_to_point{
                            //                                    {points(0,
                            //                                    idx[0]) -
                            //                                    edge_x1[0],
                            //                                     points(1,
                            //                                     idx[0]) -
                            //                                     edge_x1[1]}
                            //                                };
                            //
                            //                                const auto
                            //                                rotated_y_axis =
                            //                                closest_edge_normal;
                            //                                const
                            //                                naga::point_t<T,
                            //                                2> rotated_x_axis{
                            //                                    {-rotated_y_axis[1],
                            //                                    rotated_y_axis[0]}
                            //                                };
                            //
                            //                                auto
                            //                                point_rotated_y =
                            //                                nodal_spacing;
                            //                                auto edge_length =
                            //                                naga::distance_functions::
                            //                                    loopless::euclidean<2>{}(
                            //                                        &mesh_vertices(
                            //                                            0,
                            //                                            mesh_edges(
                            //                                                0,
                            //                                                edge_indices(closest_edge_idx)
                            //                                            )
                            //                                        ),
                            //                                        &mesh_vertices(
                            //                                            0,
                            //                                            mesh_edges(
                            //                                                1,
                            //                                                edge_indices(closest_edge_idx)
                            //                                            )
                            //                                        )
                            //                                    );
                            //                                auto
                            //                                point_rotated_x =
                            //                                edge_length / 2.f;
                            //                                points(0, idx[0])
                            //                                    = edge_x1[0]
                            //                                    +
                            //                                    point_rotated_x
                            //                                    * rotated_x_axis[0]
                            //                                    +
                            //                                    point_rotated_y
                            //                                    * rotated_y_axis[0];
                            //                                points(1, idx[0])
                            //                                    = edge_x1[1]
                            //                                    +
                            //                                    point_rotated_x
                            //                                    * rotated_x_axis[1]
                            //                                    +
                            //                                    point_rotated_y
                            //                                    * rotated_y_axis[1];
                            //                                sdf2d_result.signed_distance
                            //                                = point_rotated_y;
                            //
                            //                                auto velocity_x =
                            //                                naga::math::loopless::dot<2>(
                            //                                    &node_velocities(0,
                            //                                    idx[0]),
                            //                                    rotated_x_axis
                            //                                );
                            //                                auto velocity_y =
                            //                                naga::math::loopless::dot<2>(
                            //                                    &node_velocities(0,
                            //                                    idx[0]),
                            //                                    closest_edge_normal
                            //                                );
                            //                                velocity_y =
                            //                                naga::math::abs(velocity_y);
                            //                                node_velocities(0,
                            //                                idx[0])
                            //                                    = velocity_x *
                            //                                    rotated_x_axis[0]
                            //                                    + velocity_y *
                            //                                    rotated_y_axis[0];
                            //                                node_velocities(1,
                            //                                idx[0])
                            //                                    = velocity_x *
                            //                                    rotated_x_axis[1]
                            //                                    + velocity_y *
                            //                                    rotated_y_axis[1];
                            //                                auto velocity =
                            //                                &node_velocities(0,
                            //                                idx[0]); auto
                            //                                velocity_mag
                            //                                    =
                            //                                    naga::math::loopless::norm<2>(velocity);
                            //                                if (velocity_mag
                            //                                < 1.f) {
                            //                                    return;
                            //                                }
                            //                                naga::math::loopless::normalize<2>(velocity);
                            //                            }
                            if (sdf2d_result.signed_distance > nodal_spacing) {
                                return;
                            }
                            auto q
                                = (sdf2d_result.signed_distance + nodal_spacing)
                                / nodal_spacing;
                            auto kernel_value
                                = kf
                                * pimesh_kernel_function<T>(
                                      pimesh_kernel_alpha<T, 2>::value(),
                                      q
                                );
                            auto force = &node_forces(0, idx[0]);
                            force[0] += kernel_value * closest_edge_normal[0];
                            force[1] += kernel_value * closest_edge_normal[1];
                        }
                    );
                }).get();
            }
        }

        naga::segmentation::nd_cubic_segmentation<T, 2> points_segmentation(
            points,
            nodal_spacing * 8.f
        );

        sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
            ctx.launch(
                sclx::md_range_t<1>{{node_forces.shape()[1]}},
                node_forces,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    //                    auto injection_idx = idx[0] %
                    //                    total_injections_per_site; if
                    //                    (injection_idx >=
                    //                    num_injected_per_site) {
                    //                        return;
                    //                    }
                    const auto& point_of_interest = &points(0, idx[0]);
                    const auto& partition_idx
                        = points_segmentation.get_partition_index(
                            point_of_interest
                        );
                    for (int i = -1; i < 2; ++i) {
                        for (int j = -1; j < 2; ++j) {
                            auto partition_idx_i
                                = static_cast<int>(partition_idx[0]) + i;
                            if (partition_idx_i < 0
                                || partition_idx_i
                                       >= points_segmentation.shape()[0]) {
                                continue;
                            }
                            auto partition_idx_j
                                = static_cast<int>(partition_idx[1]) + j;
                            if (partition_idx_j < 0
                                || partition_idx_j
                                       >= points_segmentation.shape()[1]) {
                                continue;
                            }
                            sclx::md_index_t<2> partition_idx_ij{
                                {static_cast<size_t>(partition_idx_i),
                                 static_cast<size_t>(partition_idx_j)}
                            };
                            const auto& partition
                                = points_segmentation.get_partition(
                                    partition_idx_ij
                                );
                            for (size_t n = 0; n < partition.size(); ++n) {
                                const auto& neighbor_idx
                                    = partition.indices()[n];
                                //                                if
                                //                                (neighbor_idx
                                //                                == idx[0]
                                //                                    ||
                                //                                    neighbor_idx
                                //                                    %
                                //                                    total_injections_per_site
                                //                                           >=
                                //                                           num_injected_per_site)
                                //                                           {
                                //                                    continue;
                                //                                }
                                if (neighbor_idx == idx[0]) {
                                    continue;
                                }
                                const auto& neighbor_point
                                    = &points(0, neighbor_idx);
                                auto distance = naga::distance_functions::
                                    loopless::euclidean<2>{
                                    }(point_of_interest, neighbor_point);
                                constexpr T colliding_nodes_tolerance = 1e-12;
                                if (distance < colliding_nodes_tolerance) {
                                    node_velocities(0, idx[0])
                                        = -node_velocities(0, neighbor_idx);
                                    node_velocities(1, idx[0])
                                        = -node_velocities(1, neighbor_idx);
                                    continue;
                                }
                                if (distance > nodal_spacing * 2.f) {
                                    continue;
                                }
                                auto q = distance / nodal_spacing;
                                auto kernel_value
                                    = kf
                                    * pimesh_kernel_function<T>(
                                          pimesh_kernel_alpha<T, 2>::value(),
                                          q
                                    );
                                naga::point_t<T, 2> xij{
                                    {neighbor_point[0] - point_of_interest[0],
                                     neighbor_point[1] - point_of_interest[1]}
                                };
                                naga::math::loopless::normalize<2>(xij);
                                node_forces(0, idx[0])
                                    += -kernel_value * xij[0];
                                node_forces(1, idx[0])
                                    += -kernel_value * xij[1];
                            }
                        }
                    }
                }
            );
        }).get();

        sclx::execute_kernel([&](sclx::kernel_handler& ctx) {
            ctx.launch(
                sclx::md_range_t<1>{{points.shape()[1]}},
                positions_velocities_and_distances,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    const auto& force    = &node_forces(0, idx[0]);
                    const auto& velocity = &node_velocities(0, idx[0]);
                    //                    force[0] -= drag_coefficient *
                    //                    velocity[0] / delta_t; force[1] -=
                    //                    drag_coefficient * velocity[1] /
                    //                    delta_t;
                    naga::point_t<T, 2> delta_x{
                        {delta_t * velocity[0]
                             + 0.5f * delta_t * delta_t * force[0],
                         delta_t * velocity[1]
                             + 0.5f * delta_t * delta_t * force[1]}
                    };
                    points(0, idx[0]) += delta_x[0];
                    points(1, idx[0]) += delta_x[1];
                    node_velocities(0, idx[0]) += delta_t * force[0];
                    node_velocities(1, idx[0]) += delta_t * force[1];
                    distance_travelled(idx[0])
                        = naga::math::loopless::norm<2>(delta_x);

                    auto velocity_mag = naga::math::loopless::norm<2>(velocity);
                    if (velocity_mag < 1.f) {
                        return;
                    }
                    naga::math::loopless::normalize<2>(velocity);
                }
            );
        }).get();

        auto average_distance_travelled
            = sclx::algorithm::reduce(distance_travelled, T{0}, std::plus<>{})
            / static_cast<T>(distance_travelled.elements());

        if (true) {
            auto average_forces = sclx::algorithm::reduce_last_dim(
                node_forces,
                0.f,
                sclx::algorithm::plus<>()
            );
            auto average_force = sclx::algorithm::reduce(
                                     average_forces,
                                     0.f,
                                     sclx::algorithm::plus<>()
                                 )
                               / static_cast<T>(average_forces.shape()[0]);
            T expected_error = 5.f;
            auto error = naga::math::abs(average_force - 0.) / nodal_spacing;

            std::cout << "\raverage force == 0 error (desired <= "
                      << expected_error
                      << "), average "
                         "node movement (desired <=: "
                      << .0002f * nodal_spacing << "): " << error << ", "
                      << average_distance_travelled << std::flush;

            if (error < expected_error
                && average_distance_travelled < .0002f * nodal_spacing) {
                break;
            }
        }

        if (average_distance_travelled > 0.1 * nodal_spacing) {
            drag_coefficient = 0.5f;
        } else if (average_distance_travelled > .02 * nodal_spacing) {
            drag_coefficient = 0.05f;
        } else {
            drag_coefficient
                = 0.05f
                * (current_time - naga::math::pow(current_time, .02) * .001f);
            drag_coefficient = std::min(drag_coefficient, 0.5f);
        }

        current_time += delta_t;

        if (current_time / delta_t > 1000) {
            break;
        }
    }

    return {points, points.shape()[1], 0, 0};
}

auto main() -> int {
    //    auto mesh =
    //    manifold_mesh_t<float>::import_from_obj(sclx::filesystem::path(
    //        "naga/resources/lbm_example_domains/rectangle/domain.obj"
    //    ));
    //
    //    // resave faces to domain_imported.obj
    //    std::ofstream obj_file(
    //        "naga/resources/lbm_example_domains/rectangle/domain_imported.obj"
    //    );
    //    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
    //        obj_file << "v " << mesh.vertices(0, i) << " " <<
    //        mesh.vertices(1, i)
    //                 << " " << mesh.vertices(2, i) << "\n";
    //    }
    //    for (size_t i = 0; i < mesh.triangle_vert_indices.shape()[1]; ++i)
    //    {
    //        obj_file << "f " << mesh.triangle_vert_indices(0, i) + 1 << "
    //        "
    //                 << mesh.triangle_vert_indices(1, i) + 1 << " "
    //                 << mesh.triangle_vert_indices(2, i) + 1 << "\n";
    //    }
    //    obj_file.close();
    //
    //    // resave edges to domain_imported_edges.obj
    //    std::ofstream obj_file_edges(
    //        "naga/resources/lbm_example_domains/rectangle/domain_imported_edges.obj"
    //    );
    //    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
    //        obj_file_edges << "v " << mesh.vertices(0, i) << " "
    //                       << mesh.vertices(1, i) << " " <<
    //                       mesh.vertices(2, i)
    //                       << "\n";
    //    }
    //    for (size_t i = 0; i < mesh.unique_edges.shape()[1]; ++i) {
    //        obj_file_edges << "l " << mesh.unique_edges(0, i) + 1 << " "
    //                       << mesh.unique_edges(1, i) + 1 << "\n";
    //    }
    //    obj_file_edges.close();
    //
    //    auto contour
    //        =
    //        closed_contour_t<float>::import_from_obj(sclx::filesystem::path(
    //            "naga/resources/lbm_example_domains/rectangle/domain.obj"
    //        ));
    //
    //    // resave edges to domain_imported_contour.obj
    //    std::ofstream
    //    obj_file_contour("naga/resources/lbm_example_domains/"
    //                                   "rectangle/domain_imported_contour.obj");
    //    for (size_t i = 0; i < contour.input_mesh.vertices.shape()[1];
    //    ++i) {
    //        obj_file_contour << "v " << mesh.vertices(0, i) << " "
    //                         << mesh.vertices(1, i) << " " <<
    //                         mesh.vertices(2, i)
    //                         << "\n";
    //    }
    //    for (size_t i = 0; i <
    //    contour.contour_edges_in_input_mesh.shape()[0];
    //         ++i) {
    //        obj_file_contour
    //            << "l "
    //            << mesh.unique_edges(0,
    //            contour.contour_edges_in_input_mesh(i)) + 1
    //            << " "
    //            << mesh.unique_edges(1,
    //            contour.contour_edges_in_input_mesh(i)) + 1
    //            << "\n";
    //    }
    //    obj_file_contour.close();
    //
    //    auto contour_edge_normals = contour.edge_normals;
    //    // save contour vertices w/ normasl to
    //    domain_imported_contour_normals.csv std::ofstream
    //    contour_normals_file(
    //        "naga/resources/lbm_example_domains/"
    //        "rectangle/domain_imported_contour_normals.csv"
    //    );
    //    contour_normals_file << "x,y,nx,ny\n";
    //    for (size_t e = 0; e <
    //    contour.contour_edges_in_input_mesh.elements();
    //         ++e) {
    //        auto edge_idx = contour.contour_edges_in_input_mesh(e);
    //        float center_point[2]{
    //            (contour.input_mesh
    //                 .vertices(0, contour.input_mesh.unique_edges(0,
    //                 edge_idx))
    //             + contour.input_mesh.vertices(
    //                 0,
    //                 contour.input_mesh.unique_edges(1, edge_idx)
    //             )) / 2,
    //            (contour.input_mesh
    //                 .vertices(1, contour.input_mesh.unique_edges(0,
    //                 edge_idx))
    //             + contour.input_mesh.vertices(
    //                 1,
    //                 contour.input_mesh.unique_edges(1, edge_idx)
    //             )) / 2
    //        };
    //        contour_normals_file << center_point[0] << "," <<
    //        center_point[1]
    //        << ","
    //                             << contour_edge_normals(0, e) << ","
    //                             << contour_edge_normals(1, e) << "\n";
    //    }
    //    contour_normals_file.close();
    //
    //    const auto& contour_lb = contour.input_mesh.lower_bound;
    //    const auto& contour_ub = contour.input_mesh.upper_bound;
    //    float node_separation = 0.2;
    //    auto num_nodes_x    = static_cast<size_t>(
    //        (contour_ub[0] - contour_lb[0]) / node_separation
    //    );
    //    auto num_nodes_y    = static_cast<size_t>(
    //        (contour_ub[1] - contour_lb[1]) / node_separation
    //    );
    //    sclx::array<float, 2> node_positions{2, num_nodes_x *
    //    num_nodes_y}; for (size_t i = 0; i < num_nodes_x; ++i) {
    //        for (size_t j = 0; j < num_nodes_y; ++j) {
    //            node_positions(0, i * num_nodes_y + j)
    //                = contour_lb[0] + i * node_separation;
    //            node_positions(1, i * num_nodes_y + j)
    //                = contour_lb[1] + j * node_separation;
    //        }
    //    }
    //    auto sdf_results = batched_sdf2d(node_positions, contour);
    //
    //    // save node positions and sdf results to sdf2d_results.csv
    //    std::ofstream
    //    sdf2d_results_file("naga/resources/lbm_example_domains/"
    //                                     "rectangle/sdf2d_results.csv");
    //    sdf2d_results_file << "x,y,sdf\n";
    //    for (size_t i = 0; i < node_positions.shape()[1]; ++i) {
    //        sdf2d_results_file << node_positions(0, i) << ","
    //                           << node_positions(1, i) << ","
    //                           << sdf_results(i).signed_distance << "\n";
    //    }
    //    sdf2d_results_file.close();
    //
    //    // load complex 3D mesh to test speed of mesh import
    //    auto complex_mesh =
    //    manifold_mesh_t<float>::import_from_obj(sclx::filesystem::path(
    //        "resources/simulation_domains/3d_cathedral/cathedral_simplified.obj"
    //    ));
    //    // rewrite it to cathedral_simplified_imported.obj to test for
    //    correctness std::ofstream complex_mesh_file(
    //        "resources/simulation_domains/3d_cathedral/cathedral_simplified_imported.obj"
    //    );
    //    for (size_t i = 0; i < complex_mesh.vertices.shape()[1]; ++i) {
    //        complex_mesh_file << "v " << complex_mesh.vertices(0, i) << "
    //        "
    //                          << complex_mesh.vertices(1, i) << " "
    //                          << complex_mesh.vertices(2, i) << "\n";
    //    }
    //    for (size_t i = 0; i <
    //    complex_mesh.triangle_vert_indices.shape()[1];
    //         ++i) {
    //        complex_mesh_file << "f "
    //                          << complex_mesh.triangle_vert_indices(0, i)
    //                          + 1
    //                          << " "
    //                          << complex_mesh.triangle_vert_indices(1, i)
    //                          + 1
    //                          << " "
    //                          << complex_mesh.triangle_vert_indices(2, i)
    //                          + 1
    //                          << "\n";
    //    }

    auto domain_mesh_path = sclx::filesystem::path(
        "resources/simulation_domains/3d_cathedral/cathedral_simplified.obj"
    );
    auto mesh = manifold_mesh_t<float>::import_from_obj(domain_mesh_path);

    // let's compute the vertex normals by averaging the normals of the
    // surrounding faces
    std::vector<naga::point_t<float, 3>> vertex_normals;
    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
        auto vertex_normal = naga::point_t<float, 3>{{0.f, 0.f, 0.f}};
        for (size_t j = 0; j < mesh.vertex_face_neighbors.shape()[0]; ++j) {
            if (mesh.vertex_face_neighbors(j, i)
                == manifold_mesh_t<float>::no_face) {
                continue;
            }
            auto normal_idx
                = mesh.triangle_normal_indices(mesh.vertex_face_neighbors(j, i));

            vertex_normal[0] += mesh.normals(0, normal_idx);
            vertex_normal[1] += mesh.normals(1, normal_idx);
            vertex_normal[2] += mesh.normals(2, normal_idx);
        }
        auto norm = naga::math::loopless::norm<3>(vertex_normal);
        if (norm > 1e-6) {
            vertex_normal[0] /= norm;
            vertex_normal[1] /= norm;
            vertex_normal[2] /= norm;
        } else {
//            std::stringstream error_ss;
//            error_ss << "vertex " << i << " has zero normal. The normals of "
//                     << "the surrounding faces are:\n";
//            for (size_t j = 0; j < mesh.vertex_face_neighbors.shape()[0]; ++j) {
//                if (mesh.vertex_face_neighbors(j, i)
//                    == manifold_mesh_t<float>::no_face) {
//                    std::cout << "    no face\n";
//                    continue;
//                }
//                auto normal_idx = mesh.triangle_normal_indices(
//                    mesh.vertex_face_neighbors(j, i)
//                );
//                std::cout << "    " << mesh.normals(0, normal_idx) << " "
//                          << mesh.normals(1, normal_idx) << " "
//                          << mesh.normals(2, normal_idx) << "\n";
//            }
//
//            throw std::runtime_error(error_ss.str());
        }
        vertex_normals.emplace_back(vertex_normal);
    }

    std::ofstream vertex_normals_file(
        "resources/simulation_domains/3d_cathedral/"
        "cathedral_simplified_vertex_normals.csv"
    );
    vertex_normals_file << "x,y,z,nx,ny,nz\n";
    for (size_t i = 0; i < mesh.vertices.shape()[1]; ++i) {
        vertex_normals_file
            << mesh.vertices(0, i) << "," << mesh.vertices(1, i) << ","
            << mesh.vertices(2, i) << "," << vertex_normals[i][0] << ","
            << vertex_normals[i][1] << "," << vertex_normals[i][2] << "\n";
    }
    vertex_normals_file.close();

    //    // test injection points generation
    //    auto domain_mesh_path = sclx::filesystem::path(
    //        "resources/simulation_domains/2d_head_in_square_room/room.obj"
    //    );
    //    auto domain_contour = closed_contour_t<float>::import_from_obj(
    //        domain_mesh_path,
    //        /*flip_normals=*/true
    //    );
    //    auto immersed_mesh_paths = std::vector{sclx::filesystem::path(
    //        "resources/simulation_domains/2d_head_in_square_room/head.obj"
    //    )};
    //    std::vector<closed_contour_t<float>> immersed_contours;
    //    for (const auto& immersed_mesh_path : immersed_mesh_paths) {
    //        immersed_contours.emplace_back(
    //            closed_contour_t<float>::import_from_obj(immersed_mesh_path)
    //        );
    //    }
    //    float nodal_spacing  = 0.1f;
    //    auto injection_sites = create_pimesh2d_injection_sites(
    //        nodal_spacing,
    //        domain_contour,
    //        immersed_contours
    //    );
    //    std::ofstream injection_sites_file(
    //        "resources/simulation_domains/2d_head_in_square_room/"
    //        "injection_sites.csv"
    //    );
    //    injection_sites_file << "x,y\n";
    //    for (size_t i = 0; i < injection_sites.shape()[1]; ++i) {
    //        injection_sites_file << injection_sites(0, i) << ","
    //                             << injection_sites(1, i) << "\n";
    //    }
    //    injection_sites_file.close();
    //
    //    auto domain = pimesh2d_generate_domain(
    //        nodal_spacing,
    //        domain_contour,
    //        immersed_contours
    //    );
    //    std::ofstream domain_file(
    //        "resources/simulation_domains/2d_head_in_square_room/domain.csv"
    //    );
    //    domain_file << "x,y\n";
    //    for (size_t i = 0; i < domain.points.shape()[1]; ++i) {
    //        domain_file << domain.points(0, i) << "," << domain.points(1, i)
    //                    << "\n";
    //    }
    //    domain_file.close();

    return 0;
}