
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

#pragma once

/* This file is a mess... It's copied over from the original implementation
 * of NAGA as it's "good enough" and it's only used once at the beginning
 * of simulations. It's not worth the effort to clean it up.
 */

#include "../../../detail/nanoflann_utils.h"
#include "../../../distance_functions.cuh"
#include "../../../math.cuh"
#include "../../../mesh/closed_surface.cuh"
#include <scalix/algorithm/transform.cuh>
#include <sdf/sdf.hpp>

namespace naga::fluids::nonlocal_lbm::detail {

template<class T>
using nanoflann_cloud_t = ::naga::detail::PointCloud<T>;

template<class T>
using kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<T, nanoflann_cloud_t<T>>,
    nanoflann_cloud_t<T>,
    3 /* dim */
    >;

template<class T>
struct pps_temporary_layer_3d_t {
    std::vector<T> points;
    std::vector<T> field_lines;
    mesh::closed_surface_t<T> source_surface;
    uint node_layer_count;
    uint absorption_layer_count;
    T absorption_coefficient;
};

template<class T>
std::vector<T>
calculate_field(const T* p, const mesh::closed_surface_t<T>& source_surface) {
    std::vector<T> field = {0, 0, 0};
    for (size_t i = 0; i < source_surface.vertices.shape()[1]; i++) {
        T diff[3]
            = {source_surface.vertices(0, i) - p[0],
               source_surface.vertices(1, i) - p[1],
               source_surface.vertices(2, i) - p[2]};
        T distance = math::loopless::norm<3>(diff);
        if (distance == 0) {
            continue;
        }
        field[0] -= diff[0] / math::pow(distance, 3.f / 2.f);
        field[1] -= diff[1] / math::pow(distance, 3.f / 2.f);
        field[2] -= diff[2] / math::pow(distance, 3.f / 2.f);
    }
    return field;
}

template<class T>
void populate_layer_field_lines(pps_temporary_layer_3d_t<T>& layer) {
    layer.field_lines.clear();
    for (size_t i = 0; i < layer.points.size(); i += 3) {
        std::vector<T> field
            = calculate_field<T>(&layer.points[i], layer.source_surface);
        layer.field_lines.push_back(field[0]);
        layer.field_lines.push_back(field[1]);
        layer.field_lines.push_back(field[2]);
    }
}

template<class T>
pps_temporary_layer_3d_t<T> init_pps_layer(
    const mesh::closed_surface_t<T>& source_surface,
    const uint& node_layer_count,
    const uint& absorption_layer_count,
    const T& absorption_coefficient
) {
    pps_temporary_layer_3d_t<T> layer;
    layer.source_surface         = source_surface;
    layer.node_layer_count       = node_layer_count;
    layer.absorption_layer_count = absorption_layer_count;
    layer.absorption_coefficient = absorption_coefficient;
    for (size_t i = 0; i < source_surface.vertices.shape()[1]; i++) {
        T new_point[3];
        new_point[0] = source_surface.vertices(0, i);
        new_point[1] = source_surface.vertices(1, i);
        new_point[2] = source_surface.vertices(2, i);
        layer.points.push_back(new_point[0]);
        layer.points.push_back(new_point[1]);
        layer.points.push_back(new_point[2]);
    }
    populate_layer_field_lines(layer);
    return layer;
}

template<class T>
void add_layer(pps_temporary_layer_3d_t<T>& layer, T layer_thickness) {
    T average_field_mag = 0;

    for (size_t i = 0; i < layer.field_lines.size(); i += 3) {
        average_field_mag += math::loopless::norm<3>(&layer.field_lines[i]);
    }
    average_field_mag /= layer.field_lines.size() / 3;
    T step_size = layer_thickness / average_field_mag;

    for (size_t i = 0; i < layer.points.size(); i += 3) {
        layer.points[i] += step_size * layer.field_lines[i];
        layer.points[i + 1] += step_size * layer.field_lines[i + 1];
        layer.points[i + 2] += step_size * layer.field_lines[i + 2];
    }

    populate_layer_field_lines(layer);
}

template<class T>
T get_smallest_edge_length(const mesh::closed_surface_t<T>& surface) {
    T min_len = std::numeric_limits<T>::max();
    for (size_t f = 0; f < surface.faces.shape()[1]; f++) {
        for (uint e = 0; e < 3; e++) {
            T* v0 = &surface.vertices(0, surface.faces(e, f));
            T* v1 = &surface.vertices(0, surface.faces((e + 1) % 3, f));
            T len = distance_functions::loopless::euclidean<3>{}(v0, v1);
            if (len < min_len) {
                min_len = len;
            }
        }
    }
    return min_len;
}

template<class T>
T get_min_edge_length_of_all(
    const pps_temporary_layer_3d_t<T>& outer_boundary_layer,
    const std::vector<pps_temporary_layer_3d_t<T>>& inner_boundary_layers
) {
    T min_edge_length = std::numeric_limits<T>::max();

    // outer boundary
    for (size_t f = 0; f < outer_boundary_layer.source_surface.faces.shape()[1];
         f++) {
        for (size_t v = 0; v < 3; v++) {
            size_t v1 = outer_boundary_layer.source_surface.faces(v, f);
            size_t v2
                = outer_boundary_layer.source_surface.faces((v + 1) % 3, f);
            T v12[3]
                = {outer_boundary_layer.points[v1 * 3 + 0]
                       - outer_boundary_layer.points[v2 * 3 + 0],
                   outer_boundary_layer.points[v1 * 3 + 1]
                       - outer_boundary_layer.points[v2 * 3 + 1],
                   outer_boundary_layer.points[v1 * 3 + 2]
                       - outer_boundary_layer.points[v2 * 3 + 2]};
            auto edge_len = math::loopless::norm<3>(v12);
            if (edge_len < min_edge_length) {
                min_edge_length = edge_len;
            }
        }
    }

    // inner boundaries
    for (const pps_temporary_layer_3d_t<T>& layer : inner_boundary_layers) {
        for (size_t f = 0; f < layer.source_surface.faces.shape()[1]; f++) {
            for (size_t v = 0; v < 3; v++) {
                size_t v1 = layer.source_surface.faces(v, f);
                size_t v2 = layer.source_surface.faces((v + 1) % 3, f);
                T v12[3]
                    = {layer.points[v1 * 3 + 0] - layer.points[v2 * 3 + 0],
                       layer.points[v1 * 3 + 1] - layer.points[v2 * 3 + 1],
                       layer.points[v1 * 3 + 2] - layer.points[v2 * 3 + 2]};
                auto edge_len = math::loopless::norm<3>(v12);
                if (edge_len < min_edge_length) {
                    min_edge_length = edge_len;
                }
            }
        }
    }

    return min_edge_length;
}

struct naga_sdf {
    std::unique_ptr<sdf::SDF> sdf;
    std::unique_ptr<sdf::Points> points;
    std::unique_ptr<sdf::Triangles> faces;
};

template<class T>
naga_sdf
build_sdf(const std::vector<T>& points, const std::vector<size_t>& faces) {
    std::unique_ptr<sdf::Points> sdf_points
        = std::make_unique<sdf::Points>(points.size() / 3, 3);
    for (u_int32_t p = 0; p < points.size() / 3; p++) {
        sdf_points.get()[0](p, 0) = points[p * 3];
        sdf_points.get()[0](p, 1) = points[p * 3 + 1];
        sdf_points.get()[0](p, 2) = points[p * 3 + 2];
    }
    std::unique_ptr<sdf::Triangles> sdf_faces
        = std::make_unique<sdf::Triangles>(faces.size() / 3, 3);
    for (u_int32_t f = 0; f < faces.size() / 3; f++) {
        sdf_faces.get()[0](f, 0) = faces[f * 3];
        sdf_faces.get()[0](f, 1) = faces[f * 3 + 1];
        sdf_faces.get()[0](f, 2) = faces[f * 3 + 2];
    }
    return naga_sdf{
        std::make_unique<sdf::SDF>(*sdf_points, *sdf_faces),
        std::move(sdf_points),
        std::move(sdf_faces)};
}

template<class T>
std::vector<T>
get_sdf_to_points(const std::vector<T>& points, const sdf::SDF& surface) {
    sdf::Points sdf_points(points.size() / 3, 3);
    for (uint p = 0; p < points.size() / 3; p++) {
        sdf_points(p, 0) = points[p * 3];
        sdf_points(p, 1) = points[p * 3 + 1];
        sdf_points(p, 2) = points[p * 3 + 2];
    }
    auto sdfs = surface(sdf_points);
    return std::vector<T>(sdfs.data(), sdfs.data() + sdfs.size());
}

template<class T>
bool test_point_with_outer_layer(
    const T* query_pt,
    nanoflann_cloud_t<T>& outer_boundary_vertices,
    const kd_tree_t<T>& outer_boundary_kd_tree,
    const T& layer_thickness
) {

    if (!outer_boundary_vertices.pts.empty()) {
        int num_results = 1;
        std::vector<size_t> ret_index(num_results);
        std::vector<T> out_dist_sqr(num_results);
        nanoflann::KNNResultSet<T> resultSet(num_results);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        outer_boundary_kd_tree
            .findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

        if (out_dist_sqr[0] < .9f * .9f * layer_thickness * layer_thickness
            && ret_index[0] != outer_boundary_vertices.pts.size()) {
            return false;
        }
    }

    return true;
}

template<class T>
__host__ void advance_front(
    uint layer_idx,
    nanoflann_cloud_t<T>& outer_boundary_vertices,
    kd_tree_t<T>& outer_boundary_kd_tree,
    std::vector<T>& bulk_points,
    std::vector<T>& layer_points,
    std::vector<T>& layer_boundary_absorption,
    pps_temporary_layer_3d_t<T>& outer_boundary_layer,
    std::vector<pps_temporary_layer_3d_t<T>>& inner_boundary_layers,
    T layer_thickness
) {
    std::vector<T> potential_points;
    std::vector<bool> potential_points_valid;
    auto outer_sdf = build_sdf(
        std::vector<T>(
            outer_boundary_layer.source_surface.vertices.begin(),
            outer_boundary_layer.source_surface.vertices.end()
        ),
        std::vector<size_t>(
            outer_boundary_layer.source_surface.faces.begin(),
            outer_boundary_layer.source_surface.faces.end()
        )
    );
    std::vector<naga_sdf> inner_sdfs;
    for (auto& inner_boundary_layer : inner_boundary_layers) {
        auto layer_sdf = build_sdf(
            inner_boundary_layer.points,
            std::vector<size_t>(
                inner_boundary_layer.source_surface.faces.begin(),
                inner_boundary_layer.source_surface.faces.end()
            )
        );
        inner_sdfs.emplace_back(std::move(layer_sdf));
    }

    // advance outer boundary
    for (size_t i = 0;
         i < outer_boundary_layer.source_surface.vertices.shape()[1];
         i++) {
        if (layer_idx >= outer_boundary_layer.node_layer_count) {
            break;
        }
        std::vector<T> field{
            outer_boundary_layer.source_surface.vertex_normals(0, i),
            outer_boundary_layer.source_surface.vertex_normals(1, i),
            outer_boundary_layer.source_surface.vertex_normals(2, i)};
        T query_pt[3] = {
            outer_boundary_layer.points[3 * i + 0] + field[0] * layer_thickness,
            outer_boundary_layer.points[3 * i + 1] + field[1] * layer_thickness,
            outer_boundary_layer.points[3 * i + 2]
                + field[2] * layer_thickness};

        bool is_too_close_to_outer_layer = !test_point_with_outer_layer(
            query_pt,
            outer_boundary_vertices,
            outer_boundary_kd_tree,
            layer_thickness
        );

        if (!is_too_close_to_outer_layer) {
            potential_points.push_back(query_pt[0]);
            potential_points.push_back(query_pt[1]);
            potential_points.push_back(query_pt[2]);
            potential_points_valid.push_back(true);
        }

        outer_boundary_layer.points[3 * i + 0] = query_pt[0];
        outer_boundary_layer.points[3 * i + 1] = query_pt[1];
        outer_boundary_layer.points[3 * i + 2] = query_pt[2];
    }
    std::vector<T> sdf_values
        = get_sdf_to_points(potential_points, outer_sdf.sdf.get()[0]);
    for (size_t i = 0; i < sdf_values.size(); i++) {
        if (sdf_values[i] < .76f * layer_thickness) {
            potential_points_valid[i] = false;
        }
    }
    for (const auto& sdf : inner_sdfs) {
        sdf_values = get_sdf_to_points(potential_points, sdf.sdf.get()[0]);
        for (size_t i = 0; i < sdf_values.size(); i++) {
            if (-sdf_values[i] < .76f * layer_thickness) {
                potential_points_valid[i] = false;
            }
        }
    }
    for (size_t i = 0; i < potential_points.size() / 3; i++) {
        if (potential_points_valid[i]) {
            outer_boundary_vertices.pts.push_back(
                {potential_points[3 * i + 0],
                 potential_points[3 * i + 1],
                 potential_points[3 * i + 2]}
            );
            outer_boundary_kd_tree.addPoints(
                outer_boundary_vertices.pts.size() - 1,
                outer_boundary_vertices.pts.size()
            );
            if (layer_idx < outer_boundary_layer.absorption_layer_count) {
                layer_points.push_back(potential_points[3 * i + 0]);
                layer_points.push_back(potential_points[3 * i + 1]);
                layer_points.push_back(potential_points[3 * i + 2]);
                layer_boundary_absorption.push_back(
                    outer_boundary_layer.absorption_coefficient
                    * std::pow<T>(
                        (outer_boundary_layer.absorption_layer_count - 1
                         - layer_idx)
                            / ( T ) (outer_boundary_layer.absorption_layer_count
                                     - 1),
                        2
                    )
                );
            } else {
                bulk_points.push_back(potential_points[3 * i + 0]);
                bulk_points.push_back(potential_points[3 * i + 1]);
                bulk_points.push_back(potential_points[3 * i + 2]);
            }
        }
    }

    // advance inner boundaries
    for (pps_temporary_layer_3d_t<T>& layer : inner_boundary_layers) {
        add_layer(layer, layer_thickness);
        potential_points.clear();
        potential_points_valid.clear();
        for (size_t i = 0; i < layer.source_surface.vertices.shape()[1]; i++) {
            if (layer_idx >= layer.node_layer_count) {
                break;
            }
            T query_pt[3]
                = {layer.points[i * 3 + 0],
                   layer.points[i * 3 + 1],
                   layer.points[i * 3 + 2]};

            bool is_too_close_to_outer_layer = !test_point_with_outer_layer(
                query_pt,
                outer_boundary_vertices,
                outer_boundary_kd_tree,
                layer_thickness
            );

            if (!is_too_close_to_outer_layer) {
                potential_points.push_back(query_pt[0]);
                potential_points.push_back(query_pt[1]);
                potential_points.push_back(query_pt[2]);
                potential_points_valid.push_back(true);
            }
        }
        sdf_values
            = get_sdf_to_points(potential_points, outer_sdf.sdf.get()[0]);
        for (size_t i = 0; i < sdf_values.size(); i++) {
            if (sdf_values[i] < .76f * layer_thickness) {
                potential_points_valid[i] = false;
            }
        }
        for (size_t l = 0; l < inner_boundary_layers.size(); l++) {
            if (&layer == &inner_boundary_layers[l]) {
                continue;
            }
            sdf_values = get_sdf_to_points(
                potential_points,
                inner_sdfs[l].sdf.get()[0]
            );
            for (size_t i = 0; i < sdf_values.size(); i++) {
                if (-sdf_values[i] < .76f * layer_thickness) {
                    potential_points_valid[i] = false;
                }
            }
        }
        for (size_t i = 0; i < potential_points.size() / 3; i++) {
            if (potential_points_valid[i]) {
                if (layer_idx < layer.node_layer_count) {
                    layer_points.push_back(potential_points[3 * i + 0]);
                    layer_points.push_back(potential_points[3 * i + 1]);
                    layer_points.push_back(potential_points[3 * i + 2]);
                    layer_boundary_absorption.push_back(
                        layer.absorption_coefficient
                        * std::pow<T>(
                            (layer.node_layer_count - 1 - layer_idx)
                                / ( T ) (layer.node_layer_count - 1),
                            2
                        )
                    );
                } else {
                    bulk_points.push_back(potential_points[3 * i + 0]);
                    bulk_points.push_back(potential_points[3 * i + 1]);
                    bulk_points.push_back(potential_points[3 * i + 2]);
                }
            }
        }
    }
}

template<class T>
std::vector<std::vector<T>>
get_surface_bounds(const mesh::closed_surface_t<T>& surface) {
    std::vector<std::vector<T>> bounds(2, std::vector<T>(3, 0));
    for (size_t i = 0; i < 3; i++) {
        bounds[0][i] = std::numeric_limits<T>::max();
        bounds[1][i] = std::numeric_limits<T>::lowest();
    }

    for (size_t i = 0; i < surface.vertices.shape()[1]; i++) {
        for (size_t j = 0; j < 3; j++) {
            bounds[0][j] = std::min(bounds[0][j], surface.vertices(j, i));
            bounds[1][j] = std::max(bounds[1][j], surface.vertices(j, i));
        }
    }

    return bounds;
}

template<class T>
simulation_domain<T> hycaps3d_load_domain(
    const boundary_specification<T>& outer_boundary,
    const std::vector<boundary_specification<T>>& inner_boundaries
) {
    std::vector<T> bulk_points;
    std::vector<T> layer_points;
    std::vector<T> layer_absorption;
    std::vector<T> boundary_points;
    std::vector<T> boundary_normals;

    /********************** outer boundary ************************/
    auto domain_surface
        = mesh::closed_surface_t<T>::import(outer_boundary.obj_file_path);
    // flip normals to point inwards
    sclx::algorithm::transform(
        domain_surface.vertex_normals,
        domain_surface.vertex_normals,
        T{-1},
        sclx::algorithm::multiplies<>()
    );
    T min_outer_edge_len      = get_smallest_edge_length(domain_surface);
    auto outer_boundary_layer = init_pps_layer(
        domain_surface,
        outer_boundary.node_layer_count,
        outer_boundary.absorption_layer_count,
        outer_boundary.absorption_coefficient
    );
    for (size_t i = 0; i < domain_surface.vertices.shape()[1]; i++) {
        boundary_points.push_back(domain_surface.vertices(0, i));
        boundary_points.push_back(domain_surface.vertices(1, i));
        boundary_points.push_back(domain_surface.vertices(2, i));
        boundary_normals.push_back(domain_surface.vertex_normals(0, i));
        boundary_normals.push_back(domain_surface.vertex_normals(1, i));
        boundary_normals.push_back(domain_surface.vertex_normals(2, i));
    }

    /********************** inner boundaries ************************/
    std::vector<pps_temporary_layer_3d_t<T>> inner_boundary_layers;
    for (const boundary_specification<T>& boundary : inner_boundaries) {
        auto inner_surface
            = mesh::closed_surface_t<T>::import(boundary.obj_file_path);
        inner_boundary_layers.push_back(init_pps_layer(
            inner_surface,
            boundary.node_layer_count,
            boundary.absorption_layer_count,
            boundary.absorption_coefficient
        ));
        for (size_t i = 0; i < inner_surface.vertices.shape()[1]; i++) {
            boundary_points.push_back(inner_surface.vertices(0, i));
            boundary_points.push_back(inner_surface.vertices(1, i));
            boundary_points.push_back(inner_surface.vertices(2, i));
            boundary_normals.push_back(inner_surface.vertex_normals(0, i));
            boundary_normals.push_back(inner_surface.vertex_normals(1, i));
            boundary_normals.push_back(inner_surface.vertex_normals(2, i));
        }
    }

    /********************** bulk ************************/
    T nodal_spacing = get_min_edge_length_of_all(
        outer_boundary_layer,
        inner_boundary_layers
    );
    nanoflann_cloud_t<T> outer_boundary_cloud;
    kd_tree_t<T> outer_boundary_tree(
        3,
        outer_boundary_cloud,
        nanoflann::KDTreeSingleIndexAdaptorParams(10)
    );

    uint max_layer_count = outer_boundary.node_layer_count;
    for (const auto& layer : inner_boundary_layers) {
        max_layer_count = std::max(max_layer_count, layer.node_layer_count);
    }
    for (size_t i = 0; i < max_layer_count; i++) {
        advance_front(
            i,
            outer_boundary_cloud,
            outer_boundary_tree,
            bulk_points,
            layer_points,
            layer_absorption,
            outer_boundary_layer,
            inner_boundary_layers,
            nodal_spacing
        );
    }

    auto domain_bounds = get_surface_bounds(domain_surface);
    std::vector<size_t> num_nodes_per_dim{
        static_cast<size_t>(std::floor(
            (domain_bounds[1][0] - domain_bounds[0][0]) / min_outer_edge_len
        )),
        static_cast<size_t>(std::floor(
            (domain_bounds[1][1] - domain_bounds[0][1]) / min_outer_edge_len
        )),
        static_cast<size_t>(std::floor(
            (domain_bounds[1][2] - domain_bounds[0][2]) / min_outer_edge_len
        ))};
    std::vector<T> spacing_per_dim{
        (domain_bounds[1][0] - domain_bounds[0][0]) / num_nodes_per_dim[0],
        (domain_bounds[1][1] - domain_bounds[0][1]) / num_nodes_per_dim[1],
        (domain_bounds[1][2] - domain_bounds[0][2]) / num_nodes_per_dim[2]};
    size_t total_num_nodes
        = num_nodes_per_dim[0] * num_nodes_per_dim[1] * num_nodes_per_dim[2];
    bulk_points.reserve(bulk_points.size() + total_num_nodes * 3);

    std::vector<T> potential_points;
    std::vector<bool> potential_points_valid;
    auto outer_sdf = build_sdf(
        std::vector<T>(
            outer_boundary_layer.source_surface.vertices.begin(),
            outer_boundary_layer.source_surface.vertices.end()
        ),
        std::vector<size_t>(
            outer_boundary_layer.source_surface.faces.begin(),
            outer_boundary_layer.source_surface.faces.end()
        )
    );
    std::vector<naga_sdf> inner_sdfs;
    for (auto& inner_boundary_layer : inner_boundary_layers) {
        auto layer_sdf = build_sdf(
            inner_boundary_layer.points,
            std::vector<size_t>(
                inner_boundary_layer.source_surface.faces.begin(),
                inner_boundary_layer.source_surface.faces.end()
            )
        );
        inner_sdfs.emplace_back(std::move(layer_sdf));
    }

    for (size_t i = 0; i < num_nodes_per_dim[0]; i++) {
        for (size_t j = 0; j < num_nodes_per_dim[1]; j++) {
            for (size_t k = 0; k < num_nodes_per_dim[2]; k++) {
                T query_point[3]
                    = {domain_bounds[0][0] + i * spacing_per_dim[0],
                       domain_bounds[0][1] + j * spacing_per_dim[1],
                       domain_bounds[0][2] + k * spacing_per_dim[2]};

                bool is_too_close_to_outer_layer = !test_point_with_outer_layer(
                    query_point,
                    outer_boundary_cloud,
                    outer_boundary_tree,
                    nodal_spacing
                );

                if (!is_too_close_to_outer_layer) {
                    potential_points.push_back(query_point[0]);
                    potential_points.push_back(query_point[1]);
                    potential_points.push_back(query_point[2]);
                    potential_points_valid.push_back(true);
                }
            }
        }
    }
    std::vector<T> sdf_values
        = get_sdf_to_points(potential_points, outer_sdf.sdf.get()[0]);
    for (size_t i = 0; i < sdf_values.size(); i++) {
        if (sdf_values[i] < .76f * nodal_spacing) {
            potential_points_valid[i] = false;
        }
    }
    for (const auto& sdf : inner_sdfs) {
        sdf_values = get_sdf_to_points(potential_points, sdf.sdf.get()[0]);
        for (size_t i = 0; i < sdf_values.size(); i++) {
            if (-sdf_values[i] < .76f * nodal_spacing) {
                potential_points_valid[i] = false;
            }
        }
    }
    for (size_t i = 0; i < potential_points.size() / 3; i++) {
        if (potential_points_valid[i]) {
            bulk_points.push_back(potential_points[3 * i + 0]);
            bulk_points.push_back(potential_points[3 * i + 1]);
            bulk_points.push_back(potential_points[3 * i + 2]);
        }
    }

    size_t bulk_size     = bulk_points.size() / 3;
    size_t layer_size    = layer_points.size() / 3;
    size_t boundary_size = boundary_points.size() / 3;

    simulation_domain<T> domain;

    domain.points
        = sclx::array<T, 2>{3, bulk_size + layer_size + boundary_size};
    std::copy(
        bulk_points.begin(),
        bulk_points.end(),
        domain.points.data().get()
    );
    std::copy(
        layer_points.begin(),
        layer_points.end(),
        domain.points.data().get() + 3 * bulk_size
    );
    std::copy(
        boundary_points.begin(),
        boundary_points.end(),
        domain.points.data().get() + 3 * (bulk_size + layer_size)
    );

    domain.boundary_normals = sclx::array<T, 2>{3, boundary_size};
    std::copy(
        boundary_normals.begin(),
        boundary_normals.end(),
        domain.boundary_normals.data().get()
    );

    domain.layer_absorption = sclx::array<T, 1>{layer_size};
    std::copy(
        layer_absorption.begin(),
        layer_absorption.end(),
        domain.layer_absorption.data().get()
    );

    domain.num_bulk_points = bulk_size;
    domain.num_layer_points = layer_size;
    domain.num_boundary_points = boundary_size;
    domain.nodal_spacing = nodal_spacing;

    return domain;
}

template<class T>
using f_type = decltype(hycaps3d_load_domain<T>);

template f_type<float> hycaps3d_load_domain;

}  // namespace naga::fluids::nonlocal_lbm::detail
