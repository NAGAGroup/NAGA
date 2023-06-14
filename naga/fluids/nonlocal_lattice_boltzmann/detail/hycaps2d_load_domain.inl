
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
#include "../../../mesh/closed_contour.cuh"
#include <omp.h>
#include <scalix/algorithm/transform.cuh>

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
using closed_contour_t = ::naga::mesh::closed_contour_t<T>;

template<class T>
struct pps_temporary_layer_2d_t {
    std::vector<T> points;
    std::vector<T> field_lines;
    closed_contour_t<T> source_contour;
    uint node_layer_count;
    uint absorption_layer_count;
    T absorption_coefficient;
};

template<class T>
std::vector<T>
calculate_field(const T* p, const closed_contour_t<T>& source_contour) {
    std::vector<T> field = {0, 0};
    for (size_t i = 0; i < source_contour.vertices.shape()[1]; i++) {
        T diff[2]
            = {source_contour.vertices(0, i) - p[0],
               source_contour.vertices(1, i) - p[1]};
        T distance = math::loopless::norm<2>(diff);
        if (distance == 0) {
            continue;
        }
        field[0] -= diff[0] / std::pow(distance, 3.f / 2.f);
        field[1] -= diff[1] / std::pow(distance, 3.f / 2.f);
    }

    return field;
}

template<class T>
void populate_layer_field_lines(pps_temporary_layer_2d_t<T>& layer) {
    layer.field_lines.clear();
    for (size_t i = 0; i < layer.points.size(); i += 2) {
        std::vector<T> field
            = calculate_field(&layer.points[i], layer.source_contour);
        layer.field_lines.push_back(field[0]);
        layer.field_lines.push_back(field[1]);
    }
}

template<class T>
void add_layer(pps_temporary_layer_2d_t<T>& layer, T layer_thickness) {
    T average_field_mag = 0;

    for (size_t i = 0; i < layer.field_lines.size(); i += 2) {
        average_field_mag += math::loopless::norm<2>(&layer.field_lines[i]);
    }
    average_field_mag /= layer.field_lines.size() / 2;
    T step_size = layer_thickness / average_field_mag;

    for (size_t i = 0; i < layer.points.size(); i += 2) {
        layer.points[i] += step_size * layer.field_lines[i];
        layer.points[i + 1] += step_size * layer.field_lines[i + 1];
    }

    populate_layer_field_lines(layer);
}

template<class T>
pps_temporary_layer_2d_t<T> init_pps_layer(
    const closed_contour_t<T>& source_contour,
    const uint& node_layer_count,
    const uint& absorption_layer_count,
    const T& absorption_coefficient
) {
    pps_temporary_layer_2d_t<T> layer;
    layer.source_contour         = source_contour;
    layer.node_layer_count       = node_layer_count;
    layer.absorption_layer_count = absorption_layer_count;
    layer.absorption_coefficient = absorption_coefficient;
    for (size_t i = 0; i < source_contour.vertices.shape()[1]; i++) {
        T new_point[2];
        new_point[0] = source_contour.vertices(0, i);
        new_point[1] = source_contour.vertices(1, i);
        layer.points.push_back(new_point[0]);
        layer.points.push_back(new_point[1]);
    }
    populate_layer_field_lines(layer);
    return layer;
}

template<class T>
T distance_to_edge(
    const std::vector<T>& xi,
    const std::vector<T>& x1,
    const std::vector<T>& x1_normal,
    const std::vector<T>& x2,
    const std::vector<T>& x2_normal
) {
    std::vector<T> average_normal
        = {(x2_normal[0] + x1_normal[0]) / 2,
           (x2_normal[1] + x1_normal[1]) / 2};
    T norm_normal = math::loopless::norm<2>(average_normal.data());
    average_normal[0] /= norm_normal;
    average_normal[1] /= norm_normal;

    std::vector<T> x12         = {x2[0] - x1[0], x2[1] - x1[1]};
    T mag_x12                  = sqrtf(x12[0] * x12[0] + x12[1] * x12[1]);
    std::vector<T> e1          = {x12[0] / mag_x12, x12[1] / mag_x12};
    std::vector<T> edge_normal = {e1[1], -e1[0]};
    if (edge_normal[0] * average_normal[0] + edge_normal[1] * average_normal[1]
        < 0) {
        edge_normal[0] *= -1;
        edge_normal[1] *= -1;
    }
    std::vector<T> x1i = {xi[0] - x1[0], xi[1] - x1[1]};
    T mag_x1i          = sqrtf(x1i[0] * x1i[0] + x1i[1] * x1i[1]);
    T x1i_dot_e1       = x1i[0] * e1[0] + x1i[1] * e1[1];
    T x1i_dot_normal   = x1i[0] * edge_normal[0] + x1i[1] * edge_normal[1];
    T distance         = std::abs(x1i_dot_normal);
    if (x1i_dot_e1 > mag_x12) {
        T mag_x2i = sqrtf(
            (xi[0] - x2[0]) * (xi[0] - x2[0])
            + (xi[1] - x2[1]) * (xi[1] - x2[1])
        );
        distance = mag_x2i;
    } else if (x1i_dot_e1 <= 0)
        distance = mag_x1i;

    if (x1i_dot_normal <= 0)
        distance *= -1;

    return distance;
}

template<class T>
T get_distance_to_contour(
    const std::vector<T>& p_new,
    const std::vector<T>& vertices,
    const std::vector<T>& vertex_normals,
    const std::vector<size_t>& edges
) {
    std::vector<size_t> edge = {edges[0], edges[1]};
    auto min_distance        = distance_to_edge<T>(
        p_new,
        {vertices[2 * edge[0]], vertices[2 * edge[0] + 1]},
        {vertex_normals[2 * edge[0]], vertex_normals[2 * edge[0] + 1]},
        {vertices[2 * edge[1]], vertices[2 * edge[1] + 1]},
        {vertex_normals[2 * edge[1]], vertex_normals[2 * edge[1] + 1]}
    );
    for (size_t e = 1; e < edges.size() / 2; e++) {
        edge          = {edges[2 * e], edges[2 * e + 1]};
        auto distance = distance_to_edge<T>(
            p_new,
            {vertices[2 * edge[0]], vertices[2 * edge[0] + 1]},
            {vertex_normals[2 * edge[0]], vertex_normals[2 * edge[0] + 1]},
            {vertices[2 * edge[1]], vertices[2 * edge[1] + 1]},
            {vertex_normals[2 * edge[1]], vertex_normals[2 * edge[1] + 1]}
        );
        if (std::abs(distance) < std::abs(min_distance))
            min_distance = distance;
    }

    return min_distance;
}
template<class T>
T get_distance_to_contour(
    const std::vector<T>& p_new,
    const closed_contour_t<T>& boundary_contours
) {
    std::atomic<T> min_distance = std::numeric_limits<T>::max();

#pragma omp parallel for
    for (size_t e = 0; e < boundary_contours.edges.shape()[1]; e++) {
        std::vector<size_t> edge{
            boundary_contours.edges(0, e),
            boundary_contours.edges(1, e)};
        auto distance = distance_to_edge<T>(
            p_new,
            {boundary_contours.vertices(0, edge[0]),
             boundary_contours.vertices(1, edge[0])},
            {boundary_contours.vertex_normals(0, edge[0]),
             boundary_contours.vertex_normals(1, edge[0])},
            {boundary_contours.vertices(0, edge[1]),
             boundary_contours.vertices(1, edge[1])},
            {boundary_contours.vertex_normals(0, edge[1]),
             boundary_contours.vertex_normals(1, edge[1])}
        );
        if (std::abs(distance) < std::abs(min_distance))
            min_distance = distance;
    }
    return min_distance;
}

template<class T>
bool test_point_with_boundaries(
    const std::vector<T>& p_new,
    const T& particle_spacing,
    const pps_temporary_layer_2d_t<T>& outer_boundary_layer,
    const std::vector<pps_temporary_layer_2d_t<T>>& inner_boundary_layers,
    pps_temporary_layer_2d_t<T>* excluded_layer,
    bool debug = false
) {
    T min_distance_to_edge  = std::numeric_limits<T>::max();
    T min_distance_to_layer = std::numeric_limits<T>::max();
    for (const pps_temporary_layer_2d_t<T>& layer : inner_boundary_layers) {
        if (&layer == excluded_layer) {
            continue;
        }
        std::vector<size_t> edges(
            layer.source_contour.edges.begin(),
            layer.source_contour.edges.end()
        );
        T distance = get_distance_to_contour(
            p_new,
            layer.points,
            layer.field_lines,
            edges
        );
        if (distance <= .3f * particle_spacing) {
            return false;
        }
    }

    T distance
        = get_distance_to_contour(p_new, outer_boundary_layer.source_contour);
    min_distance_to_edge = distance;

    if (min_distance_to_edge < .76f * particle_spacing
        || min_distance_to_layer <= .2f * particle_spacing) {
        if (debug) {
            if (min_distance_to_edge < .76f * particle_spacing) {
                std::cout << "Rejected due to edge distance: "
                          << min_distance_to_edge << std::endl
                          << std::endl;
            } else {
                std::cout << "Rejected due to layer distance: "
                          << min_distance_to_layer << std::endl
                          << std::endl;
            }
        }
        return false;
    }

    return true;
}

template<class T>
bool is_point_valid(
    const T* query_pt,
    const pps_temporary_layer_2d_t<T>& outer_boundary_layer,
    nanoflann_cloud_t<T>& outer_boundary_vertices,
    const kd_tree_t<T>& outer_boundary_kd_tree,
    T min_outer_edge_length,
    const std::vector<pps_temporary_layer_2d_t<T>>& inner_boundary_layers,
    pps_temporary_layer_2d_t<T>* excluded_layer,
    const T& layer_thickness,
    bool debug = false
) {
    if (debug) {
        std::cout << "query_pt: " << query_pt[0] << ", " << query_pt[1]
                  << std::endl;
    }

    if (!outer_boundary_vertices.pts.empty()) {
        int num_results = 1;
        std::vector<size_t> ret_index(num_results);
        std::vector<T> out_dist_sqr(num_results);
        nanoflann::KNNResultSet<T> resultSet(num_results);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        outer_boundary_kd_tree
            .findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

        if (out_dist_sqr[0]
                < .9f * .9f * min_outer_edge_length * min_outer_edge_length
            && ret_index[0] != outer_boundary_vertices.pts.size()) {
            return false;
        }
        if (debug) {
            std::cout << "out_dist: " << sqrt(out_dist_sqr[0]) << std::endl;
            std::cout << "out_idx: " << ret_index[0] << std::endl;
            std::cout << "number of points: "
                      << outer_boundary_vertices.pts.size() << std::endl;
            std::cout << "min_outer_edge_length: " << min_outer_edge_length
                      << std::endl
                      << std::endl;
        }
    }

    if (debug) {
        std::cout << "passed outer boundary check" << std::endl;
    }

    if (!test_point_with_boundaries(
            {query_pt[0], query_pt[1]},
            layer_thickness,
            outer_boundary_layer,
            inner_boundary_layers,
            excluded_layer,
            debug
        )) {
        return false;
    }

    if (debug) {
        std::cout << "passed inner boundary check" << std::endl;
    }

    return true;
}

template<class T>
void advance_front(
    uint layer_idx,
    nanoflann_cloud_t<T>& outer_boundary_vertices,
    kd_tree_t<T>& outer_boundary_kd_tree,
    T min_outer_edge_length,
    std::vector<T>& bulk_points,
    std::vector<T>& layer_points,
    std::vector<T>& layer_boundary_absorption,
    pps_temporary_layer_2d_t<T>& outer_boundary_layer,
    std::vector<pps_temporary_layer_2d_t<T>>& inner_boundary_layers,
    T layer_thickness
) {
    // advance outer boundary
    for (size_t i = 0;
         i < outer_boundary_layer.source_contour.vertices.shape()[1];
         i++) {
        if (layer_idx >= outer_boundary_layer.node_layer_count)
            break;
        std::vector<T> field{
            outer_boundary_layer.source_contour.vertex_normals(0, i),
            outer_boundary_layer.source_contour.vertex_normals(1, i)};
        T query_pt[3] = {
            outer_boundary_layer.points[2 * i] + layer_thickness * field[0],
            outer_boundary_layer.points[2 * i + 1] + layer_thickness * field[1],
            0};

        bool valid_point = is_point_valid(
            query_pt,
            outer_boundary_layer,
            outer_boundary_vertices,
            outer_boundary_kd_tree,
            min_outer_edge_length,
            inner_boundary_layers,
            &outer_boundary_layer,
            layer_thickness
        );

        if (valid_point) {
            outer_boundary_vertices.pts.push_back({query_pt[0], query_pt[1], 0}
            );
            outer_boundary_kd_tree.addPoints(
                outer_boundary_vertices.pts.size() - 1,
                outer_boundary_vertices.pts.size()
            );
            if (layer_idx < outer_boundary_layer.absorption_layer_count) {
                layer_points.push_back(query_pt[0]);
                layer_points.push_back(query_pt[1]);
                layer_boundary_absorption.push_back(
                    outer_boundary_layer.absorption_coefficient
                    * std::pow<T>(
                        (outer_boundary_layer.absorption_layer_count - 1
                         - layer_idx)
                            / ( T
                            ) (outer_boundary_layer.absorption_layer_count - 1),
                        2
                    )
                );
            } else {
                bulk_points.push_back(query_pt[0]);
                bulk_points.push_back(query_pt[1]);
            }
        }
        outer_boundary_layer.points[2 * i]     = query_pt[0];
        outer_boundary_layer.points[2 * i + 1] = query_pt[1];
    }

    // advance inner boundaries
    for (pps_temporary_layer_2d_t<T>& layer : inner_boundary_layers) {
        if (layer_idx >= layer.node_layer_count)
            continue;
        add_layer(layer, layer_thickness);
        for (size_t i = 0; i < layer.points.size(); i += 2) {
            T query_point[3] = {layer.points[i], layer.points[i + 1], 0};
            if (is_point_valid(
                    query_point,
                    outer_boundary_layer,
                    outer_boundary_vertices,
                    outer_boundary_kd_tree,
                    min_outer_edge_length,
                    inner_boundary_layers,
                    &layer,
                    layer_thickness
                )) {
                if (layer_idx < layer.absorption_layer_count) {
                    layer_boundary_absorption.push_back(
                        layer.absorption_coefficient
                        * std::pow<T>(
                            (layer.absorption_layer_count - layer_idx - 1)
                                / ( T ) (layer.absorption_layer_count - 1),
                            2
                        )
                    );
                    layer_points.push_back(query_point[0]);
                    layer_points.push_back(query_point[1]);
                } else {
                    bulk_points.push_back(query_point[0]);
                    bulk_points.push_back(query_point[1]);
                }
            }
        }
    }
}

template<class T>
T get_smallest_edge_length(const closed_contour_t<T>& contour) {
    T min_edge_length = std::numeric_limits<T>::max();
    for (size_t i = 0; i < contour.edges.shape()[1]; i++) {
        T diff[2]
            = {contour.vertices(0, contour.edges(0, i))
                   - contour.vertices(0, contour.edges(1, i)),
               contour.vertices(1, contour.edges(0, i))
                   - contour.vertices(1, contour.edges(1, i))};
        T edge_length = math::loopless::norm<2>(diff);
        if (edge_length < min_edge_length) {
            min_edge_length = edge_length;
        }
    }
    return min_edge_length;
}

template<class T>
std::vector<std::vector<T>>
get_contour_bounds(const closed_contour_t<T>& contour) {
    std::vector<std::vector<T>> bounds(2, std::vector<T>(2));
    bounds[0][0] = std::numeric_limits<T>::max();
    bounds[0][1] = std::numeric_limits<T>::lowest();
    bounds[1][0] = std::numeric_limits<T>::max();
    bounds[1][1] = std::numeric_limits<T>::lowest();
    for (size_t i = 0; i < contour.vertices.shape()[1]; i++) {
        if (contour.vertices(0, i) < bounds[0][0]) {
            bounds[0][0] = contour.vertices(0, i);
        }
        if (contour.vertices(0, i) > bounds[0][1]) {
            bounds[0][1] = contour.vertices(0, i);
        }
        if (contour.vertices(1, i) < bounds[1][0]) {
            bounds[1][0] = contour.vertices(1, i);
        }
        if (contour.vertices(1, i) > bounds[1][1]) {
            bounds[1][1] = contour.vertices(1, i);
        }
    }
    return bounds;
}

template<class T>
simulation_domain<T> hycaps2d_load_domain(
    const boundary_specification<T>& outer_boundary,
    const std::vector<boundary_specification<T>>& inner_boundaries,
    const T& particle_spacing
) {
    std::vector<T> bulk_points;
    std::vector<T> layer_points;
    std::vector<T> layer_absorption;
    std::vector<T> boundary_points;
    std::vector<T> boundary_normals;

    /********************** outer boundary ************************/
    auto domain_contour = mesh::closed_contour_t<T>::import(
        outer_boundary.obj_file_path,
        true,
        particle_spacing
    );
    // flip normals to point inwards
    sclx::algorithm::transform(
        domain_contour.vertex_normals,
        domain_contour.vertex_normals,
        T{-1},
        sclx::algorithm::multiplies<>()
    );
    T min_outer_edge_length   = get_smallest_edge_length(domain_contour);
    auto outer_boundary_layer = init_pps_layer(
        domain_contour,
        outer_boundary.node_layer_count,
        outer_boundary.absorption_layer_count,
        outer_boundary.absorption_coefficient
    );
    for (size_t i = 0; i < domain_contour.vertices.shape()[1]; i++) {
        boundary_points.push_back(domain_contour.vertices(0, i));
        boundary_points.push_back(domain_contour.vertices(1, i));
        boundary_normals.push_back(domain_contour.vertex_normals(0, i));
        boundary_normals.push_back(domain_contour.vertex_normals(1, i));
    }

    /********************** inner boundaries ************************/
    std::vector<pps_temporary_layer_2d_t<T>> inner_boundary_layers;
    for (const boundary_specification<T>& inner_boundary : inner_boundaries) {
        auto inner_boundary_contour = mesh::closed_contour_t<T>::import(
            inner_boundary.obj_file_path,
            true,
            particle_spacing
        );
        inner_boundary_layers.emplace_back(init_pps_layer(
            inner_boundary_contour,
            inner_boundary.node_layer_count,
            inner_boundary.absorption_layer_count,
            inner_boundary.absorption_coefficient
        ));

        for (size_t i = 0; i < inner_boundary_contour.vertices.shape()[1];
             i++) {
            boundary_points.push_back(inner_boundary_contour.vertices(0, i));
            boundary_points.push_back(inner_boundary_contour.vertices(1, i));
            boundary_normals.push_back(
                inner_boundary_contour.vertex_normals(0, i)
            );
            boundary_normals.push_back(
                inner_boundary_contour.vertex_normals(1, i)
            );
        }
    }

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
            min_outer_edge_length,
            bulk_points,
            layer_points,
            layer_absorption,
            outer_boundary_layer,
            inner_boundary_layers,
            particle_spacing
        );
    }

    auto domain_bounds = get_contour_bounds(domain_contour);
    size_t num_x
        = (domain_bounds[0][1] - domain_bounds[0][0]) / particle_spacing;
    size_t num_y
        = (domain_bounds[1][1] - domain_bounds[1][0]) / particle_spacing;
    T x_spacing = (domain_bounds[0][1] - domain_bounds[0][0]) / num_x;
    T y_spacing = (domain_bounds[1][1] - domain_bounds[1][0]) / num_y;
    bulk_points.reserve(num_x * num_y * 2 + bulk_points.size());
    std::vector<bool> is_valid(num_x * num_y, false);

#pragma omp parallel for
    for (size_t n = 0; n < num_x * num_y; n++) {
        size_t i = n / num_y;
        size_t j = n % num_y;
        T query_point[3]
            = {domain_bounds[0][0] + i * x_spacing,
               domain_bounds[1][0] + j * y_spacing,
               0.f};
        if (is_point_valid(
                query_point,
                outer_boundary_layer,
                outer_boundary_cloud,
                outer_boundary_tree,
                min_outer_edge_length,
                inner_boundary_layers,
                ( pps_temporary_layer_2d_t<T>* ) nullptr,
                particle_spacing
            )) {
            is_valid[i * num_y + j] = true;
        }
    }
    for (size_t i = 0; i < num_x; i++) {
        for (size_t j = 0; j < num_y; j++) {
            if (is_valid[i * num_y + j]) {
                bulk_points.push_back(domain_bounds[0][0] + i * x_spacing);
                bulk_points.push_back(domain_bounds[1][0] + j * y_spacing);
            }
        }
    }

    simulation_domain<T> domain;

    uint dims            = 2;
    size_t bulk_size     = bulk_points.size() / dims;
    size_t layer_size    = layer_points.size() / dims;
    size_t boundary_size = boundary_points.size() / dims;

    domain.points = sclx::array<T, 2>(
        sclx::shape_t<2>{dims, bulk_size + layer_size + boundary_size}
    );

    if (bulk_size != 0) {
        domain.points
            .copy_range_from({0, 0}, {0, bulk_size}, bulk_points.data());
    }
    if (layer_size != 0) {
        domain.points.copy_range_from(
            {0, bulk_size},
            {0, bulk_size + layer_size},
            layer_points.data()
        );
    }
    if (boundary_size != 0) {
        domain.points.copy_range_from(
            {0, bulk_size + layer_size},
            {0, bulk_size + layer_size + boundary_size},
            boundary_points.data()
        );
    }

    if (boundary_size != 0) {
        domain.boundary_normals = sclx::array<T, 2>(
            sclx::shape_t<2>{dims, boundary_size},
            boundary_normals.data()
        );
    }

    if (layer_size != 0) {
        domain.layer_absorption = sclx::array<T, 1>(
            sclx::shape_t<1>{layer_size},
            layer_absorption.data()
        );
    }

    domain.num_bulk_points     = bulk_size;
    domain.num_layer_points    = layer_size;
    domain.num_boundary_points = boundary_size;
    domain.nodal_spacing       = particle_spacing;

    return domain;
}

template<class T>
using function_type = decltype(hycaps2d_load_domain<T>);

template function_type<float> hycaps2d_load_domain;

}  // namespace naga::fluids::nonlocal_lbm::detail