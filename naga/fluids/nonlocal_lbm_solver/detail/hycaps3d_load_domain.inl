
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

#include "../../../distance_functions.cuh"
#include "../../../math.cuh"
#include "../../../mesh/closed_surface.cuh"
#include <scalix/algorithm/transform.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

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
std::vector<T> calculate_field(
    const T *p,
    const mesh::closed_surface_t<T>& source_surface
) {
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

template <class T>
T get_min_edge_length_of_all(
    const pps_temporary_layer_3d_t<T>& outer_boundary_layer,
    const std::vector<pps_temporary_layer_3d_t<T>>& inner_boundary_layers
) {
    T min_edge_length = std::numeric_limits<T>::max();

    // outer boundary
    for (size_t f = 0; f < outer_boundary_layer.source_surface.faces.shape()[1]; f++) {
        for (size_t v = 0; v < 3; v++) {
            size_t v1 = outer_boundary_layer.source_surface.faces(v, f);
            size_t v2 = outer_boundary_layer.source_surface.faces((v + 1) % 3, f);
            T v12[3] = {
                outer_boundary_layer.points[v1 * 3 + 0] - outer_boundary_layer.points[v2 * 3 + 0],
                outer_boundary_layer.points[v1 * 3 + 1] - outer_boundary_layer.points[v2 * 3 + 1],
                outer_boundary_layer.points[v1 * 3 + 2] - outer_boundary_layer.points[v2 * 3 + 2]
            };
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
                T v12[3] = {
                    layer.points[v1 * 3 + 0] - layer.points[v2 * 3 + 0],
                    layer.points[v1 * 3 + 1] - layer.points[v2 * 3 + 1],
                    layer.points[v1 * 3 + 2] - layer.points[v2 * 3 + 2]
                };
                auto edge_len = math::loopless::norm<3>(v12);
                if (edge_len < min_edge_length) {
                    min_edge_length = edge_len;
                }
            }
        }
    }

    return min_edge_length;
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
    T min_outer_edge_len = get_smallest_edge_length(domain_surface);
    auto outer_boundary_layer = init_pps_layer(
        domain_surface,
        outer_boundary.node_layer_count,
        outer_boundary.absorption_layer_count,
        outer_boundary.absorption_coefficient
    );

    /********************** inner boundaries ************************/
    std::vector<pps_temporary_layer_3d_t<T>> inner_boundary_layers;
    for (const boundary_specification<T>& boundary : inner_boundaries) {
        auto inner_surface = mesh::closed_surface_t<T>::import(
            boundary.obj_file_path
        );
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
}

template<class T>
using f_type = decltype(hycaps3d_load_domain<T>);

template f_type<float> hycaps3d_load_domain;

}  // namespace naga::fluids::nonlocal_lbm::detail
