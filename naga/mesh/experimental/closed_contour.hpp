
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

#include "triangular_mesh.hpp"
#include <algorithm>
#include <naga/math.hpp>
#include <naga/point.hpp>

namespace naga::experimental::mesh {

template<class T>
bool is_edge_shared(
    size_t face,
    const int* face_edge,
    const triangular_mesh_t<T>& mesh
) {
    for (size_t face_check = 0; face_check < mesh.faces().size();
         face_check += 3) {
        if (face_check == face) {
            continue;
        }
        for (size_t fe = 0; fe < 3; fe++) {
            size_t face_edge_check[2] = {fe, (fe + 1) % 3};
            if (mesh.faces()[face_check + face_edge_check[0]]
                    == mesh.faces()[face + face_edge[0]]
                && mesh.faces()[face_check + face_edge_check[1]]
                       == mesh.faces()[face + face_edge[1]]) {
                return true;
            }
            if (mesh.faces()[face_check + face_edge_check[0]]
                    == mesh.faces()[face + face_edge[1]]
                && mesh.faces()[face_check + face_edge_check[1]]
                       == mesh.faces()[face + face_edge[0]]) {
                return true;
            }
        }
    }
    return false;
}

template<class T>
std::vector<T>
calc_v12_edge_normal_of_tri(const T* v1, const T* v2, const T* v3) {
    std::vector<T> edge_normal(2);

    // perpendicular vector to edge v1-v2
    edge_normal[0] = v2[1] - v1[1];
    edge_normal[1] = v1[0] - v2[0];

    T v1_to_v3[2] = {v3[0] - v1[0], v3[1] - v1[1]};

    if (math::loopless::dot<2>(edge_normal.data(), v1_to_v3) > 0) {
        edge_normal[0] = -edge_normal[0];
        edge_normal[1] = -edge_normal[1];
    }

    auto normal_norm = math::loopless::norm<2>(edge_normal.data());
    edge_normal[0] /= normal_norm;
    edge_normal[1] /= normal_norm;

    return edge_normal;
}

template<class T>
std::vector<T> calc_edge_normal_of_face(
    size_t face,
    const int* face_edge,
    const triangular_mesh_t<T>& mesh
) {

    T v1[2]
        = {mesh.vertices()[3 * mesh.faces()[face + face_edge[0]] + 0],
           mesh.vertices()[3 * mesh.faces()[face + face_edge[0]] + 1]};
    T v2[2]
        = {mesh.vertices()[3 * mesh.faces()[face + face_edge[1]] + 0],
           mesh.vertices()[3 * mesh.faces()[face + face_edge[1]] + 1]};
    int v3_id = (face_edge[1] + 1) % 3;
    T v3[2]
        = {mesh.vertices()[3 * mesh.faces()[face + v3_id] + 0],
           mesh.vertices()[3 * mesh.faces()[face + v3_id] + 1]};

    return calc_v12_edge_normal_of_tri(v1, v2, v3);
}

template<class T>
void populate_contour_data(
    const std::vector<std::uint32_t>& mesh_edges,
    const std::vector<T>& mesh_edge_normals,
    std::vector<std::uint32_t>& contour_edges,
    std::vector<T>& contour_vertices,
    std::vector<T>& contour_vertex_normals,
    const triangular_mesh_t<T>& mesh,
    bool subdivide       = false,
    T approx_edge_length = 0.
) {
    contour_edges = mesh_edges;
    contour_vertices.clear();
    contour_vertices.reserve(mesh.vertices().size() / 3 * 2);
    contour_vertex_normals.clear();
    contour_vertex_normals.reserve(mesh.vertices().size() / 3 * 2);

    // get unique vertex ids from mesh edges
    std::vector<std::uint32_t> unique_vertex_ids(mesh_edges.begin(), mesh_edges.end());
    std::sort(unique_vertex_ids.begin(), unique_vertex_ids.end());
    auto last = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
    unique_vertex_ids.erase(last, unique_vertex_ids.end());

    // populate contour vertices and normals
    for (auto& id : unique_vertex_ids) {
        contour_vertices.push_back(mesh.vertices()[3 * id + 0]);
        contour_vertices.push_back(mesh.vertices()[3 * id + 1]);
        auto edge_1_it = std::find(mesh_edges.begin(), mesh_edges.end(), id);
        auto edge_1    = std::distance(mesh_edges.begin(), edge_1_it) / 2;
        auto edge_2_it = std::find(
            mesh_edges.begin() + 2 * (edge_1 + 1),
            mesh_edges.end(),
            id
        );
        auto edge_2 = std::distance(mesh_edges.begin(), edge_2_it) / 2;
        float normal[2];
        normal[0] = mesh_edge_normals[2 * edge_1 + 0]
                  + mesh_edge_normals[2 * edge_2 + 0];
        normal[0] /= 2;
        normal[1] = mesh_edge_normals[2 * edge_1 + 1]
                  + mesh_edge_normals[2 * edge_2 + 1];
        normal[1] /= 2;
        auto normal_norm = math::loopless::norm<2>(normal);
        normal[0] /= normal_norm;
        normal[1] /= normal_norm;

        contour_vertex_normals.push_back(normal[0]);
        contour_vertex_normals.push_back(normal[1]);
        contour_edges[std::distance(mesh_edges.begin(), edge_1_it)]
            = (contour_vertices.size()) / 2 - 1;
        contour_edges[std::distance(mesh_edges.begin(), edge_2_it)]
            = (contour_vertices.size()) / 2 - 1;
    }

    if (!subdivide) {
        return;
    }

    auto num_edges = contour_edges.size() / 2;
    for (size_t e = 0; e < num_edges; e++) {
        size_t edge_start = contour_edges[e * 2];
        size_t edge_end   = contour_edges[e * 2 + 1];
        T edge_start_pos[2]
            = {contour_vertices[edge_start * 2],
               contour_vertices[edge_start * 2 + 1]};
        T edge_end_pos[2]
            = {contour_vertices[edge_end * 2],
               contour_vertices[edge_end * 2 + 1]};
        T edge_dir[2]
            = {edge_end_pos[0] - edge_start_pos[0],
               edge_end_pos[1] - edge_start_pos[1]};
        auto edge_length = math::loopless::norm<2>(edge_dir);
        auto num_subdivisions
            = static_cast<uint>(std::ceil(edge_length / approx_edge_length));
        if (num_subdivisions < 2) {
            continue;
        }

        T length_sub_div = edge_length / num_subdivisions;
        edge_dir[0] /= edge_length;
        edge_dir[1] /= edge_length;

        for (size_t s = 1; s < num_subdivisions; s++) {
            T sub_div_pos[2]
                = {edge_start_pos[0] + edge_dir[0] * length_sub_div * s,
                   edge_start_pos[1] + edge_dir[1] * length_sub_div * s};
            contour_vertices.push_back(sub_div_pos[0]);
            contour_vertices.push_back(sub_div_pos[1]);
            contour_vertex_normals.push_back(mesh_edge_normals[2 * e + 0]);
            contour_vertex_normals.push_back(mesh_edge_normals[2 * e + 1]);

            if (s == 1) {
                contour_edges[e * 2 + 1] = contour_vertices.size() / 2 - 1;
            } else {
                contour_edges.push_back(contour_vertices.size() / 2 - 2);
                contour_edges.push_back(contour_vertices.size() / 2 - 1);
            }
        }
        contour_edges.push_back(contour_vertices.size() / 2 - 1);
        contour_edges.push_back(edge_end);
    }
}

template<class T>
class closed_contour_t {
  public:
    using index_t = std::uint32_t;

    closed_contour_t(
        std::vector<T> vertices,
        std::vector<T> vertex_normals,
        std::vector<std::uint32_t> edges,
        point_t<T, 2> lower_bound,
        point_t<T, 2> upper_bound
    )
        : vertices_(std::move(vertices)),
          vertex_normals_(std::move(vertex_normals)),
          edges_(std::move(edges)),
          lower_bound_(std::move(lower_bound)),
          upper_bound_(std::move(upper_bound)) {}

    static closed_contour_t import(
        const std::filesystem::path& path,
        bool subdivide       = false,
        T approx_edge_length = 0.
    ) {
        auto mesh = triangular_mesh_t<T>::import(path);

        std::vector<std::uint32_t> mesh_edges;
        std::vector<T> mesh_edge_normals;

        for (index_t face = 0; face < mesh.faces().size(); face += 3) {
            for (int e = 0; e < 3; e++) {
                int face_edge[2] = {e, (e + 1) % 3};
                if (is_edge_shared(face, face_edge, mesh))
                    continue;
                mesh_edges.push_back(mesh.faces()[face + face_edge[0]]);
                mesh_edges.push_back(mesh.faces()[face + face_edge[1]]);
                auto edge_normal
                    = calc_edge_normal_of_face(face, face_edge, mesh);
                mesh_edge_normals.push_back(edge_normal[0]);
                mesh_edge_normals.push_back(edge_normal[1]);
            }
        }

        std::vector<std::uint32_t> contour_edges;
        std::vector<T> contour_vertices;
        std::vector<T> contour_vertex_normals;

        populate_contour_data(
            mesh_edges,
            mesh_edge_normals,
            contour_edges,
            contour_vertices,
            contour_vertex_normals,
            mesh,
            subdivide,
            approx_edge_length
        );

        point_t<T, 2> lower_bound(
            {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()}
        );
        point_t<T, 2> upper_bound(
            {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()}
        );

        for (size_t i = 0; i < contour_vertices.size(); i += 2) {
            lower_bound[0] = std::min(lower_bound[0], contour_vertices[i]);
            lower_bound[1] = std::min(lower_bound[1], contour_vertices[i + 1]);
            upper_bound[0] = std::max(upper_bound[0], contour_vertices[i]);
            upper_bound[1] = std::max(upper_bound[1], contour_vertices[i + 1]);
        }

        return closed_contour_t<T>{
            std::move(contour_vertices),
            std::move(contour_vertex_normals),
            std::move(contour_edges),
            std::move(lower_bound),
            std::move(upper_bound)};
    }

    const std::vector<T>& vertices() const { return vertices_; }

    const std::vector<T>& vertex_normals() const { return vertex_normals_; }

    const std::vector<std::uint32_t>& edges() const { return edges_; }

    const point_t<T, 2>& lower_bound() const { return lower_bound_; }

    const point_t<T, 2>& upper_bound() const { return upper_bound_; }

  private:
    std::vector<T> vertices_;
    std::vector<T> vertex_normals_;
    std::vector<std::uint32_t> edges_;
    point_t<T, 2> lower_bound_;
    point_t<T, 2> upper_bound_;
};

}  // namespace naga::experimental::mesh
