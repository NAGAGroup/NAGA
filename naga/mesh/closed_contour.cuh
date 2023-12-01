
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

#include "../math.hpp"
#include "triangular_mesh.cuh"
#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>

namespace naga::mesh {

template<class FPType>
bool is_edge_shared(
    uint face,
    const int* face_edge,
    const triangular_mesh_t<FPType>& mesh
) {
    for (uint face_check = 0; face_check < mesh.faces.shape()[1];
         face_check++) {
        if (face_check == face) {
            continue;
        }
        for (uint fe = 0; fe < 3; fe++) {
            uint face_edge_check[2] = {fe, (fe + 1) % 3};
            if (mesh.faces(face_edge_check[0], face_check)
                    == mesh.faces(face_edge[0], face)
                && mesh.faces(face_edge_check[1], face_check)
                       == mesh.faces(face_edge[1], face)) {
                return true;
            }
            if (mesh.faces(face_edge_check[0], face_check)
                    == mesh.faces(face_edge[1], face)
                && mesh.faces(face_edge_check[1], face_check)
                       == mesh.faces(face_edge[0], face)) {
                return true;
            }
        }
    }
    return false;
}

template<class FPType>
std::vector<FPType> calc_edge_normal_of_face(
    uint face,
    const int* face_edge,
    const triangular_mesh_t<FPType>& mesh
) {
    std::vector<FPType> edge_normal(2);

    FPType v1[2]
        = {mesh.vertices(0, mesh.faces(face_edge[0], face)),
           mesh.vertices(1, mesh.faces(face_edge[0], face))};
    FPType v2[2]
        = {mesh.vertices(0, mesh.faces(face_edge[1], face)),
           mesh.vertices(1, mesh.faces(face_edge[1], face))};
    int v3_id = (face_edge[1] + 1) % 3;
    FPType v3[2]
        = {mesh.vertices(0, mesh.faces(v3_id, face)),
           mesh.vertices(1, mesh.faces(v3_id, face))};

    // perpendicular vector to edge v1-v2
    edge_normal[0] = v2[1] - v1[1];
    edge_normal[1] = v1[0] - v2[0];

    FPType v1_to_v3[2] = {v3[0] - v1[0], v3[1] - v1[1]};

    if (math::loopless::dot<2>(edge_normal.data(), v1_to_v3) > 0) {
        edge_normal[0] = -edge_normal[0];
        edge_normal[1] = -edge_normal[1];
    }

    auto normal_norm = math::loopless::norm<2>(edge_normal.data());
    edge_normal[0] /= normal_norm;
    edge_normal[1] /= normal_norm;

    return edge_normal;
}

template<class FPType>
void populate_contour_data(
    const std::vector<uint>& mesh_edges,
    const std::vector<FPType>& mesh_edge_normals,
    std::vector<uint>& contour_edges,
    std::vector<FPType>& contour_vertices,
    std::vector<FPType>& contour_vertex_normals,
    const triangular_mesh_t<FPType>& mesh,
    bool subdivide            = false,
    FPType approx_edge_length = 0.
) {
    contour_edges = mesh_edges;
    contour_vertices.clear();
    contour_vertices.reserve(mesh.vertices.shape()[1] * 2);
    contour_vertex_normals.clear();
    contour_vertex_normals.reserve(mesh.vertices.shape()[1] * 2);

    // get unique vertex ids from mesh edges
    std::vector<uint> unique_vertex_ids(mesh_edges.begin(), mesh_edges.end());
    std::sort(unique_vertex_ids.begin(), unique_vertex_ids.end());
    auto last = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
    unique_vertex_ids.erase(last, unique_vertex_ids.end());

    // populate contour vertices and normals
    for (auto& id : unique_vertex_ids) {
        contour_vertices.push_back(mesh.vertices(0, id));
        contour_vertices.push_back(mesh.vertices(1, id));
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
    for (uint e = 0; e < num_edges; e++) {
        uint edge_start = contour_edges[e * 2];
        uint edge_end   = contour_edges[e * 2 + 1];
        FPType edge_start_pos[2]
            = {contour_vertices[edge_start * 2],
               contour_vertices[edge_start * 2 + 1]};
        FPType edge_end_pos[2]
            = {contour_vertices[edge_end * 2],
               contour_vertices[edge_end * 2 + 1]};
        FPType edge_dir[2]
            = {edge_end_pos[0] - edge_start_pos[0],
               edge_end_pos[1] - edge_start_pos[1]};
        auto edge_length = math::loopless::norm<2>(edge_dir);
        auto num_subdivisions
            = static_cast<uint>(std::ceil(edge_length / approx_edge_length));
        if (num_subdivisions < 2) {
            continue;
        }

        FPType length_sub_div = edge_length / num_subdivisions;
        edge_dir[0] /= edge_length;
        edge_dir[1] /= edge_length;

        for (uint s = 1; s < num_subdivisions; s++) {
            FPType sub_div_pos[2]
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

template<class FPType>
struct closed_contour_t {
    sclx::array<FPType, 2> vertices;
    sclx::array<FPType, 2> vertex_normals;
    sclx::array<uint, 2> edges;

    static closed_contour_t import(
        const sclx::filesystem::path& path,
        bool subdivide            = false,
        FPType approx_edge_length = 0.
    ) {
        auto mesh = triangular_mesh_t<FPType>::import(path);

        std::vector<uint> mesh_edges;
        std::vector<FPType> mesh_edge_normals;

        for (uint face = 0; face < mesh.faces.shape()[1]; face++) {
            for (int e = 0; e < 3; e++) {
                int face_edge[2] = {e, (e + 1) % 3};
                if (is_edge_shared(face, face_edge, mesh))
                    continue;
                mesh_edges.push_back(mesh.faces(face_edge[0], face));
                mesh_edges.push_back(mesh.faces(face_edge[1], face));
                auto edge_normal
                    = calc_edge_normal_of_face(face, face_edge, mesh);
                mesh_edge_normals.push_back(edge_normal[0]);
                mesh_edge_normals.push_back(edge_normal[1]);
            }
        }

        std::vector<uint> contour_edges;
        std::vector<FPType> contour_vertices;
        std::vector<FPType> contour_vertex_normals;

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

        return closed_contour_t<FPType>{
            sclx::array<FPType, 2>(
                sclx::shape_t<2>{2, contour_vertices.size() / 2},
                contour_vertices.data()
            ),
            sclx::array<FPType, 2>(
                sclx::shape_t<2>{2, contour_vertex_normals.size() / 2},
                contour_vertex_normals.data()
            ),
            sclx::array<uint, 2>(
                sclx::shape_t<2>{2, contour_edges.size() / 2},
                contour_edges.data()
            )};
    }
};

}  // namespace naga::mesh
