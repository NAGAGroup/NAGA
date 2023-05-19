
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

#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>
#include "triangular_mesh.cuh"

namespace naga::mesh {

template <class T>
struct closed_surface_t {
    sclx::array<T, 2> vertices;
    sclx::array<T, 2> vertex_normals;
    sclx::array<sclx::index_t, 2> faces;

    static closed_surface_t import(const sclx::filesystem::path &path) {
        auto mesh = triangular_mesh_t<T>::import(path);
        std::vector<T> vertices(mesh.vertices.data().get(), mesh.vertices.data().get() + mesh.vertices.elements());
        std::vector<size_t> faces(mesh.faces.data().get(), mesh.faces.data().get() + mesh.faces.elements());
        std::vector<T> vertex_normals(mesh.vertices.shape()[1] * 3, 0);
        std::vector<bool> vertex_normal_set(mesh.vertices.shape()[1], false);
        for (size_t f = 0; f < mesh.face_normals.shape()[1]; f++) {
            for (uint v = 0; v < 3; v++) {
                size_t global_v = faces[f * 3 + v];
                if (!vertex_normal_set[global_v]) {
                    vertex_normal_set[global_v] = true;
                    for (uint d = 0; d < 3; d++) {
                        vertex_normals[global_v * 3 + d] = mesh.normals(d, mesh.face_normals(v, f));
                    }
                }
            }
        }

        return closed_surface_t{
            sclx::array<T, 2>(sclx::shape_t<2>{3, mesh.vertices.shape()[1]}, vertices.data()),
            sclx::array<T, 2>(sclx::shape_t<2>{3, mesh.vertices.shape()[1]}, vertex_normals.data()),
            sclx::array<sclx::index_t, 2>(sclx::shape_t<2>{3, mesh.faces.shape()[1]}, faces.data())
        };
    }
};

}
