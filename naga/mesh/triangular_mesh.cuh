
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

#include <fstream>
#include <sstream>
#include <vector>

namespace naga::mesh {

template<class T>
struct triangular_mesh_t {
    sclx::array<T, 2> vertices;
    sclx::array<T, 2> normals;
    sclx::array<sclx::index_t, 2> faces;
    sclx::array<sclx::index_t, 2>
        face_normals;  // (3 x faces) array of indices into normals

    static triangular_mesh_t import(const sclx::filesystem::path& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            sclx::throw_exception<std::invalid_argument>(
                "Could not open file: " + path.string(),
                "naga::mesh::triangular_mesh_t::"
            );
        }

        std::vector<T> vertices;
        std::vector<T> normals;
        std::vector<sclx::index_t> faces;
        std::vector<sclx::index_t> face_normals;

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            if (type == "v") {
                T x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            } else if (type == "vn") {
                T x, y, z;
                iss >> x >> y >> z;
                normals.push_back(x);
                normals.push_back(y);
                normals.push_back(z);
            } else if (type == "f") {
                // check if face includes texture coordinates and vertex normals
                // if so ignore them
                std::string face;
                while (iss >> face) {
                    std::istringstream face_iss(face);
                    std::string vertex;
                    std::getline(face_iss, vertex, '/');
                    faces.push_back(std::stoi(vertex) - 1);
                    std::getline(face_iss, vertex, '/');
                    std::getline(face_iss, vertex, '/');
                    face_normals.push_back(std::stoi(vertex) - 1);
                }
            }
        }


        return triangular_mesh_t{
            sclx::array<T, 2>(sclx::shape_t<2>{3, vertices.size() / 3}, vertices.data()),
            sclx::array<T, 2>(sclx::shape_t<2>{3, normals.size() / 3}, normals.data()),
            sclx::array<sclx::index_t, 2>(sclx::shape_t<2>{3, faces.size() / 3}, faces.data()),
            sclx::array<sclx::index_t, 2>(sclx::shape_t<2>{3, face_normals.size() / 3}, face_normals.data())
        };
    }
};

}  // namespace naga::mesh
