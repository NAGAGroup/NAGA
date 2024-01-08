
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

namespace naga::fluids::nonlocal_lbm {

template<class T>
struct simulation_nodes {
    sclx::array<T, 2> points{};  // order: bulk, boundary layers, boundary
    sclx::array<T, 2> boundary_normals{};
    sclx::array<T, 1> layer_absorption{};  // absorption coefficient for each
                                           // layer, with size num_layer_points
    size_t num_bulk_points{};
    size_t num_layer_points{};
    size_t num_ghost_nodes{0};
    size_t num_boundary_points{};

    T nodal_spacing{};

    template<class T_ = const T>
    operator simulation_nodes<T_>(
    ) const {  // NOLINT(google-explicit-constructor)
        return simulation_nodes<T_>{
            points,
            boundary_normals,
            layer_absorption,
            num_bulk_points,
            num_layer_points,
            num_ghost_nodes,
            num_boundary_points,
            nodal_spacing};
    }
};

}  // namespace naga::fluids::nonlocal_lbm
