
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

#include "detail/hycaps2d_load_domain.cuh"
#include "detail/hycaps3d_load_domain.cuh"
#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>

namespace naga::fluids::nonlocal_lbm {

template<class T>
struct boundary_specification {
    sclx::filesystem::path obj_file_path;
    uint node_layer_count;
    uint absorption_layer_count;
    T absorption_coefficient;

    boundary_specification(
        sclx::filesystem::path obj_file_path,
        uint node_layer_count,
        uint absorption_layer_count,
        T absorption_coefficient
    )
        : obj_file_path(std::move(obj_file_path)),
          node_layer_count(node_layer_count),
          absorption_layer_count(absorption_layer_count),
          absorption_coefficient(absorption_coefficient) {
        if (absorption_layer_count > node_layer_count) {
            sclx::throw_exception<std::runtime_error>(
                "Absorption layer count must be less than or equal to node "
                "layer count",
                "naga::fluids::nonlocal_lbm::"
            );
        }
    }
};

template<class T>
struct simulation_domain {
    sclx::array<T, 2> points{};  // order: bulk, boundary layers, boundary
    sclx::array<T, 2> boundary_normals{};
    sclx::array<T, 1> layer_absorption{};  // absorption coefficient for each
                                           // layer, with size num_layer_points
    size_t num_bulk_points{};
    size_t num_layer_points{};
    size_t num_boundary_points{};

    T nodal_spacing{};

    template<uint Dimensions>
    static simulation_domain import(
        const boundary_specification<T>& outer_boundary,
        const std::vector<boundary_specification<T>>& inner_boundaries,
        const T& particle_spacing = T(0)
    ) {
        static_assert(
            Dimensions == 2 || Dimensions == 3,
            "Dimensions must be 2 or 3"
        );

        if constexpr (Dimensions == 3) {
            if (particle_spacing != T(0)) {
                sclx::throw_exception<std::runtime_error>(
                    "Particle spacing must be 0 for 3D simulations as "
                    "subdivision is not supported",
                    "naga::fluids::nonlocal_lbm::"
                );
            }
            return detail::hycaps3d_load_domain<T>(
                outer_boundary,
                inner_boundaries
            );
        } else {
            return detail::hycaps2d_load_domain<T>(
                outer_boundary,
                inner_boundaries,
                particle_spacing
            );
        }
    }

    template <class T_ = const T>
    operator simulation_domain<T_>() const { // NOLINT(google-explicit-constructor)
        return simulation_domain<T_>{
            points,
            boundary_normals,
            layer_absorption,
            num_bulk_points,
            num_layer_points,
            num_boundary_points,
            nodal_spacing
        };
    }
};

}  // namespace naga::fluids::nonlocal_lbm

#include "detail/hycaps2d_load_domain.inl"
#include "detail/hycaps3d_load_domain.inl"
