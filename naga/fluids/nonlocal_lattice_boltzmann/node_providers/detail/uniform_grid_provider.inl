
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

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
__host__ void assign_boundary_info(
    const uniform_grid_provider<Lattice>& provider,
    sclx::array<typename Lattice::value_type, 2>& points,
    sclx::array<typename Lattice::value_type, 2>& normals,
    size_t boundary_offset
) {
    using value_type          = typename Lattice::value_type;
    constexpr uint dimensions = Lattice::dimensions;

    sclx::array<uint, 1> atomic_counter{1};
    atomic_counter[0] = 0;

    sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
        sclx::array_list<value_type, 2, 2> result_list{
            points,
            normals};

        auto& grid_range = provider.grid_range_;

        handler.launch(
            sclx::md_range_t<1>{grid_range.size()},
            result_list,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                if (ranges::uniform_grid_inspector<value_type, dimensions>::
                        is_boundary_index(grid_range, idx[0])) {
                    uint boundary_count  = atomicAdd(&atomic_counter[0], 1);
                    auto boundary_point  = grid_range[idx[0]];
                    auto boundary_normal = ranges::uniform_grid_inspector<
                        value_type,
                        dimensions>::get_boundary_normal(grid_range, idx[0]);

                    for (uint i = 0; i < dimensions; ++i) {
                        points(i, boundary_count + boundary_offset) = boundary_point[i];
                        normals(i, boundary_count + boundary_offset)
                            = boundary_normal[i];
                    }
                }
            }
        );
    }).get();
}

template<class Lattice>
__host__ void assign_bulk_info(
    const uniform_grid_provider<Lattice>& provider,
    sclx::array<typename Lattice::value_type, 2>& points
) {
    using value_type          = typename Lattice::value_type;
    constexpr uint dimensions = Lattice::dimensions;

    sclx::array<uint, 1> atomic_counter{1};
    atomic_counter[0] = 0;

    sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
        auto& grid_range = provider.grid_range_;

        handler.launch(
            sclx::md_range_t<1>{grid_range.size()},
            points,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                if (!ranges::uniform_grid_inspector<value_type, dimensions>::
                        is_boundary_index(grid_range, idx[0])) {
                    uint bulk_count = atomicAdd(&atomic_counter[0], 1);
                    auto bulk_point = grid_range[idx[0]];

                    for (uint i = 0; i < dimensions; ++i) {
                        points(i, bulk_count) = bulk_point[i];
                    }
                }
            }
        );
    }).get();
}

}  // namespace naga::fluids::nonlocal_lbm::detail
