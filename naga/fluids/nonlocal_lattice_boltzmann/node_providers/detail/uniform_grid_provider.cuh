
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

#include "../../../../ranges/uniform_grid.cuh"
#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>

namespace naga::fluids::nonlocal_lbm {
template<class Lattice>
class uniform_grid_provider;
}

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
__host__ void assign_boundary_info(
    const uniform_grid_provider<Lattice>& provider,
    sclx::array<typename Lattice::value_type, 2>& points,
    sclx::array<typename Lattice::value_type, 2>& normals,
    size_t boundary_offset
);

template<class Lattice>
__host__ void assign_bulk_info(
    const uniform_grid_provider<Lattice>& provider,
    sclx::array<typename Lattice::value_type, 2>& points
);

template<class Lattice>
struct boundary_count_criteria_functor {
    using value_type                 = typename Lattice::value_type;
    constexpr static uint dimensions = Lattice::dimensions;
    using range_type = typename ranges::uniform_grid<value_type, dimensions>;

    __host__ __device__ bool operator()(const size_t& i) const {
        return ranges::uniform_grid_inspector<value_type, dimensions>::
            is_boundary_index(grid_range_, i);
    }

    range_type grid_range_;
};

}  // namespace naga::fluids::nonlocal_lbm::detail
