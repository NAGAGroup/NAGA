
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

namespace naga {

template<class T, uint Dimensions>
class nd_cubic_segmentation;

}

namespace naga::detail {

template<class T, uint Dimensions>
__host__ void compute_bounds(
    point_view_t<T, Dimensions>& lower_bounds,
    point_view_t<T, Dimensions>& upper_bounds,
    const sclx::array<const T, 2>& points
);

template<class T, uint Dimensions>
__host__ void compute_partition_sizes(
    sclx::array<uint, Dimensions>& partition_sizes_,
    const sclx::shape_t<Dimensions>& segmentation_shape,
    sclx::array<const T, 2>& points_,
    const nd_cubic_segmentation<T, Dimensions>& segmentation
);

template<uint Dimensions>
__host__ void compute_index_offsets(
    sclx::array<size_t, Dimensions>& partition_index_offsets,
    const sclx::array<const uint, Dimensions>& partition_sizes
);

template<class T, uint Dimensions>
__host__ void assign_indices(
    sclx::array<size_t, 1>& partition_index,
    const sclx::array<const uint, Dimensions>& partition_sizes_,
    const sclx::array<const size_t, Dimensions>& partition_index_offsets_,
    const sclx::array<const T, 2>& points_,
    const nd_cubic_segmentation<T, Dimensions>& segmentation
);

}  // namespace naga::detail
