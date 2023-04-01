
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
#include "point.cuh"
#include <scalix/array.cuh>

namespace naga {

template<class PointMapType>
struct point_map_traits {
    using point_type                 = typename PointMapType::point_type;
    using point_traits = point_traits<point_type>;
};

template<class FloatingPointType, uint Dimensions>
class default_point_map {
  public:
    using point_type = point_view_t<const FloatingPointType, Dimensions>;

    __host__ __device__ explicit default_point_map(
        const sclx::array<typename point_type::value_type, point_type::dimensions>& source_points
    )
        : source_points_(source_points) {}

    __host__ __device__ point_type operator[](const sclx::md_index_t<1> &index) const {
        return point_type(&source_points_(0, index[0]));
    }

    __host__ __device__ point_type operator[](const sclx::index_t &index) const {
        return point_type(&source_points_(0, index));
    }

    __host__ __device__ size_t size() const {
        return source_points_.shape()[1];
    }

  private:
    sclx::array<typename point_type::value_type, point_type::dimensions> source_points_;
};

}  // namespace naga
