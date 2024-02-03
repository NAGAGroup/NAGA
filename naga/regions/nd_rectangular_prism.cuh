
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
#include "../distance_functions.hpp"
#include <scalix/constexpr_assign_array.cuh>

namespace naga::regions {

template<class T, uint Dimensions>
class nd_rectangular_prism {
  public:
    template<class VectorT, class VectorU>
    __host__ __device__
    nd_rectangular_prism(const VectorT& origin, const VectorU& side_lengths) {
        for (uint i = 0; i < Dimensions; ++i) {
            origin_[i]       = origin[i];
            side_lengths_[i] = side_lengths[i];
            center_[i]       = origin[i] + side_lengths[i] / 2;
        }
    }

    __host__ __device__ constexpr nd_rectangular_prism(
        const T (&origin)[Dimensions],
        const T (&side_lengths)[Dimensions]
    ) {
        sclx::constexpr_assign_array<Dimensions>(origin_, origin);
        sclx::constexpr_assign_array<Dimensions>(side_lengths_, side_lengths);
    }

    template<class VectorT>
    __host__ __device__ bool contains(const VectorT& point) const {
        for (uint i = 0; i < Dimensions; ++i) {
            if (point[i] < origin_[i]
                || point[i] > origin_[i] + side_lengths_[i]) {
                return false;
            }
        }
        return true;
    }

    template<class VectorT>
    __host__ __device__ auto& shift_region(const VectorT& point) {
        for (uint i = 0; i < Dimensions; ++i) {
            origin_[i] += point[i];
        }

        return *this;
    }

    template<class VectorT>
    __host__ __device__ auto shift_region(const VectorT& point) const {
        auto copy = *this;
        copy.shift_region(point);
        return copy;
    }

    __host__ __device__ const T* center() const { return center_; }


  private:
    T origin_[Dimensions];
    T side_lengths_[Dimensions];
    T center_[Dimensions];
};
}  // namespace naga::regions