
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
#include <scalix/cexpr_memcpy.cuh>
#include "../distance_functions.cuh"

namespace naga::regions {

template <class T, uint Dimensions>
class hypersphere {
  public:
    template <class VectorT>
    __host__ __device__ hypersphere(const T &radius, const VectorT& center)
        : radius_(radius) {
        for (uint i = 0; i < Dimensions; ++i) {
            this->center_[i] = center[i];
        }
    }

    __host__ __device__ constexpr hypersphere(const T& radius, const T *center)
        : radius_(radius) {
        sclx::cexpr_memcpy<Dimensions>(this->center_, center);
    }

    template <class VectorT>
    __host__ __device__ bool contains(const VectorT& point) const {
        distance_functions::loopless::euclidean_squared<Dimensions> dist;
        return dist(point, center_) <= radius_ * radius_;
    }

  private:
    T radius_;
    T center_[Dimensions];
};

}