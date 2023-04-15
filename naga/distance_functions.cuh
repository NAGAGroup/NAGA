
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
#include "math.cuh"

namespace naga::distance_functions {

template<class DistanceFunctor>
struct distance_function_traits {
    static constexpr bool is_loopless = DistanceFunctor::is_loopless;
    static constexpr bool is_squared  = DistanceFunctor::is_squared;
};

template<class VectorLikeT = void, class VectorLikeU = void>
struct euclidean_squared {
    static constexpr bool is_loopless = false;
    static constexpr bool is_squared  = true;

    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint dimensions)
        const {
        using value_type = decltype(a[0] - b[0]);
        value_type sum   = 0;
        for (uint i = 0; i < dimensions; ++i) {
            value_type diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};

template<>
struct euclidean_squared<void, void> {
    static constexpr bool is_loopless = false;
    static constexpr bool is_squared  = true;

    template<class VectorLikeT, class VectorLikeU>
    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint dimensions)
        const {
        using value_type = decltype(a[0] - b[0]);
        value_type sum   = 0;
        for (uint i = 0; i < dimensions; ++i) {
            value_type diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};

template<class VectorLikeT = void, class VectorLikeU = void>
struct euclidean {
    static constexpr bool is_loopless = false;
    static constexpr bool is_squared  = false;

    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint dimensions)
        const {
        return math::sqrt(
            euclidean_squared<VectorLikeT, VectorLikeU>{}(a, b, dimensions)
        );
    }
};

}  // namespace naga::distance_functions

namespace naga::distance_functions::loopless {

template<uint Dimensions, class VectorLikeT = void, class VectorLikeU = void>
class euclidean_squared {
  public:
    static constexpr bool is_loopless = true;
    static constexpr bool is_squared  = true;

    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint) const {
        return accumulate_pow2_diffs(a, b);
    }

  private:
    template<uint D=Dimensions>
    __host__ __device__ auto
    accumulate_pow2_diffs(const VectorLikeT& a, const VectorLikeU& b) const {
        if constexpr (D == 0) {
            return 0;
        } else {
            return (a[D - 1] - b[D - 1]) * (a[D - 1] - b[D - 1])
                 + accumulate_pow2_diffs<D - 1>(a, b);
        }
    }
};

template<uint Dimensions>
class euclidean_squared<Dimensions, void, void> {
  public:
    static constexpr bool is_loopless = true;
    static constexpr bool is_squared  = true;

    template<class VectorLikeT, class VectorLikeU>
    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint) const {
        return accumulate_pow2_diffs<Dimensions>(a, b);
    }

  private:
    template<uint D, class VectorLikeT, class VectorLikeU>
    __host__ __device__ auto
    accumulate_pow2_diffs(const VectorLikeT& a, const VectorLikeU& b) const {
        if constexpr (D == 0) {
            return 0;
        } else {
            return (a[D - 1] - b[D - 1]) * (a[D - 1] - b[D - 1])
                 + accumulate_pow2_diffs<D - 1>(a, b);
        }
    }
};

template<uint Dimensions, class VectorLikeT = void, class VectorLikeU = void>
class euclidean {
  public:
    static constexpr bool is_loopless = true;
    static constexpr bool is_squared  = false;

    __host__ __device__ auto
    operator()(const VectorLikeT& a, const VectorLikeU& b, uint) const {
        return math::sqrt(
            euclidean_squared<Dimensions, VectorLikeT, VectorLikeU>{}(a, b)
        );
    }
};

}  // namespace naga::distance_functions::loopless
