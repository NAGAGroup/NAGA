
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
#include <cmath>

namespace naga::math {

template<class T, class U>
__host__ __device__ auto min(const T& x, const U& y)
    -> decltype(x < y ? x : y) {
    return x < y ? x : y;
}

template<class T, class U>
__host__ __device__ auto max(const T& x, const U& y)
    -> decltype(x > y ? x : y) {
    return x > y ? x : y;
}

template<class T>
static constexpr T pi = static_cast<T>(M_PI);

template<class T>
__host__ __device__ T sin(const T& x) {
    return std::sin(x);
}

template<class T>
__host__ __device__ T cos(const T& x) {
    return std::cos(x);
}

template<class T>
__host__ __device__ T tan(const T& x) {
    return std::tan(x);
}

template<class T>
__host__ __device__ T abs(const T& x) {
    return std::abs(x);
}

template<class T>
__host__ __device__ T sqrt(const T& x) {
    return std::sqrt(x);
}

template<class T, class E>
__host__ __device__ auto pow(const T& x, const E& e)
    -> decltype(std::pow(x, e)) {
    return std::pow(x, e);
}

template <class T>
__host__ __device__ auto exp(const T& x) -> decltype(std::exp(x)) {
    return std::exp(x);
}

namespace loopless {

template<uint N, class T>
__host__ __device__ auto pow(const T& x) -> decltype(x * x) {
    if constexpr (N == 0) {
        return 1;
    } else if constexpr (N == 1) {
        return x;
    } else if constexpr (N % 2 == 0) {
        auto y = pow<N / 2>(x);
        return y * y;
    } else {
        auto y = pow<N / 2>(x);
        return y * y * x;
    }
}

}  // namespace loopless

}  // namespace naga::math
