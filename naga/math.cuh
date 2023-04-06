
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

template<class T>
__host__ __device__ T abs(T&& x) {
    return std::abs(x);
}

template<class T>
__host__ __device__ T sqrt(T&& x) {
    return std::sqrt(x);
}

template<class T, class U>
__host__ __device__ auto pow(T&& x, U&& y) -> decltype(std::pow(x, y)) {
    return std::pow(x, y);
}

namespace loopless {

template<uint N, class T>
__host__ __device__ auto pow(T&& x) -> decltype(x * x) {
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