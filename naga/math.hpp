
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

#include "compatability.h"

namespace naga::math {

template<class T, class U>
NAGA_HOST NAGA_DEVICE auto min(const T& x, const U& y)
    -> decltype(x < y ? x : y) {
    return x < y ? x : y;
}

template<class T, class U>
NAGA_HOST NAGA_DEVICE auto max(const T& x, const U& y)
    -> decltype(x > y ? x : y) {
    return x > y ? x : y;
}

template<class T>
static constexpr T pi = static_cast<T>(M_PI);

template<class T>
NAGA_HOST NAGA_DEVICE T sin(const T& x) {
    return std::sin(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T cos(const T& x) {
    return std::cos(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T tan(const T& x) {
    return std::tan(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T asin(const T& x) {
    return std::asin(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T acos(const T& x) {
    return std::acos(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T atan(const T& x) {
    return std::atan(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T atan2(const T& y, const T& x) {
    return std::atan2(y, x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T abs(const T& x) {
    return std::abs(x);
}

template<class T>
NAGA_HOST NAGA_DEVICE T sqrt(const T& x) {
    return std::sqrt(x);
}

template<class T, class E>
NAGA_HOST NAGA_DEVICE auto pow(const T& x, const E& e)
    -> decltype(std::pow(x, e)) {
    return std::pow(x, e);
}

template<class T>
NAGA_HOST NAGA_DEVICE auto exp(const T& x) -> decltype(std::exp(x)) {
    return std::exp(x);
}

template<class VectorType>
NAGA_HOST NAGA_DEVICE auto norm_squared(const VectorType& v, uint dims) {
    using T = decltype(v[0]);
    T sum   = 0;
    for (int i = 0; i < dims; ++i) {
        sum += v[i] * v[i];
    }
    return sum;
}

template<class VectorType>
NAGA_HOST NAGA_DEVICE auto norm(const VectorType& v, uint dims) {
    return sqrt(norm_squared(v, dims));
}

template<class VectorTypeT, class VectorTypeU>
NAGA_HOST NAGA_DEVICE auto
dot(const VectorTypeT& v, const VectorTypeU& u, uint dims) {
    using T = decltype(v[0]);
    T sum   = 0;
    for (int i = 0; i < dims; ++i) {
        sum += v[i] * u[i];
    }
    return sum;
}

template<class VectorTypeR, class VectorTypeT>
struct cross_return_type {
    using type = VectorTypeR;
};

template<class VectorTypeT>
struct cross_return_type<void, VectorTypeT> {
    using type = VectorTypeT;
};

template<class VectorTypeR, class VectorTypeT, class VectorTypeU>
NAGA_HOST NAGA_DEVICE void
cross(VectorTypeR& result, const VectorTypeT& v, const VectorTypeU& u) {
    result[0] = v[1] * u[2] - v[2] * u[1];
    result[1] = v[2] * u[0] - v[0] * u[2];
    result[2] = v[0] * u[1] - v[1] * u[0];
}

template<class VectorTypeR = void, class VectorTypeT, class VectorTypeU>
NAGA_HOST NAGA_DEVICE typename cross_return_type<VectorTypeR, VectorTypeT>::type
cross(const VectorTypeT& v, const VectorTypeU& u) {
    constexpr bool valid_types = !std::is_same_v<VectorTypeR, void>
                              || std::is_same_v<VectorTypeT, VectorTypeU>;
    static_assert(
        valid_types,
        "Invalid types for cross product. If return type is void, the input "
        "types must be the same."
    );
    using return_type =
        typename cross_return_type<VectorTypeR, VectorTypeT>::type;
    return_type result;
    cross(result, v, u);
    return result;
}

template<class VectorType>
NAGA_HOST NAGA_DEVICE void normalize(VectorType& v, uint dims) {
    using T = decltype(v[0]);
    T norm  = math::norm(v, dims);
    for (int i = 0; i < dims; ++i) {
        v[i] /= norm;
    }
}

namespace loopless {

template<uint N, class T>
NAGA_HOST NAGA_DEVICE auto pow(const T& x) -> decltype(x * x) {
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

template<uint Dimensions, class VectorTypeT, class VectorTypeU>
NAGA_HOST NAGA_DEVICE auto dot(const VectorTypeT& v, const VectorTypeU& u) {
    using T = decltype(v[0]);
    if constexpr (Dimensions == 1) {
        return v[0] * u[0];
    } else {
        return v[Dimensions - 1] * u[Dimensions - 1]
             + dot<Dimensions - 1, VectorTypeT, VectorTypeU>(v, u);
    }
}

template<uint Dimensions, class VectorType>
NAGA_HOST NAGA_DEVICE auto norm_squared(const VectorType& v) {
    return dot<Dimensions, VectorType, VectorType>(v, v);
}

template<uint Dimensions, class VectorType>
NAGA_HOST NAGA_DEVICE auto norm(const VectorType& v) {
    return sqrt(norm_squared<Dimensions, VectorType>(v));
}

template<uint Dimensions, class VectorType, class NT>
NAGA_HOST NAGA_DEVICE void normalize(VectorType& v, const NT& norm) {
    if constexpr (Dimensions == 1) {
        v[0] /= norm;
    } else {
        v[Dimensions - 1] /= norm;
        normalize<Dimensions - 1, VectorType, NT>(v, norm);
    }
}

template<uint Dimensions, class VectorType>
NAGA_HOST NAGA_DEVICE void normalize(VectorType& v) {
    using T = std::decay_t<decltype(v[0])>;
    T norm  = math::loopless::norm<Dimensions, VectorType>(v);
    normalize<Dimensions, VectorType, T>(v, norm);
}

}  // namespace loopless

}  // namespace naga::math
