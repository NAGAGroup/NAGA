
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
#include "compatability.h"
#include <type_traits>
#include <ostream>

namespace naga {

template<class PointType>
struct point_traits {
    using value_type                   = typename PointType::value_type;
    static constexpr uint dimensions   = PointType::dimensions;
    static constexpr bool is_view_type = PointType::is_view_type;
};

template<class T, uint Dimensions>
class point_view_t {
  public:
    using value_type                   = T;
    static constexpr uint dimensions   = Dimensions;
    static constexpr bool is_view_type = true;

    NAGA_HOST NAGA_DEVICE point_view_t(T* data) : data_(data) {}

    NAGA_HOST NAGA_DEVICE T& operator[](uint index) const {
        return data_[index];
    }

    NAGA_HOST NAGA_DEVICE T& x() const { return data_[0]; }

    NAGA_HOST NAGA_DEVICE T& y() const { return data_[1]; }

    template<uint D = Dimensions>
    NAGA_HOST NAGA_DEVICE std::enable_if_t<D == 3 && D == Dimensions, T&>
    z() const {
        return data_[2];
    }

  private:
    T* data_;
};

template<class T, uint Dimensions>
class point_t {
  public:
    using value_type                   = T;
    static constexpr uint dimensions   = Dimensions;
    static constexpr bool is_view_type = false;

    point_t() = default;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "google-explicit-constructor"
    operator point_view_t<T, Dimensions>() {
        return point_view_t<T, Dimensions>(data_);
    }
#pragma clang diagnostic pop

    NAGA_HOST NAGA_DEVICE explicit point_t(const T (&x)[Dimensions]) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = x[i];
        }
    }

    NAGA_HOST NAGA_DEVICE point_t(const point_view_t<T, Dimensions>& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    NAGA_HOST NAGA_DEVICE point_t&
    operator=(const point_view_t<T, Dimensions>& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    NAGA_HOST NAGA_DEVICE point_t(point_view_t<T, Dimensions>&& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    NAGA_HOST NAGA_DEVICE point_t(point_t<T, Dimensions>&& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    NAGA_HOST NAGA_DEVICE point_t& operator=(point_t<T, Dimensions>&& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    NAGA_HOST NAGA_DEVICE point_t(const point_t<T, Dimensions>& other) {
        if (this == &other) {
            return;
        }
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    NAGA_HOST NAGA_DEVICE point_t& operator=(const point_t<T, Dimensions>& other
    ) {
        if (this == &other) {
            return *this;
        }
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    NAGA_HOST NAGA_DEVICE T& operator[](uint index) { return data_[index]; }

    NAGA_HOST NAGA_DEVICE const T& operator[](uint index) const {
        return data_[index];
    }

    NAGA_HOST NAGA_DEVICE T& x() { return data_[0]; }

    NAGA_HOST NAGA_DEVICE const T& x() const { return data_[0]; }

    NAGA_HOST NAGA_DEVICE T& y() { return data_[1]; }

    NAGA_HOST NAGA_DEVICE const T& y() const { return data_[1]; }

    template<uint D = Dimensions>
    NAGA_HOST NAGA_DEVICE std::enable_if_t<D == 3 && D == Dimensions, T&> z() {
        return data_[2];
    }

    template<uint D = Dimensions>
    NAGA_HOST NAGA_DEVICE std::enable_if_t<D == 3 && D == Dimensions, const T&>
    z() const {
        return data_[2];
    }

  private:
    T data_[Dimensions];
    static_assert(!std::is_const<T>::value, "point_t cannot have const T");
};

template <class T, uint Dimensions>
NAGA_HOST std::ostream& operator<<(std::ostream& os, const point_t<T, Dimensions>& p)  {
    os << "(";
    for (uint i = 0; i < Dimensions; ++i) {
        os << p[i];
        if (i != Dimensions - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

template <class T, uint Dimensions>
NAGA_HOST std::ostream& operator<<(std::ostream& os, const point_view_t<T, Dimensions>& p)  {
    os << "(";
    for (uint i = 0; i < Dimensions; ++i) {
        os << p[i];
        if (i != Dimensions - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}




template <class Point, class TranslationVector>
NAGA_HOST NAGA_DEVICE auto translate_point(const Point& p, const TranslationVector& translate) -> naga::point_t<decltype(p[0] + translate[0]), 3> {
    using value_type = decltype(p[0] + translate[0]);

    return naga::point_t<value_type, 3>{{p[0] + translate[0], p[1] + translate[1], p[2] + translate[2]}};
}

template <class Point, class RotationVector>
NAGA_HOST NAGA_DEVICE auto rotate_point(const Point& p, const RotationVector& rot) -> naga::point_t<decltype(p[0] * rot[0]), 3> {
    using value_type = decltype(p[0] * rot[0]);
    auto x = p[0];
    auto y = p[1];
    auto z = p[2];

    auto alpha = naga::math::pi<value_type> / 180.f * rot[0];
    auto beta  = naga::math::pi<value_type> / 180.f * rot[1];
    auto gamma = naga::math::pi<value_type> / 180.f * rot[2];

    constexpr auto& cos = naga::math::cos<value_type>;
    constexpr auto& sin = naga::math::sin<value_type>;

    value_type rotation_matrix[3][3]{
        {cos(beta) * cos(gamma),
         -cos(beta) * sin(gamma),
         sin(beta)},
        {cos(alpha) * sin(gamma)
             + sin(alpha) * sin(beta) * cos(gamma),
         cos(alpha) * cos(gamma)
             - sin(alpha) * sin(beta) * sin(gamma),
         -sin(alpha) * cos(beta)},
        {sin(alpha) * sin(gamma)
             - cos(alpha) * sin(beta) * cos(gamma),
         sin(alpha) * cos(gamma)
             + cos(alpha) * sin(beta) * sin(gamma),
         cos(alpha) * cos(beta)}
    };

    auto x_rotated = rotation_matrix[0][0] * x
                   + rotation_matrix[0][1] * y
                   + rotation_matrix[0][2] * z;
    auto y_rotated = rotation_matrix[1][0] * x
                   + rotation_matrix[1][1] * y
                   + rotation_matrix[1][2] * z;
    auto z_rotated = rotation_matrix[2][0] * x
                   + rotation_matrix[2][1] * y
                   + rotation_matrix[2][2] * z;

    return naga::point_t<value_type, 3>{{x_rotated, y_rotated, z_rotated}};
}

}  // namespace naga
