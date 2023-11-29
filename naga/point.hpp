
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

}  // namespace naga
