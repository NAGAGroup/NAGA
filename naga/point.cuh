
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

namespace naga {

template<class FloatingPointType, uint Dimensions>
class point_view_t {
  public:
    __host__ __device__ point_view_t(FloatingPointType* data) : data_(data) {}

    __host__ __device__ FloatingPointType& operator[](uint index) const {
        return data_[index];
    }

    __host__ __device__ FloatingPointType& x() const { return data_[0]; }

    __host__ __device__ FloatingPointType& y() const { return data_[1]; }

    template<uint D = Dimensions>
    __host__ __device__
        std::enable_if_t<D == 3 && D == Dimensions, FloatingPointType&>
        z() const {
        return data_[2];
    }

  private:
    FloatingPointType* data_;
};

template<class FloatingPointType, uint Dimensions>
class point_t {
  public:
    point_t() = default;

    __host__ __device__
    point_t(const point_view_t<FloatingPointType, Dimensions>& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    __host__ __device__ point_t&
    operator=(const point_view_t<FloatingPointType, Dimensions>& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    __host__ __device__
    point_t(point_view_t<FloatingPointType, Dimensions>&& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    __host__ __device__ point_t(point_t<FloatingPointType, Dimensions>&& other
    ) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    __host__ __device__ point_t&
    operator=(point_t<FloatingPointType, Dimensions>&& other) {
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    __host__ __device__
    point_t(const point_t<FloatingPointType, Dimensions>& other) {
        if (this == &other) {
            return;
        }
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
    }

    __host__ __device__ point_t&
    operator=(const point_t<FloatingPointType, Dimensions>& other) {
        if (this == &other) {
            return *this;
        }
        for (uint i = 0; i < Dimensions; ++i) {
            data_[i] = other[i];
        }
        return *this;
    }

    __host__ __device__ FloatingPointType& operator[](uint index) {
        return data_[index];
    }

    __host__ __device__ const FloatingPointType& operator[](uint index) const {
        return data_[index];
    }

    __host__ __device__ FloatingPointType& x() { return data_[0]; }

    __host__ __device__ const FloatingPointType& x() const { return data_[0]; }

    __host__ __device__ FloatingPointType& y() { return data_[1]; }

    __host__ __device__ const FloatingPointType& y() const { return data_[1]; }

    template<uint D = Dimensions>
    __host__ __device__
        std::enable_if_t<D == 3 && D == Dimensions, FloatingPointType&>
        z() {
        return data_[2];
    }

    template<uint D = Dimensions>
    __host__ __device__
        std::enable_if_t<D == 3 && D == Dimensions, const FloatingPointType&>
        z() const {
        return data_[2];
    }

  private:
    FloatingPointType data_[Dimensions];
    static_assert(
        !std::is_const<FloatingPointType>::value,
        "point_t cannot have const FloatingPointType"
    );
};

}  // namespace naga
