
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

#include "../math.cuh"
#include "../point.cuh"
#include "scalix/index.cuh"
#include <scalix/throw_exception.hpp>

namespace naga::ranges {

template<class T, uint Dimensions>
class uniform_grid;

template<class T, uint Dimensions>
struct uniform_grid_inspector;

template<class T, uint Dimensions>
class uniform_grid_iterator {
  public:
    using value_type        = point_t<T, Dimensions>;
    using difference_type   = size_t;
    using iterator_category = std::random_access_iterator_tag;

    __host__ __device__ uniform_grid_iterator(
        const point_t<T, Dimensions>& min,
        const point_t<T, Dimensions>& max,
        const T& grid_spacing
    )
        : min_(min) {
        for (uint i = 0; i < Dimensions; ++i) {
            grid_size_[i]    = std::ceil((max[i] - min[i]) / grid_spacing) + 1;
            grid_spacing_[i] = (max[i] - min[i]) / (grid_size_[i] - 1);
        }
    }

    __host__ __device__ uniform_grid_iterator(
        const point_t<T, Dimensions>& min,
        const point_t<T, Dimensions>& max,
        const point_t<T, Dimensions>& grid_spacing
    )
        : min_(min) {
        for (uint i = 0; i < Dimensions; ++i) {
            grid_size_[i] = std::ceil((max[i] - min[i]) / grid_spacing[i]) + 1;
            grid_spacing_[i] = (max[i] - min[i]) / (grid_size_[i] - 1);
        }
    }

    uniform_grid_iterator(const uniform_grid_iterator& other) = default;

    uniform_grid_iterator(uniform_grid_iterator&& other) noexcept = default;

    uniform_grid_iterator& operator=(const uniform_grid_iterator& other)
        = default;

    uniform_grid_iterator& operator=(uniform_grid_iterator&& other) noexcept
        = default;

    __host__ __device__ value_type operator*() const {
        value_type result;
        size_t index = current_index_;
        for (uint i = 0; i < Dimensions; ++i) {
            result[i] = min_[i] + (index % grid_size_[i]) * grid_spacing_[i];
            index /= grid_size_[i];
        }
        return result;
    }

    __host__ __device__ value_type operator[](const sclx::index_t& n) const {
        return *(*this + n);
    }

    __host__ __device__ uniform_grid_iterator& operator++() {
        ++current_index_;
        return *this;
    }

    __host__ __device__ uniform_grid_iterator operator++(int) {
        uniform_grid_iterator result = *this;
        ++current_index_;
        return result;
    }

    __host__ __device__ uniform_grid_iterator& operator--() {
        --current_index_;
        return *this;
    }

    __host__ __device__ uniform_grid_iterator operator--(int) {
        uniform_grid_iterator result = *this;
        --current_index_;
        return result;
    }

    __host__ __device__ uniform_grid_iterator&
    operator+=(const difference_type& n) {
        current_index_ += n;
        return *this;
    }

    __host__ __device__ uniform_grid_iterator&
    operator-=(const difference_type& n) {
        current_index_ -= n;
        return *this;
    }

    __host__ __device__ uniform_grid_iterator operator+(const difference_type& n
    ) const {
        uniform_grid_iterator result = *this;
        result += n;
        return result;
    }

    __host__ __device__ uniform_grid_iterator operator-(const difference_type& n
    ) const {
        uniform_grid_iterator result = *this;
        result -= n;
        return result;
    }

    __host__ __device__ difference_type
    operator-(const uniform_grid_iterator& other) const {
        return current_index_ - other.current_index_;
    }

    __host__ __device__ bool operator==(const uniform_grid_iterator& other
    ) const {
        return current_index_ == other.current_index_;
    }

    __host__ __device__ bool operator!=(const uniform_grid_iterator& other
    ) const {
        return current_index_ != other.current_index_;
    }

    __host__ __device__ bool operator<(const uniform_grid_iterator& other
    ) const {
        return current_index_ < other.current_index_;
    }

    __host__ __device__ bool operator<=(const uniform_grid_iterator& other
    ) const {
        return current_index_ <= other.current_index_;
    }

    __host__ __device__ bool operator>(const uniform_grid_iterator& other
    ) const {
        return current_index_ > other.current_index_;
    }

    __host__ __device__ bool operator>=(const uniform_grid_iterator& other
    ) const {
        return current_index_ >= other.current_index_;
    }

  private:
    friend class uniform_grid<T, Dimensions>;
    friend struct uniform_grid_inspector<T, Dimensions>;

    size_t current_index_{0};
    size_t grid_size_[Dimensions];
    T grid_spacing_[Dimensions];
    point_t<T, Dimensions> min_;
};

template<class T, uint Dimensions>
class uniform_grid {
  public:
    using iterator   = uniform_grid_iterator<T, Dimensions>;
    using value_type = typename iterator::value_type;

    __host__ __device__ uniform_grid(
        const point_t<T, Dimensions>& min,
        const point_t<T, Dimensions>& max,
        const T& grid_spacing
    )
        : begin_(min, max, grid_spacing) {}

    __host__ __device__ uniform_grid(
        const point_t<T, Dimensions>& min,
        const point_t<T, Dimensions>& max,
        const point_t<T, Dimensions>& grid_spacing
    )
        : begin_(min, max, grid_spacing) {}

    __host__ __device__ iterator begin() const { return begin_; }

    __host__ __device__ iterator end() const { return begin_ + size(); }

    __host__ __device__ size_t size() const {
        size_t result = 1;
        for (uint i = 0; i < Dimensions; ++i) {
            result *= begin_.grid_size_[i];
        }
        return result;
    }

    __host__ __device__ value_type operator[](const sclx::index_t& n) const {
        return begin_[n];
    }

  private:
    friend struct uniform_grid_inspector<T, Dimensions>;

    iterator begin_;
};

template<class T, uint Dimensions>
struct uniform_grid_inspector {
    using grid_t = uniform_grid<T, Dimensions>;

    __host__ __device__ static bool
    is_boundary_index(const grid_t& grid, sclx::index_t n) {
        for (uint i = 0; i < Dimensions; ++i) {
            if (n % grid.begin_.grid_size_[i] == 0
                || n % grid.begin_.grid_size_[i]
                       == grid.begin_.grid_size_[i] - 1) {
                return true;
            }
            n /= grid.begin_.grid_size_[i];
        }
        return false;
    }

    __host__ __device__ static typename grid_t::value_type
    get_boundary_normal(const grid_t& grid, sclx::index_t boundary_index) {
        if (!is_boundary_index(grid, boundary_index)) {
#ifdef __CUDA_ARCH__
            printf("get_boundary_normal called with non-boundary index\n");
            return typename grid_t::value_type{};
#else
            sclx::throw_exception<std::invalid_argument>(
                "get_boundary_normal called with non-boundary index",
                "naga::ranges::uniform_grid::"
            );
#endif
        }

        size_t dim_indices[Dimensions];
        for (uint i = 0; i < Dimensions; ++i) {
            dim_indices[i] = boundary_index % grid.begin_.grid_size_[i];
            boundary_index /= grid.begin_.grid_size_[i];
        }

        typename grid_t::value_type normal;

        for (uint i = 0; i < Dimensions; ++i) {
            if (dim_indices[i] == 0) {
                normal[i] = -1;
            } else if (dim_indices[i] == grid.begin_.grid_size_[i] - 1) {
                normal[i] = 1;
            } else {
                normal[i] = 0;
            }
        }

        auto norm = math::loopless::norm<Dimensions>(normal);
        for (uint i = 0; i < Dimensions; ++i) {
            normal[i] /= (norm == 0 ? 1 : norm);
        }

        return normal;
    }
};

}  // namespace naga::ranges
