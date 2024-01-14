
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

#include "../point.hpp"
#include "detail/nd_cubic_segmentation.cuh"
#include <scalix/algorithm/inclusive_scan.cuh>
#include <scalix/fill.cuh>

namespace naga::segmentation {

template<class T, uint Dimensions>
class partition_t;

template<class T, uint Dimensions>
class partition_iterator {
  public:
    using value_type        = point_view_t<const T, Dimensions>;
    using difference_type   = sclx::index_t;
    using iterator_category = std::random_access_iterator_tag;

    __host__ __device__ operator bool() const {
        return counter_ < partition_size_;
    }

    __host__ __device__ value_type operator*() const {
        return {&((*points_)(0, indices_[counter_]))};
    }

    __host__ __device__ value_type operator->() const {
        return {&((*points_)(0, indices_[counter_]))};
    }

    __host__ __device__ partition_iterator& operator++() {
        ++counter_;
        return *this;
    }

    __host__ __device__ partition_iterator operator++(int) {
        partition_iterator tmp(*this);
        ++counter_;
        return tmp;
    }

    __host__ __device__ partition_iterator& operator--() {
        --counter_;
        return *this;
    }

    __host__ __device__ partition_iterator operator--(int) {
        partition_iterator tmp(*this);
        --counter_;
        return tmp;
    }

    __host__ __device__ partition_iterator& operator+=(difference_type n) {
        counter_ += n;
        return *this;
    }

    __host__ __device__ partition_iterator operator+(difference_type n) const {
        partition_iterator tmp(*this);
        tmp += n;
        return tmp;
    }

    __host__ __device__ partition_iterator& operator-=(difference_type n) {
        counter_ -= n;
        return *this;
    }

    __host__ __device__ partition_iterator operator-(difference_type n) const {
        partition_iterator tmp(*this);
        tmp -= n;
        return tmp;
    }

    __host__ __device__ difference_type
    operator-(const partition_iterator& other) const {
        return counter_ - other.counter_;
    }

    __host__ __device__ bool operator==(const partition_iterator& other) const {
        return counter_ == other.counter_;
    }

    __host__ __device__ bool operator!=(const partition_iterator& other) const {
        return counter_ != other.counter_;
    }

    __host__ __device__ bool operator<(const partition_iterator& other) const {
        return counter_ < other.counter_;
    }

    __host__ __device__ bool operator>(const partition_iterator& other) const {
        return counter_ > other.counter_;
    }

    __host__ __device__ bool operator<=(const partition_iterator& other) const {
        return counter_ <= other.counter_;
    }

    __host__ __device__ bool operator>=(const partition_iterator& other) const {
        return counter_ >= other.counter_;
    }

    __host__ __device__ value_type operator[](difference_type n) const {
        return {&((*points_)(0, indices_[counter_ + n]))};
    }

    friend class partition_t<T, Dimensions>;

  private:
    const sclx::array<const T, 2>* points_;
    const size_t* indices_;
    uint partition_size_;

    uint counter_ = 0;

    __host__ __device__ partition_iterator(
        const sclx::array<const T, 2>* points,
        const size_t* indices,
        const uint& partition_size
    )
        : points_(points),
          indices_(indices),
          partition_size_(partition_size) {}
};

template<class T, uint Dimensions>
class nd_cubic_segmentation;

template<class T, uint Dimensions>
class partition_t {
  public:
    using iterator = partition_iterator<T, Dimensions>;

    __host__ __device__ typename iterator::value_type operator[](uint index
    ) const {
        return &((*points_)(0, indices_[index]));
    }

    __host__ __device__ const size_t& point_dimensions() const {
        return points_->shape()[0];
    }

    __host__ __device__ iterator begin() const {
        return {points_, indices_, partition_size_};
    }

    __host__ __device__ iterator end() const {
        return iterator(points_, indices_, partition_size_)
             + static_cast<typename iterator::difference_type>(partition_size_);
    }

    __host__ __device__ uint size() const { return partition_size_; }

    __host__ __device__ bool empty() const { return partition_size_ == 0; }

    __host__ __device__ const size_t* indices() const { return indices_; }

    friend class nd_cubic_segmentation<T, Dimensions>;

  private:
    const sclx::array<const T, 2>* points_;
    const size_t* indices_;
    uint partition_size_;

    __host__ __device__ partition_t(
        const sclx::array<const T, 2>* points,
        const size_t* indices,
        const uint& partition_size
    )
        : points_(points),
          indices_(indices),
          partition_size_(partition_size) {}
};

template<class T, uint Dimensions>
class nd_cubic_segmentation_iterator;

template<class T, uint Dimensions>
__host__ __device__ nd_cubic_segmentation_iterator<T, Dimensions>
make_nd_cubic_segmentation_iterator(
    const nd_cubic_segmentation<T, Dimensions>* segmentation,
    const sclx::md_index_t<Dimensions>& index
);

template<class T, uint Dimensions>
class nd_cubic_segmentation {
  public:
    using partition_type = partition_t<T, Dimensions>;
    using iterator       = nd_cubic_segmentation_iterator<T, Dimensions>;

    nd_cubic_segmentation() = default;

    __host__ nd_cubic_segmentation(
        const sclx::array<const T, 2>& points,
        uint approx_partition_size
    )
        : points_(points) {
        if (points_.shape()[0] != Dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the points array does not match "
                "the number of dimensions in the segmentation",
                "naga::"
            );
        }

        // compute bounds

        point_view_t<T, Dimensions> lower_bounds_view(lower_bounds_);
        point_view_t<T, Dimensions> upper_bounds_view(upper_bounds_);
        detail::compute_bounds<T, Dimensions>(
            lower_bounds_view,
            upper_bounds_view,
            points_
        );

        T tol = 1e-3;
        for (uint i = 0; i < Dimensions; ++i) {
            if (upper_bounds_[i] - lower_bounds_[i] < tol) {
                upper_bounds_[i] = lower_bounds_[i] + tol;
            }
        }

        // Calculate the partition size
        T volume = 1.0f;
        for (uint i = 0; i < Dimensions; ++i) {
            volume *= upper_bounds_[i] - lower_bounds_[i];
        }

        partition_width_ = std::pow(
            static_cast<T>(approx_partition_size)
                / static_cast<T>(points_.shape()[1]) * volume,
            1.0f / static_cast<T>(Dimensions)
        );

        sclx::shape_t<Dimensions> segmentation_shape{};
        for (uint i = 0; i < Dimensions; ++i) {
            segmentation_shape[i] = static_cast<uint>(std::ceil(
                (upper_bounds_[i] - lower_bounds_[i]) / partition_width_
            ));
        }

        for (uint i = 0; i < Dimensions; ++i) {
            upper_bounds_[i]
                = lower_bounds_[i] + segmentation_shape[i] * partition_width_;
        }

        // Calculate partition sizes
        detail::compute_partition_sizes<T, Dimensions>(
            partition_sizes_,
            segmentation_shape,
            points_,
            *this
        );

        // Calculate the offsets
        detail::compute_index_offsets<Dimensions>(
            partition_index_offsets_,
            partition_sizes_
        );

        // Calculate the indices
        detail::assign_indices<T, Dimensions>(
            indices_,
            partition_sizes_,
            partition_index_offsets_,
            points_,
            *this
        );
    }

    __host__ nd_cubic_segmentation(
        const sclx::array<const T, 2>& points,
        T partition_width
    )
        : points_(points), partition_width_(partition_width) {
        static_assert(!std::is_integral_v<T>);
        if (points_.shape()[0] != Dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "The number of dimensions in the points array does not match "
                "the number of dimensions in the segmentation",
                "naga::"
            );
        }

        // compute bounds
        point_view_t<T, Dimensions> lower_bounds_view(lower_bounds_);
        point_view_t<T, Dimensions> upper_bounds_view(upper_bounds_);
        detail::compute_bounds<T, Dimensions>(
            lower_bounds_view,
            upper_bounds_view,
            points_
        );

        T tol = 1e-3;
        for (uint i = 0; i < Dimensions; ++i) {
            if (upper_bounds_[i] - lower_bounds_[i] < tol) {
                upper_bounds_[i] = lower_bounds_[i] + tol;
            }
        }

        // Calculate the partition size
        T volume = 1.0f;
        for (uint i = 0; i < Dimensions; ++i) {
            volume *= upper_bounds_[i] - lower_bounds_[i];
        }

        sclx::shape_t<Dimensions> segmentation_shape{};
        for (uint i = 0; i < Dimensions; ++i) {
            segmentation_shape[i] = static_cast<uint>(std::ceil(
                (upper_bounds_[i] - lower_bounds_[i]) / partition_width_
            ));
        }

        for (uint i = 0; i < Dimensions; ++i) {
            upper_bounds_[i]
                = lower_bounds_[i] + segmentation_shape[i] * partition_width_;
        }

        // Calculate partition sizes
        detail::compute_partition_sizes<T, Dimensions>(
            partition_sizes_,
            segmentation_shape,
            points_,
            *this
        );

        // Calculate the offsets
        detail::compute_index_offsets<Dimensions>(
            partition_index_offsets_,
            partition_sizes_
        );

        // Calculate the indices
        detail::assign_indices<T, Dimensions>(
            indices_,
            partition_sizes_,
            partition_index_offsets_,
            points_,
            *this
        );
    }

    template<class PointType>
    __host__ __device__ sclx::md_index_t<Dimensions>
    get_partition_index(const PointType& point) const {
        sclx::md_index_t<Dimensions> index;
        for (uint i = 0; i < Dimensions; ++i) {
            if (point[i] < lower_bounds_[i]) {
                index[i] = 0;
            } else if (point[i] >= upper_bounds_[i]) {
                index[i] = shape()[i] - 1;
            } else {
                index[i] = static_cast<uint>(
                    std::floor((point[i] - lower_bounds_[i]) / partition_width_)
                );
            }
        }
        return index;
    }

    __host__ __device__ partition_type
    get_partition(const sclx::md_index_t<Dimensions>& index) const {
        return {
            &points_,
            &indices_(partition_index_offsets_[index]),
            partition_sizes_[index]};
    }

    __host__ __device__ partition_type get_partition(sclx::index_t index
    ) const {
        return {
            &points_,
            &indices_(partition_index_offsets_[index]),
            partition_sizes_[index]};
    }

    template<class PointType>
    __host__ __device__ partition_type get_partition(const PointType& point
    ) const {
        return get_partition(get_partition_index(point));
    }

    __host__ __device__ const sclx::shape_t<Dimensions>& shape() const {
        return partition_sizes_.shape();
    }

    __host__ __device__ size_t partition_count() const {
        return partition_sizes_.elements();
    }

    __host__ __device__ size_t point_count() const {
        return points_.shape()[1];
    }

    __host__ __device__ iterator begin() const {
        return make_nd_cubic_segmentation_iterator(
            this,
            sclx::md_index_t<Dimensions>{}
        );
    }

    __host__ __device__ iterator end() const {
        return make_nd_cubic_segmentation_iterator(
            this,
            sclx::md_index_t<Dimensions>::create_from_linear(
                partition_count(),
                shape()
            )
        );
    }

    sclx::array<const T, 2> points() const {
        return points_;
    }

    friend class nd_cubic_segmentation_iterator<T, Dimensions>;

  private:
    sclx::array<uint, Dimensions> partition_sizes_;
    sclx::array<size_t, Dimensions> partition_index_offsets_;
    sclx::array<const T, 2> points_;
    sclx::array<size_t, 1> indices_;

    T lower_bounds_[Dimensions];
    T upper_bounds_[Dimensions];
    T partition_width_;
};

template<class T, uint Dimensions>
class nd_cubic_segmentation_iterator {
  public:
    using segmentation_type = nd_cubic_segmentation<T, Dimensions>;
    using partition_type    = typename segmentation_type::partition_type;

    using value_type        = partition_type;
    using difference_type   = sclx::index_t;
    using iterator_category = std::random_access_iterator_tag;

    __host__ __device__ operator bool() const {
        return counter_ < segmentation_->partition_count();
    }

    __host__ __device__ partition_type operator*() const {
        return segmentation_->get_partition(counter_);
    }

    __host__ __device__ partition_type operator->() const {
        return segmentation_->get_partition(counter_);
    }

    __host__ __device__ nd_cubic_segmentation_iterator& operator++() {
        counter_++;
        return *this;
    }

    __host__ __device__ nd_cubic_segmentation_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    __host__ __device__ bool
    operator==(const nd_cubic_segmentation_iterator& other) const {
        return counter_ == other.counter_;
    }

    __host__ __device__ bool
    operator!=(const nd_cubic_segmentation_iterator& other) const {
        return counter_ != other.counter_;
    }

    __host__ __device__ bool
    operator<(const nd_cubic_segmentation_iterator& other) const {
        return counter_ < other.counter_;
    }

    __host__ __device__ bool
    operator<=(const nd_cubic_segmentation_iterator& other) const {
        return counter_ <= other.counter_;
    }

    __host__ __device__ bool
    operator>(const nd_cubic_segmentation_iterator& other) const {
        return counter_ > other.counter_;
    }

    __host__ __device__ bool
    operator>=(const nd_cubic_segmentation_iterator& other) const {
        return counter_ >= other.counter_;
    }

    __host__ __device__ difference_type
    operator-(const nd_cubic_segmentation_iterator& other) const {
        return counter_ - other.counter_;
    }

    __host__ __device__ nd_cubic_segmentation_iterator
    operator+(difference_type n) const {
        return nd_cubic_segmentation_iterator(segmentation_, counter_ + n);
    }

    __host__ __device__ nd_cubic_segmentation_iterator
    operator-(difference_type n) const {
        return nd_cubic_segmentation_iterator(segmentation_, counter_ - n);
    }

    __host__ __device__ nd_cubic_segmentation_iterator&
    operator+=(difference_type n) {
        counter_ += n;
        return *this;
    }

    __host__ __device__ nd_cubic_segmentation_iterator&
    operator-=(difference_type n) {
        counter_ -= n;
        return *this;
    }

    __host__ __device__ nd_cubic_segmentation_iterator(
        const segmentation_type* segmentation,
        size_t counter
    )
        : segmentation_(segmentation),
          counter_(counter) {}

    __host__ __device__ partition_t<T, Dimensions> operator[](difference_type n
    ) const {
        return segmentation_->get_partition(n);
    }

  private:
    const segmentation_type* segmentation_;

    size_t counter_;
};

template<class T, uint Dimensions>
__host__ __device__ nd_cubic_segmentation_iterator<T, Dimensions>
make_nd_cubic_segmentation_iterator(
    const nd_cubic_segmentation<T, Dimensions>* segmentation,
    const sclx::md_index_t<Dimensions>& index
) {
    return nd_cubic_segmentation_iterator<T, Dimensions>(
        segmentation,
        index.as_linear(segmentation->shape())
    );
}

}  // namespace naga::segmentation

#include "detail/nd_cubic_segmentation.inl"
