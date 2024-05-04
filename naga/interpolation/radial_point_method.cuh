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
#include "../distance_functions.hpp"
#include "../linalg/batched_matrix_inverse.cuh"
#include "../linalg/matrix.cuh"
#include "../point_map.cuh"
#include "detail/radial_point_method.cuh"
#include <scalix/algorithm/transform.cuh>
#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>

namespace naga::interpolation {

template<class PointType>
struct mq_shape_function {
    using value_type = typename point_traits<PointType>::value_type;
    constexpr static uint dimensions = point_traits<PointType>::dimensions;

    __host__ __device__ mq_shape_function(
        const value_type& q       = .5f,
        const value_type& alpha_c = .1f
    )
        : q_(q),
          alpha_c_(alpha_c) {}

    __host__ __device__ value_type operator()(
        const value_type& r_squared,
        const value_type& approx_particle_spacing
    ) const {
        value_type c = approx_particle_spacing * alpha_c_;
        return math::pow(r_squared + c * c, q_);
    }

    value_type q_;
    value_type alpha_c_;
};

template<class PointType>
struct exp_shape_function {
    using value_type = typename point_traits<PointType>::value_type;
    constexpr static uint dimensions = point_traits<PointType>::dimensions;

    __host__ __device__ exp_shape_function(
        const value_type& b = 0.03f
    )
        : b_(b) {}

    __host__ __device__ value_type operator()(
        const value_type& r_squared,
        const value_type& approx_particle_spacing
    ) const {
        return math::exp(b_ * r_squared);
    }

    value_type b_;
};

template<class PointMapType>
using default_shape_function
    = mq_shape_function<typename point_map_traits<PointMapType>::point_type>;

template<class T>
class radial_point_method {
  public:
    radial_point_method() = default;

    static size_t get_scratchpad_size(
        size_t query_size,
        uint support_size,
        uint dimensions,
        uint group_size = 1
    ) {
        if (query_size % group_size != 0) {
            sclx::throw_exception<std::invalid_argument>(
                "query_size must be a multiple of group_size",
                "naga::interpolation::radial_point_method::"
            );
        }
        size_t mem_per_group
            = detail::radial_point_method::get_scratchpad_size_per_group<T>(
                support_size,
                dimensions,
                group_size
            );
        return query_size / group_size * mem_per_group;
    }

    template<
        class PointMapType,
        class ShapeFunctionType = default_shape_function<PointMapType>>
    static sclx::array<T, 2> compute_weights(
        sclx::array<const T, 2> source_points,
        sclx::array<const std::uint32_t, 2> interpolating_indices,
        const PointMapType& query_points,
        const T& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        const std::vector<int>& devices         = {}
    ) {
        return detail::radial_point_method::compute_weights(
            source_points,
            interpolating_indices,
            query_points,
            approx_particle_spacing,
            group_size,
            shape_function,
            devices
        );
    }

    template<
        class PointMapType,
        class ShapeFunctionType = default_shape_function<PointMapType>>
    static radial_point_method<T> create_interpolator(
        sclx::array<const T, 2> source_points,
        sclx::array<std::uint32_t, 2> interpolating_indices,
        const PointMapType& query_points,
        const T& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        const std::vector<int>& devices         = {}
    ) {
        radial_point_method<T> interpolator{};
        auto weights = compute_weights(
            source_points,
            interpolating_indices,
            query_points,
            approx_particle_spacing,
            group_size,
            shape_function,
            devices
        );
        interpolator.weights_            = weights;
        interpolator.indices_            = interpolating_indices;
        interpolator.group_size_         = group_size;
        interpolator.source_points_size_ = source_points.shape()[1];
        return interpolator;
    }

    std::future<void> interpolate(
        sclx::array<const T, 2> field,
        sclx::array<T, 2>& destination,
        T centering_offset = T{0}
    ) const {
        if (field.shape()[1] != source_points_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "field array must have the same number of rows as the number "
                "of source points used to construct the interpolator"
            );
        }

        if (destination.shape()[1] != weights_.shape()[1]) {
            sclx::throw_exception<std::invalid_argument>(
                "destination array must have the same number of columns as the "
                "number of points used to construct the interpolator"
            );
        }

        auto weights    = weights_;
        auto indices    = indices_;
        auto group_size = group_size_;

        return sclx::execute_kernel([=](const sclx::kernel_handler& handler
                                    ) mutable {
            handler.launch(
                sclx::md_range_t<2>(destination.shape()),
                destination,
                [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                    T sum = 0;
                    for (uint i = 0; i < weights.shape()[0]; ++i) {
                        sum += weights(i, idx[1])
                             * (field(idx[0], indices(i, idx[1] / group_size))
                                - centering_offset);
                    }
                    destination(idx[0], idx[1]) = sum + centering_offset;
                }
            );
        });
    }

    std::future<void> interpolate(
        sclx::array<T, 1> field,
        sclx::array<T, 1> destination,
        T centering_offset = T{0}
    ) const {
        if (field.shape()[0] != source_points_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "field array must have the same number of rows as the number "
                "of source points used to construct the interpolator"
            );
        }

        if (destination.shape()[0] != weights_.shape()[1]) {
            sclx::throw_exception<std::invalid_argument>(
                "destination array must have the same number of columns as the "
                "number of points used to construct the interpolator"
            );
        }

        if (!cusparse_enabled_) {
            auto weights    = weights_;
            auto indices    = indices_;
            auto group_size = group_size_;

            return sclx::execute_kernel([=](const sclx::kernel_handler& handler
                                        ) mutable {
                handler.launch(
                    sclx::md_range_t<1>{destination.shape()[0]},
                    destination,
                    [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                        T sum = 0;
                        for (uint i = 0; i < weights.shape()[0]; ++i) {
                            sum += weights(i, idx[0])
                                 * (field(indices(i, idx[0] / group_size))
                                    - centering_offset);
                        }
                        destination(idx[0]) = sum + centering_offset;
                    }
                );
            });
        }

        if (centering_offset != T{0}) {
            sclx::fill(destination, centering_offset);
        }

        auto& A        = cusparse_desc_->A;
        auto& mat_mult = cusparse_desc_->mat_mult;
        vector_type X(field);
        vector_type Y(destination);
        auto fut = mat_mult(1.0, centering_offset == T{0} ? 0. : -1., A, X, Y);
        if (centering_offset == T{0}) {
            return fut;
        }

        return std::async(
            std::launch::async,
            [=, fut = std::move(fut)]() mutable {
                fut.get();
                auto Y_values = Y.values();
                sclx::algorithm::transform(
                    Y_values,
                    Y_values,
                    centering_offset,
                    thrust::plus<T>{}
                )
                    .get();
            }
        );
    }

    void enable_cusparse_algorithm() {
        if (sclx::cuda::traits::device_count() > 1) {
            std::cerr
                << "Warning: cusparse algorithm only supports single GPU, \n"
                   "but multiple GPUs are available. For any arrays \n"
                   "distributed across multiple devices, the algorithm will \n"
                   "perform less efficiently than the default algorithm which "
                   "\n"
                   "distributes the computation across all available "
                   "devices.\n";
        }

        if (cusparse_desc_ == nullptr) {
            if (group_size_ != 1) {
                sclx::throw_exception<std::invalid_argument>(
                    "cusparse algorithm only supports group_size = 1",
                    "naga::interpolation::radial_point_method::"
                );
            }
            cusparse_desc_    = std::make_shared<cusparse_algo_desc>();
            cusparse_desc_->A = matrix_type::create_from_index_stencil(
                source_points_size_,
                indices_,
                weights_
            );
            cusparse_enabled_ = true;
        }
    }

    void disable_cusparse_algorithm() {
        cusparse_desc_    = nullptr;
        cusparse_enabled_ = false;
    }

    template<class Archive>
    void save(Archive& ar) const {
        ar(group_size_, source_points_size_);
        sclx::serialize_array(ar, weights_);
        sclx::serialize_array(ar, indices_);
    }

    template<class Archive>
    void load(Archive& ar) {
        ar(group_size_, source_points_size_);
        sclx::deserialize_array(ar, weights_);
        sclx::array<std::uint32_t, 2> indices;
        sclx::deserialize_array(ar, indices);
        indices_ = indices;
    }

  private:
    // group size allows for groups of interpolated points to use the same
    // source points. This is useful, for example, when interpolating to
    // quadrature points in a meshless method as the quadrature points can
    // usually use the nearest neighbors of their associated, integrated point
    // as their source points.
    uint group_size_{};
    size_t source_points_size_{};
    sclx::array<T, 2> weights_{};
    sclx::array<std::uint32_t, 2> indices_{};

    using matrix_type
        = naga::linalg::matrix<T, naga::linalg::storage_type::sparse_csr>;
    using vector_type
        = naga::linalg::vector<T, naga::linalg::storage_type::dense>;
    struct cusparse_algo_desc {
        matrix_type A;
        naga::linalg::matrix_mult<matrix_type, vector_type, vector_type>
            mat_mult;
    };
    std::shared_ptr<cusparse_algo_desc> cusparse_desc_{nullptr};
    bool cusparse_enabled_{false};
};

}  // namespace naga::interpolation
