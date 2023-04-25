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
#include "../distance_functions.cuh"
#include "../linalg/batched_matrix_inverse.cuh"
#include "../point_map.cuh"
#include "detail/radial_point_method.cuh"
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

template<class PointMapType>
using default_shape_function
    = mq_shape_function<typename point_map_traits<PointMapType>::point_type>;

template<class T = float>
class radial_point_method {
  public:
    template<
        class T_,
        class PointMapType,
        class ShapeFunctionType = default_shape_function<PointMapType>>
    static sclx::array<T_, 2> compute_weights(
        sclx::array<T_, 2>& source_points,
        sclx::array<size_t, 2>& interpolating_indices,
        sclx::array<T_, 2>& source2query_dist_squared,
        const PointMapType& query_points,
        const T& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        const std::vector<int>& devices         = {}
    ) {
        return detail::radial_point_method::compute_weights(
            source_points,
            interpolating_indices,
            source2query_dist_squared,
            query_points,
            approx_particle_spacing,
            group_size,
            shape_function,
            devices
        );
    }

    template<
        class T_,
        class PointMapType,
        class ShapeFunctionType = default_shape_function<PointMapType>>
    static radial_point_method<T_> create_interpolator(
        sclx::array<T_, 2>& source_points,
        sclx::array<size_t, 2>& interpolating_indices,
        sclx::array<T_, 2>& source2query_dist_squared,
        const PointMapType& query_points,
        const T& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        const std::vector<int>& devices         = {}
    ) {
        radial_point_method<T_> interpolator{};
        auto weights = compute_weights(
            source_points,
            interpolating_indices,
            source2query_dist_squared,
            query_points,
            approx_particle_spacing,
            group_size,
            shape_function,
            devices
        );
        interpolator.weights_                 = weights;
        interpolator.indices_                 = interpolating_indices;
        interpolator.group_size_              = group_size;
        interpolator.source_points_size_      = source_points.shape()[1];
        interpolator.approx_particle_spacing_ = approx_particle_spacing;
        return interpolator;
    }

    void interpolate(sclx::array<T, 2> field, sclx::array<T, 2> destination) {
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

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<2>(destination.shape()),
                destination,
                [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                    T sum = 0;
                    for (uint i = 0; i < weights.shape()[0]; ++i) {
                        sum += weights(i, idx[1])
                             * field(idx[0], indices(i, idx[1]));
                    }
                    destination(idx[0], idx[1]) = sum;
                }
            );
        });
    }

    void interpolate(sclx::array<T, 1> field, sclx::array<T, 1> destination) {
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

        auto weights    = weights_;
        auto indices    = indices_;
        auto group_size = group_size_;

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{destination.shape()[0]},
                destination,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    T sum = 0;
                    for (uint i = 0; i < weights.shape()[0]; ++i) {
                        sum += weights(i, idx[0])
                             * field(indices(i, idx[0] / group_size));
                    }
                    destination(idx[0]) = sum;
                }
            );
        }).get();
    }

    template<class T_>
    friend class radial_point_method;

  private:
    radial_point_method() = default;

    // group size allows for groups of interpolated points to use the same
    // source points. This is useful, for example, when interpolating to
    // quadrature points in a meshless method as the quadrature points can
    // usually use the nearest neighbors of their associated, integrated point
    // as their source points.
    uint group_size_{};
    size_t source_points_size_{};
    sclx::array<T, 2> weights_{};
    sclx::array<size_t, 2> indices_{};
    T approx_particle_spacing_{};
};

}  // namespace naga::interpolation
