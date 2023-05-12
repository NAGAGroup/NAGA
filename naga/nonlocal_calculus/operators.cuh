
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

#include "../interpolation/radial_point_method.cuh"
#include "../segmentation/nearest_neighbors.cuh"
#include "detail/quadrature_points.cuh"
#include <scalix/algorithm/reduce.cuh>

namespace naga::nonlocal_calculus {

template<class T, uint Dimensions>
class operator_builder {
    static_assert(
        Dimensions == 2 || Dimensions == 3,
        "Only 2D and 3D operators are supported."
    );

  public:
    explicit operator_builder(const sclx::array<T, 2>& domain)
        : domain_(domain) {
        naga::segmentation::nd_cubic_segmentation<T, Dimensions>
            domain_segmentation(domain, detail::num_interp_support);

        {
            auto knn_result = naga::segmentation::batched_nearest_neighbors(
                detail::num_interp_support,
                default_point_map<T, Dimensions>{domain},
                domain_segmentation
            );

            auto distances_squared = std::get<0>(knn_result);
            interaction_radii_ = sclx::array<T, 1>{distances_squared.shape()[1]};
            detail::compute_interaction_radii(
                distances_squared,
                interaction_radii_
            );

            support_indices_ = std::get<1>(knn_result);
        }

        detail::quadrature_point_map<T, Dimensions> quadrature_points_map(
            domain_,
            interaction_radii_
        );

        auto min_particle_spacing_sum = sclx::algorithm::reduce(
            interaction_radii_,
            T(0),
            sclx::algorithm::plus<>{}
        );

        T approx_particle_spacing
            = min_particle_spacing_sum / (interaction_radii_.elements());

        quadrature_interpolating_weights_
            = interpolation::radial_point_method<>::compute_weights(
                domain_,
                support_indices_,
                quadrature_points_map,
                approx_particle_spacing,
                (Dimensions == 2) ? detail::num_quad_points_2d
                                  : detail::num_quad_points_3d
            );
    }

    void invalidate() {
        quadrature_interpolating_weights_ = sclx::array<T, 2>{};
        domain_                           = sclx::array<T, 2>{};
        support_indices_                  = sclx::array<sclx::index_t, 2>{};
        interaction_radii_                = sclx::array<T, 2>{};
    }

    [[nodiscard]] bool is_valid() const {
        return quadrature_interpolating_weights_.elements() > 0;
    }

    template<template<class, uint> class Operator>
    [[nodiscard]] Operator<T, Dimensions> create() const {
        return Operator<T, Dimensions>::create(
            domain_,
            support_indices_,
            quadrature_interpolating_weights_,
            interaction_radii_
        );
    }

  private:
    sclx::array<T, 2> domain_;
    sclx::array<sclx::index_t, 2> support_indices_;
    sclx::array<T, 2> quadrature_interpolating_weights_;
    sclx::array<T, 1> interaction_radii_;
};

}  // namespace naga::nonlocal_calculus
