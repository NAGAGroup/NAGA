
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

#include "../../../segmentation/nearest_neighbors.cuh"
#include "../node_provider.cuh"
#include "detail/uniform_grid_provider.cuh"
#include <scalix/algorithm/count_if.cuh>
#include <scalix/iota.cuh>

namespace naga::fluids::nonlocal_lbm {

template<class Lattice>
class uniform_grid_provider : public node_provider<Lattice> {
  public:
    using base                       = node_provider<Lattice>;
    static constexpr uint dimensions = base::dimensions;
    using value_type                 = typename base::value_type;
    using range_type = ranges::uniform_grid<value_type, dimensions>;

    __host__ uniform_grid_provider(
        const point_t<value_type, dimensions>& min,
        const point_t<value_type, dimensions>& max,
        const value_type& grid_spacing,
        const value_type& absorption_layer_thickness = 0,
        const value_type& peak_absorption_rate       = 0
    )
        : grid_range_(min, max, grid_spacing),
          grid_spacing_(grid_spacing),
          absorption_layer_thickness_(absorption_layer_thickness),
          peak_absorption_rate_(peak_absorption_rate) {}

    __host__ simulation_nodes<value_type> get() const final {
        simulation_nodes<value_type> result{};

        result.nodal_spacing = grid_spacing_;

        result.points
            = sclx::array<value_type, 2>{dimensions, grid_range_.size()};
        sclx::array<value_type, 2> normals{dimensions, grid_range_.size()};

        size_t boundary_size = count_boundary_points();

        assign_boundary_info(
            result.points,
            normals,
            grid_range_.size() - boundary_size
        );

        result.boundary_normals = sclx::array<value_type, 2>(
            {dimensions, boundary_size},
            &normals(0, grid_range_.size() - boundary_size),
            true,
            sclx::data_capture_mode::copy,
            sclx::copy_policy::devicedevice
        );
        sclx::algorithm::transform(
            result.boundary_normals,
            result.boundary_normals,
            -1.f,
            sclx::algorithm::multiplies<>{}
        );

        assign_bulk_info(result.points);

        if (absorption_layer_thickness_ == 0) {
            result.num_boundary_points = boundary_size;
            result.num_bulk_points     = grid_range_.size() - boundary_size;
            return result;
        }

        auto bulk_slice = result.points.get_range(
            {0},
            {grid_range_.size() - boundary_size}
        );
        auto boundary_slice = result.points.get_range(
            {grid_range_.size() - boundary_size},
            {grid_range_.size()}
        );
        using segmentation_t
            = segmentation::nd_cubic_segmentation<value_type, dimensions>;
        using point_map_t = default_point_map<value_type, dimensions>;

        segmentation_t boundary_segmentation(
            boundary_slice,
            16
        );  // 16 seems like a good number
        auto [boundary_distances_squared, indices]
            = segmentation::batched_nearest_neighbors(
                1,
                point_map_t{bulk_slice},
                boundary_segmentation
            );
        size_t num_layer_points
            = count_layer_points(boundary_distances_squared);

        sclx::array<value_type, 2> bulk_points_new(
            {dimensions, grid_range_.size() - boundary_size}
        );
        sclx::array<value_type, 1> absorption_rates{
            grid_range_.size() - boundary_size};

        size_t layer_points_offset
            = grid_range_.size() - boundary_size - num_layer_points;

        assign_bulk_and_layer_info(
            boundary_distances_squared,
            layer_points_offset,
            bulk_slice,
            bulk_points_new,
            absorption_rates
        );

        sclx::assign_array(bulk_points_new, bulk_slice);

        result.layer_absorption = sclx::array<value_type, 1>(
            {num_layer_points},
            &absorption_rates(layer_points_offset),
            true,
            sclx::data_capture_mode::copy,
            sclx::copy_policy::devicedevice
        );

        result.num_bulk_points     = layer_points_offset;
        result.num_layer_points    = num_layer_points;
        result.num_boundary_points = boundary_size;

        return result;
    }

  private:
    range_type grid_range_;
    value_type grid_spacing_;
    value_type absorption_layer_thickness_;
    value_type peak_absorption_rate_;

    __host__ size_t count_boundary_points() const {
        sclx::array<size_t, 1> iota_range{grid_range_.size()};
        sclx::iota(iota_range, 0);
        return sclx::algorithm::count_if(
            iota_range,
            detail::boundary_count_criteria_functor<Lattice>{grid_range_}
        );
    }

    __host__ size_t count_layer_points(
        const sclx::array<value_type, 2>& boundary_distances_squared
    ) const {
        return sclx::algorithm::count_if(
            boundary_distances_squared,
            detail::layer_count_criteria_functor<Lattice>{
                absorption_layer_thickness_}
        );
    }

    friend __host__ void detail::assign_boundary_info<Lattice>(
        const uniform_grid_provider<Lattice>& provider,
        sclx::array<value_type, 2>& points,
        sclx::array<value_type, 2>& normals,
        size_t boundary_offset
    );

    __host__ void assign_boundary_info(
        sclx::array<value_type, 2>& points,
        sclx::array<value_type, 2>& normals,
        size_t boundary_offset
    ) const {
        detail::assign_boundary_info(*this, points, normals, boundary_offset);
    }

    friend __host__ void detail::assign_bulk_info<Lattice>(
        const uniform_grid_provider<Lattice>& provider,
        sclx::array<value_type, 2>& points
    );

    __host__ void assign_bulk_info(sclx::array<value_type, 2>& points) const {
        detail::assign_bulk_info(*this, points);
    }

    friend __host__ void detail::assign_bulk_and_layer_info<Lattice>(
        const uniform_grid_provider<Lattice>& provider,
        const sclx::array<value_type, 2>& boundary_distances_squared,
        const size_t& layer_points_offset,
        const sclx::array<value_type, 2>& old_points,
        sclx::array<value_type, 2>& points,
        sclx::array<value_type, 1>& absorption_coefficients
    );

    __host__ void assign_bulk_and_layer_info(
        const sclx::array<value_type, 2>& boundary_distances_squared,
        const size_t& layer_points_offset,
        const sclx::array<value_type, 2>& old_points,
        sclx::array<value_type, 2>& points,
        sclx::array<value_type, 1>& absorption_coefficients
    ) const {
        detail::assign_bulk_and_layer_info(
            *this,
            boundary_distances_squared,
            layer_points_offset,
            old_points,
            points,
            absorption_coefficients
        );
    }
};

}  // namespace naga::fluids::nonlocal_lbm

#include "detail/uniform_grid_provider.inl"
