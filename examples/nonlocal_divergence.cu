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

#include <chrono>
#include <naga/interpolation/radial_point_method.cuh>
#include <naga/nonlocal_calculus/operators.cuh>
#include <naga/segmentation/nearest_neighbors.cuh>
#include <scalix/filesystem.hpp>
#include <scalix/fill.cuh>

template<class PointType>
__host__ __device__ float field_function(const PointType& x) {
    return naga::math::sin(x[0]) * naga::math::cos(x[1]);
}

int main() {
    uint support_size = 32;  // number of support nodes used for interpolation

    size_t grid_size   = 50;
    float grid_length  = 2 * naga::math::pi<float>;
    float grid_spacing = grid_length / (static_cast<float>(grid_size) - 1.0f);

    sclx::array<float, 2> source_grid{2, grid_size * grid_size};
    sclx::array<float, 1> source_values{grid_size * grid_size};
    sclx::array<float, 1> interp_values{
        grid_size * grid_size
        * naga::nonlocal_calculus::detail::num_quad_points_2d};
    sclx::array<float, 1> interaction_radii{grid_size * grid_size};

    sclx::fill(interaction_radii, grid_spacing * 0.1f);

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            source_grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                source_grid(0, idx[0]) = (idx[0] % grid_size) * grid_spacing;
                source_grid(1, idx[0]) = (idx[0] / grid_size) * grid_spacing;
            }
        );
    });
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            source_values,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                source_values(idx[0]) = field_function(&source_grid(0, idx[0]));
            }
        );
    });

    naga::nonlocal_calculus::detail::quadrature_point_map<float, 2>
        quadrature_points_map(source_grid, interaction_radii);

    naga::segmentation::nd_cubic_segmentation<float, 2> source_segmentation(
        source_grid,
        support_size
    );

    auto [distances_squared, indices]
        = naga::segmentation::batched_nearest_neighbors(
            support_size,
            naga::default_point_map<float, 2>{source_grid},
            source_segmentation
        );

    auto interpolator
        = naga::interpolation::radial_point_method<>::create_interpolator(
            source_grid,
            indices,
            distances_squared,
            quadrature_points_map,
            grid_spacing,
            naga::nonlocal_calculus::detail::num_quad_points_2d
        );

    interpolator.interpolate(source_values, interp_values);

    sclx::array<float, 2> quad_points{2, quadrature_points_map.size()};

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{quadrature_points_map.size()},
            quad_points,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                auto quad_point        = quadrature_points_map[idx];
                quad_points(0, idx[0]) = quad_point[0];
                quad_points(1, idx[0]) = quad_point[1];
            }
        );
    });

    auto save_dir = sclx::filesystem::path(__FILE__).parent_path()
                  / "nonlocal_divergence_results";
    sclx::filesystem::create_directories(save_dir);
    std::cout << "Save directory: " << save_dir << std::endl;
    auto save_path = save_dir / "quad_points.csv";
    quad_points.prefetch_async({sclx::cuda::traits::cpu_device_id});
    std::ofstream save_file(save_path);
    save_file << "x,y,f" << std::endl;
    for (size_t i = 0; i < quad_points.shape()[1]; ++i) {
        save_file << quad_points(0, i) << "," << quad_points(1, i) << ","
                  << interp_values(i) << std::endl;
    }
    save_file.close();

    return 0;
}
