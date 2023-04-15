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

/** @file radial_point_interpolation.cu
 * @brief Showcasing our GPU accelerated radial point interpolation method.
 *
 * This example creates a two identical 2D grids, one with a known function and
 * the other with an "unknown" function. The "unknown" function is interpolated
 * to and verified against the known function.
 */

#include <chrono>
#include <fstream>
#include <naga/interpolation/radial_point_method.cuh>
#include <naga/particle_segmentation/nearest_neighbors.cuh>
#include <scalix/algorithm/reduce.cuh>
#include <scalix/filesystem.hpp>

__device__ float field_function(const float* x) {
    return naga::math::sin(x[0]) * naga::math::cos(x[1]);
}

int main() {
    uint support_size = 32;  // number of support nodes used for interpolation

    size_t grid_size   = 500;
    float grid_length  = 2 * naga::math::pi<float>;
    float grid_spacing = grid_length / (static_cast<float>(grid_size) - 1.0f);

    size_t interp_grid_size  = 500;
    float interp_grid_length = 2 * naga::math::pi<float>;
    float interp_grid_spacing
        = interp_grid_length / (static_cast<float>(interp_grid_size) - 1.0f);

    sclx::array<float, 2> source_grid{2, grid_size * grid_size};
    sclx::array<float, 1> source_values{grid_size * grid_size};
    sclx::array<float, 2> interp_grid{2, interp_grid_size * interp_grid_size};
    sclx::array<float, 1> interp_values{interp_grid_size * interp_grid_size};

    // populate source grid and field values
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

    // populate interpolated grid
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{interp_grid_size * interp_grid_size},
            interp_grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                interp_grid(0, idx[0])
                    = (idx[0] % interp_grid_size) * interp_grid_spacing;
                interp_grid(1, idx[0])
                    = (idx[0] / interp_grid_size) * interp_grid_spacing;
            }
        );
    });

    std::cout << "Source point count: " << source_grid.shape()[1] << std::endl;
    std::cout << "Interpolated point count: " << interp_grid.shape()[1]
              << std::endl;

    // First, we need to create a partitioner for the source grid, so we
    // can compute the nearest neighbors.
    auto start = std::chrono::high_resolution_clock::now();
    naga::rectangular_partitioner<float, 2> source_partitioner(
        source_grid,
        support_size
    );
    auto end_partition = std::chrono::high_resolution_clock::now();

    // We use the nearest neighbors algorithm to provide the interpolation
    // indices to the radial point method.
    auto [distances, indices] = naga::batched_nearest_neighbors(
        support_size,
        naga::default_point_map<float, 2>{interp_grid},
        source_partitioner
    );
    auto end_neighbors = std::chrono::high_resolution_clock::now();

    // construct the interpolator
    auto interpolator
        = naga::interpolation::radial_point_method<>::create_interpolator(
            source_grid,
            indices,
            naga::default_point_map<float, 2>{interp_grid},
            grid_spacing
        );
    auto end_interpolator = std::chrono::high_resolution_clock::now();

    // Now we interpolate the field values
    //
    // Note that we run it 3 times. We do this to show how the unified memory
    // driver improves performance over successive runs. This is especially
    // noticeable when running multiple devices.
    interpolator.interpolate(source_values, interp_values);
    auto end_interpolate = std::chrono::high_resolution_clock::now();

    interpolator.interpolate(source_values, interp_values);
    auto end_interpolate2 = std::chrono::high_resolution_clock::now();

    interpolator.interpolate(source_values, interp_values);
    auto end_interpolate3 = std::chrono::high_resolution_clock::now();

    // compute the l2 error
    auto interp_errors
        = sclx::zeros<float, 1>({interp_grid_size * interp_grid_size});
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{interp_grid_size * interp_grid_size},
            interp_errors,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                interp_errors(idx[0]) = naga::math::loopless::pow<2>(
                    interp_values(idx[0])
                    - field_function(&interp_grid(0, idx[0]))
                );
            }
        );
    });

    auto sum_error = sclx::algorithm::reduce(
        interp_errors,
        0.0f,
        sclx::algorithm::plus<>{}
    );

    std::cout << std::endl
              << "l2 grid error, normalized: "
              << sum_error / interp_grid.shape()[1] << std::endl
              << std::endl;

    std::chrono::duration<double> partition_time
        = (end_partition - start) * 1000.f;
    std::chrono::duration<double> neighbors_time
        = (end_neighbors - end_partition) * 1000.f;
    std::chrono::duration<double> interpolator_time
        = (end_interpolator - end_neighbors) * 1000.f;
    std::chrono::duration<double> interpolate_time
        = (end_interpolate - end_interpolator) * 1000.f;
    std::chrono::duration<double> interpolate_time2
        = (end_interpolate2 - end_interpolate) * 1000.f;
    std::chrono::duration<double> interpolate_time3
        = (end_interpolate3 - end_interpolate2) * 1000.f;

    std::cout << "Time to construct partitioner: " << partition_time.count()
              << " ms" << std::endl;
    std::cout << "Time to find nearest neighbors: " << neighbors_time.count()
              << " ms" << std::endl;
    std::cout << "Time to construct interpolator: " << interpolator_time.count()
              << " ms" << std::endl;
    std::cout << "Time to interpolate: " << interpolate_time.count() << " ms"
              << std::endl;
    std::cout << "Time to interpolate (2nd run): " << interpolate_time2.count()
              << " ms" << std::endl;
    std::cout << "Time to interpolate (3rd run): " << interpolate_time3.count()
              << " ms" << std::endl;

    // save the results to view in paraview
    auto save_dir = sclx::filesystem::path(__FILE__).parent_path()
                  / "radial_point_method_results";
    sclx::filesystem::create_directories(save_dir);
    auto save_path = save_dir / "radial_point_method_results.csv";
    interp_values.prefetch_async({sclx::cuda::traits::cpu_device_id});
    std::ofstream save_file(save_path);
    save_file << "x,y,f" << std::endl;
    for (size_t i = 0; i < interp_grid_size * interp_grid_size; ++i) {
        save_file << interp_grid(0, i) << "," << interp_grid(1, i) << ","
                  << interp_values(i) << std::endl;
    }
    save_file.close();

    return 0;
}
