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

#include <naga/interpolation/radial_point_method.cuh>
#include <naga/particle_segmentation/nearest_neighbors.cuh>
#include <scalix/algorithm/reduce.cuh>
#include <scalix/filesystem.hpp>
#include <fstream>

__device__ float field_function(const float* x) {
    return naga::math::sin(x[0]) * naga::math::cos(x[1]);
}

int main() {
    size_t grid_size   = 32;
    float grid_length  = 2 * naga::math::pi<float>;
    float grid_spacing = grid_length / (static_cast<float>(grid_size) - 1.0f);

    size_t interp_grid_size  = 48;
    float interp_grid_length = 2 * naga::math::pi<float>;
    float interp_grid_spacing
        = interp_grid_length / (static_cast<float>(interp_grid_size) - 1.0f);

    sclx::array<float, 2> source_grid{2, grid_size * grid_size};
    sclx::array<float, 1> source_values{grid_size * grid_size};
    sclx::array<float, 2> interp_grid{2, interp_grid_size * interp_grid_size};
    sclx::array<float, 1> interp_values{interp_grid_size * interp_grid_size};

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

    naga::rectangular_partitioner<float, 2> source_partitioner(source_grid, 32);

    auto [distances, indices] = naga::batched_nearest_neighbors(
        32,
        naga::default_point_map<float, 2>{interp_grid},
        source_partitioner
    );

    naga::interpolation::radial_point_method<naga::default_point_map<float, 2>>
        interpolator(
            source_grid,
            indices,
            naga::default_point_map<float, 2>{interp_grid},
            grid_spacing
        );

    interpolator.interpolate(source_values, interp_values);

    auto interp_errors
        = sclx::zeros<float, 1>({interp_grid_size * interp_grid_size});

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{interp_grid_size * interp_grid_size},
            interp_errors,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                interp_errors(idx[0])
                    = naga::math::abs(
                        interp_values(idx[0])
                        - field_function(&interp_grid(0, idx[0]))
                    );
            }
        );
    });

    auto max_error = sclx::algorithm::reduce(
        interp_errors,
        0.0f,
        sclx::algorithm::max<float>{}
    );

    std::cout << "Max error: " << max_error << std::endl;

    auto save_dir = sclx::filesystem::path(__FILE__).parent_path() / "radial_point_method_results";
    sclx::filesystem::create_directories(save_dir);
    auto save_path = save_dir / "radial_point_method_results.csv";
    interp_values.prefetch_async({sclx::cuda::traits::cpu_device_id});
    std::ofstream save_file(save_path);
    save_file << "x,y,f" << std::endl;
    for (size_t i = 0; i < interp_grid_size * interp_grid_size; ++i) {
        save_file
            << interp_grid(0, i) << ","
            << interp_grid(1, i) << ","
            << interp_values(i) << std::endl;
    }
    save_file.close();

    return 0;
}
