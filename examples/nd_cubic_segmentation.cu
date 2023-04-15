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
#include <fstream>
#include <naga/segmentation/nd_cubic_segmentation.cuh>
#include <scalix/filesystem.hpp>
#include <string>

int main() {
    float domain_length = 1.f;
    size_t grid_size    = 25;
    float step_size     = domain_length / (grid_size - 1);
    sclx::array<float, 2> grid2d({2, grid_size * grid_size});
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            grid2d,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                grid2d(0, idx[0]) = (idx[0] % grid_size) * step_size;
                grid2d(1, idx[0]) = (idx[0] / grid_size) * step_size;
            }
        );
    });

    auto start         = std::chrono::high_resolution_clock::now();
    int partition_size = 64;
    naga::segmentation::nd_cubic_segmentation<float, 2> segmentation2d(
        grid2d,
        partition_size
    );
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time to build 2D segmentation: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - start
                 )
                         .count()
                     / 1000.f
              << "ms" << std::endl;

    const auto& segmentation_shape = segmentation2d.shape();
    auto partition_count           = segmentation2d.partition_count();
    auto results_path = sclx::filesystem::path(__FILE__).parent_path()
                      / "nd_cubic_segmentation_results";
    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "nd_cubic_segmentation2d.csv");
    file << "x,y,p" << std::endl;
    size_t p_idx = 0;
    for (auto partition : segmentation2d) {
        for (const auto& point : partition) {
            file << point.x() << "," << point.y() << "," << p_idx << std::endl;
        }
        p_idx++;
    }
    file.close();

    sclx::array<float, 2> grid3d({3, grid_size * grid_size * grid_size});
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>{grid_size * grid_size * grid_size},
            grid3d,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                size_t i = idx[0] / (grid_size * grid_size);
                size_t j = (idx[0] - i * grid_size * grid_size) / grid_size;
                size_t k = idx[0] - i * grid_size * grid_size - j * grid_size;
                grid3d(0, idx[0]) = i * step_size;
                grid3d(1, idx[0]) = j * step_size;
                grid3d(2, idx[0]) = k * step_size;
            }
        );
    }).get();

    start = std::chrono::high_resolution_clock::now();
    naga::segmentation::nd_cubic_segmentation<float, 3> segmentation3d(
        grid3d,
        partition_size
    );
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time to build 3D segmentation: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - start
                 )
                         .count()
                     / 1000.f
              << "ms" << std::endl;

    const auto& segmentation3d_shape = segmentation3d.shape();
    partition_count                  = segmentation3d.partition_count();
    file = std::ofstream(results_path / "nd_cubic_segmentation3d.csv");
    file << "x,y,z,p" << std::endl;
    p_idx = 0;
    for (auto partition : segmentation3d) {
        for (const auto& point : partition) {
            file << point.x() << "," << point.y() << "," << point.z() << ","
                 << p_idx << std::endl;
        }
        p_idx++;
    }
    file.close();

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        sclx::array<char, 1> dummy({1});
        handler.launch(
            sclx::md_range_t<1>(dummy.shape()),
            dummy,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                int p_idx = 0;
                for (auto partition : segmentation3d) {
                    for (const auto& point : partition) {
                        printf("%f, %f, %d\n", point.x(), point.y(), p_idx);
                    }
                    p_idx++;
                }
            }
        );
    }).get();

    return 0;
}
