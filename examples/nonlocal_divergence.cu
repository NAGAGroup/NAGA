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
#include <naga/nonlocal_calculus/divergence.cuh>
#include <scalix/filesystem.hpp>

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
    sclx::array<float, 2> source_values{2, grid_size * grid_size};

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
                source_values(0, idx[0]) = field_function(&source_grid(0, idx[0]));
                source_values(1, idx[0]) = 0.f;
            }
        );
    });

    naga::nonlocal_calculus::operator_builder<float, 2> builder(source_grid);

    auto divergence
        = builder.create<naga::nonlocal_calculus::divergence_operator>();

    sclx::array<float, 1> divergence_values{grid_size * grid_size};

    divergence.apply(source_values, divergence_values);

    auto results_path = sclx::filesystem::path(__FILE__).parent_path()
                      / "div_results";
    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "div_results.csv");
    file << "x,y,divergence\n";
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << divergence_values(i) << "\n";
    }
    file.close();

    return 0;
}
