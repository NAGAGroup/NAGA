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

#include "utils.hpp"
#include <chrono>
#include <naga/interpolation/radial_point_method.cuh>
#include <naga/nonlocal_calculus/divergence.cuh>
#include <scalix/filesystem.hpp>

using value_type = float;

template<class PointType>
__host__ __device__ naga::point_t<value_type, 2>
field_function(const PointType& x) {
    return naga::point_t<value_type, 2>{
        {naga::math::sin(x[0]) * naga::math::cos(x[1]), 0}};
}

template<class PointType>
__host__ __device__ value_type expected_divergence_function(const PointType& x
) {
    return naga::math::cos(x[0]) * naga::math::cos(x[1]);
}

int main() {
    size_t grid_size       = 103;
    value_type grid_length = 2 * naga::math::pi<value_type>;
    value_type grid_spacing
        = grid_length / (static_cast<value_type>(grid_size) - 1.0f);

    sclx::array<value_type, 2> source_grid{2, grid_size * grid_size};
    value_type vector_scaler[2] = {1.0f, 0.0f};
    sclx::array<value_type, 1> source_values_scalar{grid_size * grid_size};
    sclx::array<value_type, 2> source_values{2, grid_size * grid_size};
    sclx::array<value_type, 1> expected_divergence{grid_size * grid_size};

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            source_grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                source_grid(0, idx[0])
                    = static_cast<value_type>(idx[0] % grid_size)
                    * grid_spacing;
                source_grid(1, idx[0])
                    = static_cast<value_type>(idx[0] / grid_size)
                    * grid_spacing;
            }
        );
    });
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            source_values,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                const auto& field_value
                    = field_function(&source_grid(0, idx[0]));
                source_values(0, idx[0]) = field_value[0];
                source_values(1, idx[0]) = field_value[1];
            }
        );
    });
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            source_values_scalar,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                const auto& field_value
                    = field_function(&source_grid(0, idx[0]));
                source_values_scalar(idx[0]) = field_value[0];
            }
        );
    });
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{grid_size * grid_size},
            expected_divergence,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                expected_divergence[idx]
                    = expected_divergence_function(&source_grid(idx[0]));
            }
        );
    });

    naga::nonlocal_calculus::operator_builder<value_type, 2> builder(source_grid
    );

    auto divergence
        = builder.create<naga::nonlocal_calculus::divergence_operator>();
    divergence.enable_cusparse_algorithm();

    sclx::array<value_type, 1> divergence_values{grid_size * grid_size};

    auto start = std::chrono::high_resolution_clock::now();
    divergence.apply_v2(vector_scaler, source_values_scalar, divergence_values);
    auto end1 = std::chrono::high_resolution_clock::now();

    divergence.apply_v2(vector_scaler, source_values_scalar, divergence_values);
    auto end2 = std::chrono::high_resolution_clock::now();

    divergence.apply_v2(vector_scaler, source_values_scalar, divergence_values);
    auto end3 = std::chrono::high_resolution_clock::now();

    sclx::array<value_type, 1> divergence_values_no_cusparse{grid_size * grid_size};
    divergence.disable_cusparse_algorithm();
    divergence.apply(source_values, divergence_values_no_cusparse);

    std::chrono::duration<double> elapsed_ms1 = (end1 - start) * 1000;
    std::chrono::duration<double> elapsed_ms2 = (end2 - end1) * 1000;
    std::chrono::duration<double> elapsed_ms3 = (end3 - end2) * 1000;

    std::cout << "First run: " << elapsed_ms1.count() << " ms\n";
    std::cout << "Second run: " << elapsed_ms2.count() << " ms\n";
    std::cout << "Third run: " << elapsed_ms3.count() << " ms\n";

    auto results_path
        = get_examples_results_dir() / "nonlocal_divergence_results";
    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "nonlocal_divergence_results.csv");
    file << "x,y,divergence\n";
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << divergence_values(i) << "\n";
    }
    file.close();
    file = std::ofstream(results_path / "nonlocal_divergence_expected.csv");
    file << "x,y,divergence\n";
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << expected_divergence[i] << "\n";
    }
    file.close();
    file = std::ofstream(results_path / "nonlocal_divergence_no_cusparse.csv");
    file << "x,y,divergence\n";
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << divergence_values_no_cusparse(i) << "\n";
    }

    return 0;
}
