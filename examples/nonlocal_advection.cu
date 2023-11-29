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
#include <naga/nonlocal_calculus/advection.cuh>
#include <numeric>
#include <scalix/filesystem.hpp>

using value_type = float;

template<class PointType>
__host__ __device__ value_type field_function(const PointType& x) {
    // 2D gaussian pulse with radius 0.05 and center (0.5, 0.5)
    value_type sigma_sq = naga::math::loopless::pow<2>(value_type{0.02});
    return 0.2f
         * naga::math::exp(
               -naga::math::pow(x[0] - 0.5f, 2) / sigma_sq / 2.0f
               - naga::math::pow(x[1] - 0.5f, 2) / sigma_sq / 2.0f
         );
}

int main() {
    size_t grid_size       = 131;
    value_type grid_length = 1.f;
    value_type grid_spacing
        = grid_length / (static_cast<value_type>(grid_size) - 1.0f);

    sclx::array<value_type, 2> source_grid{2, grid_size * grid_size};
    sclx::array<value_type, 1> source_values{grid_size * grid_size};

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
                source_values[idx] = field_value;
            }
        );
    });

    auto results_path
        = get_examples_results_dir() / "nonlocal_advection_results";
    sclx::filesystem::remove_all(results_path);
    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "initial_condition.csv");
    file << "x,y,value\n";
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << source_values(i) << "\n";
    }
    file.close();

    value_type time_step = naga::math::loopless::pow<1>(grid_spacing) / std::sqrt(2.f) / 5.f;

    sclx::array<value_type, 1> advection_result{grid_size * grid_size};
    value_type velocity[2] = {.1f, .1f};
    auto velocity_field    = naga::nonlocal_calculus::
        constant_velocity_field<value_type, 2>::create(velocity);

    auto advection_op
        = naga::nonlocal_calculus::advection_operator<value_type, 2>::create(
            source_grid
        );

    advection_op.enable_implicit(velocity, time_step);
//    auto start = std::chrono::high_resolution_clock::now();
//
//    advection_op
//        .step_forward(
//            velocity_field,
//            source_values,
//            advection_result,
//            time_step
//        )
//        .get();
//    auto end1 = std::chrono::high_resolution_clock::now();
//
//    advection_op
//        .step_forward(
//            velocity_field,
//            source_values,
//            advection_result,
//            time_step
//        )
//        .get();
//    auto end2 = std::chrono::high_resolution_clock::now();
//
//    advection_op
//        .step_forward(
//            velocity_field,
//            source_values,
//            advection_result,
//            time_step
//        )
//        .get();
//    auto end3 = std::chrono::high_resolution_clock::now();
//
//    std::chrono::duration<double> elapsed_ms1 = (end1 - start) * 1000;
//    std::chrono::duration<double> elapsed_ms2 = (end2 - end1) * 1000;
//    std::chrono::duration<double> elapsed_ms3 = (end3 - end2) * 1000;
//
//    std::cout << "First run: " << elapsed_ms1.count() << " ms\n";
//    std::cout << "Second run: " << elapsed_ms2.count() << " ms\n";
//    std::cout << "Third run: " << elapsed_ms3.count() << " ms\n";
//
    value_type time = 0.0f;
    uint save_frame = 0;
    value_type fps  = 60.0f;
    sclx::array<int, 1> node_types{grid_size * grid_size};
    sclx::fill(node_types, 0);
    for (auto& global_idx: advection_op.explicit_indices_) {
        node_types[global_idx] = 1;
    }
    while (time < 2.f) {
        advection_op
            .apply_implicit(
                source_values,
                advection_result
            );
        time += time_step;
        if (time * fps >= static_cast<value_type>(save_frame)) {
            std::cout << "Time: " << time << "\n";
            file.open(
                results_path
                / (std::string("advection_result.csv.")
                   + std::to_string(save_frame))
            );
            file << "x,y,value,type\n";
            for (size_t i = 0; i < grid_size * grid_size; ++i) {
                file << source_grid(0, i) << "," << source_grid(1, i) << ","
                     << advection_result(i) << "," << node_types(i) << "\n";
            }
            file.close();
            ++save_frame;
        }
        sclx::assign_array(advection_result, source_values);
    }

    return 0;
}
