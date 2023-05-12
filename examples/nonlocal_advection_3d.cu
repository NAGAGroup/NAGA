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

#include <naga/nonlocal_calculus/advection.cuh>
#include <scalix/filesystem.hpp>

using value_type = double;

template<class PointType>
__host__ __device__ value_type field_function(const PointType& x) {
    // 3D gaussian pulse with radius 0.05 and center (0.5, 0.5)
    value_type sigma_sq = naga::math::loopless::pow<2>(value_type{0.02});
    return 0.2f
         * naga::math::exp(
               -naga::math::pow(x[0] - 0.5f, 2) / sigma_sq / 2.0f
               - naga::math::pow(x[1] - 0.5f, 2) / sigma_sq / 2.0f
               - naga::math::pow(x[2] - 0.5f, 2) / sigma_sq / 2.0f
         );
}

int main() {
    size_t grid_size       = 64;
    value_type grid_length = 1.f;
    value_type grid_spacing
        = grid_length / (static_cast<value_type>(grid_size) - 1.0f);

    size_t point_count = grid_size * grid_size * grid_size;
    sclx::array<value_type, 2> source_grid{3, point_count};
    sclx::array<value_type, 1> source_values{point_count};

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{point_count},
            source_grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                source_grid(0, idx[0])
                    = static_cast<value_type>(idx[0] % grid_size)
                    * grid_spacing;
                source_grid(1, idx[0])
                    = static_cast<value_type>((idx[0] / grid_size) % grid_size)
                    * grid_spacing;
                source_grid(2, idx[0])
                    = static_cast<value_type>(idx[0] / grid_size / grid_size)
                    * grid_spacing;
            }
        );
    });
    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{point_count},
            source_values,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                const auto& field_value
                    = field_function(&source_grid(0, idx[0]));
                source_values[idx] = field_value;
            }
        );
    });

    auto results_path = sclx::filesystem::path(__FILE__).parent_path()
                      / "nonlocal_advection_results_3d";
    sclx::filesystem::create_directories(results_path);
    std::ofstream file(results_path / "initial_condition.csv");
    file << "x,y,z,value\n";
    for (size_t i = 0; i < point_count; ++i) {
        file << source_grid(0, i) << "," << source_grid(1, i) << ","
             << source_grid(2, i) << "," << source_values(i) << "\n";
    }
    file.close();

    value_type time_step = naga::math::loopless::pow<2>(grid_spacing) * 2.f;

    sclx::array<value_type, 1> advection_result{point_count};
    value_type velocity[3] = {0.1f, 0.1, 0.1f};
    auto velocity_field    = naga::nonlocal_calculus::
        constant_velocity_field<value_type, 3>::create(velocity);
    auto advection_op
        = naga::nonlocal_calculus::advection_operator<value_type, 3>::create(
            source_grid
        );

    value_type time = 0.0f;
    uint save_frame = 0;
    value_type fps  = 60.0f;
    while (time < 1.f) {
        advection_op.step_forward(
            velocity_field,
            source_values,
            advection_result,
            time_step
        );
        time += time_step;
        std::cout << "Time: " << time << "\n";
        if (time * fps >= static_cast<value_type>(save_frame)) {
            file.open(
                results_path
                / (std::string("advection_result.csv.")
                   + std::to_string(save_frame))
            );
            file << "x,y,z,value\n";
            for (size_t i = 0; i < point_count; ++i) {
                file << source_grid(0, i) << "," << source_grid(1, i) << ","
                     << source_grid(2, i) << "," << advection_result(i) << "\n";
            }
            file.close();
            ++save_frame;
        }
        source_values.assign_from(advection_result);
    }

    return 0;
}
