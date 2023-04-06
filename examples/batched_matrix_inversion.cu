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
#include <naga/linalg/batched_matrix_inverse.cuh>
#include <scalix/algorithm/reduce.cuh>

int main() {
    // clang-format off
    float A_inv_check[3][3] = {
        {
            -1.791666667, 0.916666667, -0.125000000,
        },
        {
            1.583333333, -0.833333333, 0.250000000,
        },
        {
            -0.125000000, 0.250000000, -0.125000000,
        },
    };

    float A[3][3] = {
        {
            1.0, 2.0, 3.0,
        },
        {
            4.0, 5.0, 6.0,
        },
        {
            7.0, 8.0, 1.,
        },
    };
    // clang-format on

    size_t batch_size = 10'000'000;
    sclx::array<float, 3> A_batched{3, 3, batch_size};

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<3>(A_batched.shape()),
            A_batched,
            [=] __device__(const sclx::md_index_t<3>& index, const auto&) {
                A_batched[index] = A[index[0]][index[1]];
            }
        );
    }).get();

    auto begin         = std::chrono::high_resolution_clock::now();
    auto A_inv_batched = naga::linalg::batched_matrix_inverse(A_batched);
    auto end           = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms"
              << std::endl;

    auto error_count = sclx::zeros<int>({batch_size});

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>(error_count.shape()),
            error_count,
            [=] __device__(const sclx::md_index_t<1>& index, const auto&) {
                float tol = 1e-6;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (std::abs(
                                A_inv_batched(i, j, index[0])
                                - A_inv_check[i][j]
                            )
                            > tol) {
                            error_count[index] += 1;
                            return;
                        }
                    }
                }
            }
        );
    }).get();

    auto sum_errors
        = sclx::algorithm::reduce(error_count, 0, sclx::algorithm::plus<>());

    if (sum_errors != 0) {
        std::cerr << "Error: " << sum_errors << " errors found." << std::endl;
        return 1;
    }

    auto first_A_inv = A_inv_batched.get_slice({0});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << first_A_inv(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
