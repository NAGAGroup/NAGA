
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
#include "detail/batched_matrix_inverse.cuh"
#include <future>
#include <scalix/fill.cuh>

namespace naga::linalg {

template<class T>
__host__ sclx::array<T, 3> batched_matrix_inverse(sclx::array<T, 3>& A) {

    if (A.shape()[0] != A.shape()[1]) {
        sclx::throw_exception<std::invalid_argument>(
            "Matrices must be square",
            "naga::linalg::"
        );
    }

    if (A.shape()[2] >= std::numeric_limits<int>::max()) {
        sclx::throw_exception<std::invalid_argument>(
            "Batch size must be less than std::numeric_limits<int>::max()",
            "naga::linalg::"
        );
    }

    sclx::array<T, 3> A_inv
        = sclx::zeros<T>({A.shape()[0], A.shape()[1], A.shape()[2]});
    const std::vector<std::tuple<int, size_t, size_t>>& split_info
        = sclx::get_device_split_info(A);
    A_inv.set_primary_devices(split_info);

    auto current = std::experimental::source_location::current();

    std::vector<std::future<void>> futures;

    for (const auto& [device_id, slice_start, slice_len] : split_info) {
        auto A_slice     = A.get_range({slice_start}, {slice_len});
        auto A_inv_slice = A_inv.get_range({slice_start}, {slice_len});

        sclx::array<T*, 1> A_slice_ptr = sclx::zeros<T*>({A_slice.shape()[2]});
        sclx::array<T*, 1> A_inv_slice_ptr
            = sclx::zeros<T*>({A_inv_slice.shape()[2]});
        auto ptr_setup_fut = sclx::execute_kernel([&](sclx::kernel_handler&
                                                          handler) {
            sclx::array_list<T*, 1, 2> result = {A_slice_ptr, A_inv_slice_ptr};
            handler.launch(
                sclx::md_range_t<1>(A_slice_ptr.shape()),
                result,
                [=] __device__(sclx::md_index_t<1> & idx, auto&) {
                    result[0][idx] = &A_slice(0, 0, idx[0]);
                    result[1][idx] = &A_inv_slice(0, 0, idx[0]);
                }
            );
        });

        auto lambda = [=](int d_id, std::future<void> ptr_setup_fut) {
            sclx::cuda::set_device(d_id);

            cublasHandle_t handle;
            cublasCreate(&handle);
            int *info, *pivot;
            int dims       = A_slice.shape()[0];
            int batch_size = A_slice.shape()[2];
            cudaMalloc(&info, batch_size * sizeof(int*));
            cudaMalloc(&pivot, batch_size * dims * sizeof(int*));

            ptr_setup_fut.get();
            auto status = detail::cublas_getrf_batched<T>::compute(
                handle,
                dims,
                A_slice_ptr.data().get(),
                dims,
                pivot,
                info,
                batch_size
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                sclx::throw_exception<std::runtime_error>(
                    "CUBLAS error: " + std::to_string(status),
                    "naga::linalg::",
                    current
                );
            }

            status = detail::cublas_getri_batched<T>::compute(
                handle,
                dims,
                A_slice_ptr.data().get(),
                dims,
                pivot,
                A_inv_slice_ptr.data().get(),
                dims,
                info,
                batch_size
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                sclx::throw_exception<std::runtime_error>(
                    "CUBLAS error: " + std::to_string(status),
                    "naga::linalg::",
                    current
                );
            }

            cudaFree(info);
            cudaFree(pivot);
            cublasDestroy(handle);
        };

        futures.push_back(std::async(
            std::launch::async,
            lambda,
            device_id,
            std::move(ptr_setup_fut)
        ));
    }

    for (auto& future : futures) {
        future.get();
    }

    return A_inv;
}

}  // namespace naga::linalg
