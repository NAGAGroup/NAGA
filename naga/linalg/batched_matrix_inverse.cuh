
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
#include "../detail/cublas.cuh"
#include "detail/batched_matrix_inverse.cuh"
#include <future>
#include <scalix/fill.cuh>

namespace naga::linalg {

template<class T>
__host__ void batched_matrix_inverse(
    sclx::array<T, 3>& A,
    sclx::array<T, 3>& A_inv,
    bool copy_A = true
) {

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

    if (A_inv.shape() != A.shape()) {
        sclx::throw_exception<std::invalid_argument>(
            "A_inv must have the same shape as A",
            "naga::linalg::"
        );
    }

    const std::vector<std::tuple<int, size_t, size_t>>& split_info
        = sclx::get_device_split_info(A);

    auto current = std::experimental::source_location::current();

    std::vector<std::future<void>> futures;

    for (const auto& split : split_info) {
        int device_id      = std::get<0>(split);
        size_t slice_start = std::get<1>(split);
        size_t slice_len   = std::get<2>(split);

        auto task_lambda = [=]() {
            auto handle = naga::detail::cublas::handle_t::create();

            auto A_slice
                = A.get_range({slice_start}, {slice_start + slice_len});
            auto A_inv_slice
                = A_inv.get_range({slice_start}, {slice_start + slice_len});

            int *info, *pivot;
            int dims       = A_slice.shape()[0];
            int batch_size = A_slice.shape()[2];
            auto error     = cudaMallocManaged(&info, batch_size * sizeof(int));
            sclx::cuda::cuda_exception::raise_if_not_success(
                error,
                current,
                "naga::linalg::"
            );
            error = cudaMallocManaged(&pivot, batch_size * dims * sizeof(int));
            sclx::cuda::cuda_exception::raise_if_not_success(
                error,
                current,
                "naga::linalg::"
            );

            sclx::array<T, 3> A_slice_copy;
            std::future<void> prefetched_future1;
            if (copy_A) {
                A_slice_copy = sclx::array<T, 3>(
                    {static_cast<size_t>(dims),
                     static_cast<size_t>(dims),
                     static_cast<size_t>(batch_size)},
                    false
                );

                A_slice_copy.set_primary_devices(
                    std::vector<int>{device_id},
                    false
                );
                prefetched_future1 = std::move(A_slice_copy.prefetch_async());
            } else {
                A_slice_copy       = A_slice;
                prefetched_future1 = std::async(std::launch::deferred, []() {});
            }

            sclx::array<T*, 1> A_slice_ptr({A_slice.shape()[2]}, false);
            sclx::array<T*, 1> A_inv_slice_ptr({A_inv_slice.shape()[2]}, false);

            A_slice_ptr.set_primary_devices(std::vector<int>{device_id}, false);
            auto prefetched_future2 = A_slice_ptr.prefetch_async();

            A_inv_slice_ptr.set_primary_devices(
                std::vector<int>{device_id},
                false
            );
            auto prefetched_future3 = A_inv_slice_ptr.prefetch_async();

            prefetched_future1.get();
            prefetched_future2.get();
            prefetched_future3.get();

            std::future<void> copy_future;
            if (copy_A) {
                auto copy_future_
                    = sclx::execute_kernel([&](sclx::kernel_handler& handler) {
                          handler.launch(
                              sclx::md_range_t<3>(A_slice_copy.shape()),
                              A_slice_copy,
                              [=] __device__(sclx::md_index_t<3> & idx, auto&) {
                                  A_slice_copy[idx] = A_slice[idx];
                              }
                          );
                      });
                copy_future = std::move(copy_future_);
            } else {
                copy_future = std::async(std::launch::deferred, []() {});
            }

            auto ptr_setup_future
                = sclx::execute_kernel([=](sclx::kernel_handler& handler) {
                      sclx::array_list<T*, 1, 2> ptrs
                          = {A_slice_ptr, A_inv_slice_ptr};

                      handler.launch(
                          sclx::md_range_t<1>(A_slice_ptr.shape()),
                          ptrs,
                          [=] __device__(sclx::md_index_t<1> & idx, auto&) {
                              A_slice_ptr[idx] = &A_slice_copy(0, 0, idx[0]);
                              A_inv_slice_ptr[idx] = &A_inv_slice(0, 0, idx[0]);
                          }
                      );
                  });

            copy_future.get();
            ptr_setup_future.get();
            A_slice_copy.unset_read_mostly();
            auto status_getrf = detail::cublas_getrf_batched<T>::compute(
                handle,
                dims,
                A_slice_ptr.data().get(),
                dims,
                pivot,
                info,
                batch_size
            );

            A_inv_slice.unset_read_mostly();
            auto status_getri = detail::cublas_getri_batched<T>::compute(
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

            cudaFree(info);
            cudaFree(pivot);

            if (status_getrf != CUBLAS_STATUS_SUCCESS) {
                sclx::throw_exception<std::runtime_error>(
                    "CUBLAS error on getrf: "
                        + std::string(cublasGetStatusString(
                            static_cast<cublasStatus_t>(status_getrf)
                        )),
                    "naga::linalg::",
                    current
                );
            }

            if (status_getri != CUBLAS_STATUS_SUCCESS) {
                sclx::throw_exception<std::runtime_error>(
                    "CUBLAS error on getri: "
                        + std::string(cublasGetStatusString(
                            static_cast<cublasStatus_t>(status_getri)
                        )),
                    "naga::linalg::",
                    current
                );
            }
        };

        //        sclx::cuda::set_device(device_id);
        //        task_lambda();
        //        sclx::cuda::set_device(0);

        futures.emplace_back(
            sclx::cuda::task_scheduler::submit_task(device_id, task_lambda)
        );
    }

    for (auto& future : futures) {
        future.get();
    }

    A.set_read_mostly();
    A_inv.set_read_mostly();
}

template<class T>
__host__ sclx::array<T, 3>
batched_matrix_inverse(sclx::array<T, 3>& A, bool copy_A = true) {
    sclx::array<T, 3> A_inv({A.shape()[0], A.shape()[1], A.shape()[2]});
    auto split_info = sclx::get_device_split_info(A);
    A_inv.set_primary_devices(split_info);
    batched_matrix_inverse(A, A_inv, copy_A);
    return A_inv;
}

}  // namespace naga::linalg
