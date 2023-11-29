
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

namespace detail {
template<class T>
class batched_matrix_inverse_executor{
  public:
    std::future<void> static submit(
        int device_id,
        sclx::array<T, 3>& A,
        sclx::array<T, 3>& A_inv,
        bool copy_A,
        std::experimental::source_location current) {
        auto& device_executor = get_executor_for_device(device_id);
        auto expected = false;
        while (!device_executor.problem_definition_->is_preparing.compare_exchange_weak(expected, true)) {
            expected = false;
            std::this_thread::yield();
        }
        while (device_executor.problem_definition_->is_task_submitted.load()) {
            std::this_thread::yield();
        }
        device_executor.problem_definition_->A = A;
        device_executor.problem_definition_->A_inv = A_inv;
        device_executor.problem_definition_->copy_A = copy_A;
        device_executor.problem_definition_->current = current;
        device_executor.problem_definition_->promise = std::promise<void>();
        auto fut = device_executor.problem_definition_->promise.get_future();
        device_executor.problem_definition_->is_task_submitted.store(true);
        device_executor.problem_definition_->is_preparing.store(false);
        return fut;
    }

    ~batched_matrix_inverse_executor() {
        if (problem_definition_ == nullptr) {
            return;
        }
        while (problem_definition_->is_preparing.load()) {
            std::this_thread::yield();
        }
        while (problem_definition_->is_task_submitted.load()) {
            std::this_thread::yield();
        }
        problem_definition_->stop_thread.store(true);
        while (problem_definition_->is_running.load()) {
            std::this_thread::yield();
        }
    }


    struct problem_definition{
        std::atomic<bool> is_running{false};
        std::atomic<bool> is_task_submitted{false};
        std::atomic<bool> stop_thread{false};
        std::atomic<bool> is_preparing{false};
        sclx::array<T, 3> A;
        sclx::array<T, 3> A_inv;
        std::experimental::source_location current;
        bool copy_A;
        std::promise<void> promise;
    };

    static void thread_function(problem_definition* raw_problem_ptr, int device_id) {
        sclx::cuda::set_device(device_id);
        auto handle = naga::detail::cublas::handle_t::create();
        sclx::array<T, 3> A_copy;
        sclx::array<T, 1> A_copy_flat;
        sclx::array<T*, 1> A_ptr;
        sclx::array<T*, 1> A_inv_ptr;
        sclx::array<int, 1> info;
        sclx::array<int, 1> pivot;
        raw_problem_ptr->is_running.store(true);
        while (!raw_problem_ptr->stop_thread.load()) {
            if (raw_problem_ptr->is_task_submitted.load()) {
                auto A = raw_problem_ptr->A;
                auto A_inv = raw_problem_ptr->A_inv;
                auto current = raw_problem_ptr->current;
                auto copy_A = raw_problem_ptr->copy_A;

                int dims       = A.shape()[0];
                int batch_size = A.shape()[2];
                if (info.elements() < batch_size) {
                    info = sclx::array<int, 1>(sclx::shape_t<1>{static_cast<size_t>(batch_size)}, false);
                    info.set_primary_devices(std::vector<int>{device_id});
                }
                if (pivot.elements() < batch_size * dims) {
                    pivot = sclx::array<int, 1>(sclx::shape_t<1>{static_cast<size_t>(batch_size * dims)}, false);
                    pivot.set_primary_devices(std::vector<int>{device_id});
                }

                std::future<void> copy_future;
                if (copy_A) {
                    if (A_copy_flat.elements() < A.elements()) {
                        A_copy_flat = sclx::array<T, 1>(
                            sclx::shape_t<1>{static_cast<size_t>(A.elements())},
                            false
                        );
                        A_copy = sclx::array<T, 3>(A.shape(), A_copy_flat.data());
                        A_copy.set_primary_devices(
                            std::vector<int>{device_id}
                        );
                    }
                    auto copy_future_
                        = sclx::execute_kernel([&](sclx::kernel_handler& handler) {
                              handler.launch(
                                  sclx::md_range_t<3>(A_copy.shape()),
                                  A_copy,
                                  [=] __device__(sclx::md_index_t<3> & idx, auto&) {
                                      A_copy[idx] = A[idx];
                                  }
                              );
                          });
                    copy_future = std::move(copy_future_);
                } else {
                    A_copy = A;
                    copy_future = std::async(std::launch::deferred, []() {});
                }


                if (A_ptr.elements() < A.shape()[2]){
                    A_ptr = sclx::array<T*, 1>(
                        sclx::shape_t<1>{static_cast<size_t>(A.shape()[2])},
                        false
                    );
                    A_ptr.set_primary_devices(
                        std::vector<int>{device_id}
                    );
                }
                if (A_inv_ptr.elements() < A_inv.shape()[2]){
                    A_inv_ptr = sclx::array<T*, 1>(
                        sclx::shape_t<1>{static_cast<size_t>(A_inv.shape()[2])},
                        false
                    );
                    A_inv_ptr.set_primary_devices(
                        std::vector<int>{device_id}
                    );
                }

                auto ptr_setup_future
                    = sclx::execute_kernel([=](sclx::kernel_handler& handler) {
                          sclx::array_list<T*, 1, 2> ptrs
                              = {A_ptr, A_inv_ptr};

                          handler.launch(
                              sclx::md_range_t<1>(A_ptr.shape()),
                              ptrs,
                              [=] __device__(sclx::md_index_t<1> & idx, auto&) {
                                  A_ptr[idx] = &A_copy(0, 0, idx[0]);
                                  A_inv_ptr[idx] = &A_inv(0, 0, idx[0]);
                              }
                          );
                      });

                copy_future.get();
                ptr_setup_future.get();
                A_copy.unset_read_mostly();
                auto status_getrf = detail::cublas_getrf_batched<T>::compute(
                    handle,
                    dims,
                    A_ptr.data().get(),
                    dims,
                    pivot.data().get(),
                    info.data().get(),
                    batch_size
                );

                A_inv.unset_read_mostly();
                auto status_getri = detail::cublas_getri_batched<T>::compute(
                    handle,
                    dims,
                    A_ptr.data().get(),
                    dims,
                    pivot.data().get(),
                    A_inv_ptr.data().get(),
                    dims,
                    info.data().get(),
                    batch_size
                );

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

                sclx::cuda::stream_synchronize();
                raw_problem_ptr->promise.set_value();
                raw_problem_ptr->is_task_submitted.store(false);
            } else {
                std::this_thread::yield();
            }
        }
        raw_problem_ptr->is_running.store(false);
    }

    batched_matrix_inverse_executor() = default;
    batched_matrix_inverse_executor(const batched_matrix_inverse_executor&) = delete;
    batched_matrix_inverse_executor& operator=(const batched_matrix_inverse_executor&) = delete;
    batched_matrix_inverse_executor(batched_matrix_inverse_executor&&) = default;
    batched_matrix_inverse_executor& operator=(batched_matrix_inverse_executor&&) = default;

  private:

    void init_thread(int device_id) {
        problem_definition_ = std::make_unique<problem_definition>();
        problem_definition* raw_problem_ptr = problem_definition_.get();
        std::thread([raw_problem_ptr, device_id](){
            thread_function(raw_problem_ptr, device_id);
        }).detach();
        while (!problem_definition_->is_running.load()) {
            std::this_thread::yield();
        }
    }

    static std::vector<batched_matrix_inverse_executor>* create_executors() {
        auto* executors = new std::vector<batched_matrix_inverse_executor>(sclx::cuda::traits::device_count());
        for (int device_id = 0; device_id < sclx::cuda::traits::device_count(); ++device_id) {
            (*executors)[device_id].init_thread(device_id);
        }
        return executors;
    }

    static batched_matrix_inverse_executor& get_executor_for_device(int device_id) {
        static std::unique_ptr<std::vector<batched_matrix_inverse_executor>> executors(create_executors());
        return (*executors)[device_id];
    }
    std::unique_ptr<problem_definition> problem_definition_{nullptr};
};
}

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


        auto A_slice
            = A.get_range({slice_start}, {slice_start + slice_len});
        auto A_inv_slice
            = A_inv.get_range({slice_start}, {slice_start + slice_len});

        auto fut = detail::batched_matrix_inverse_executor<T>::submit(
            device_id,
            A_slice,
            A_inv_slice,
            copy_A,
            current
        );
        futures.push_back(std::move(fut));
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
