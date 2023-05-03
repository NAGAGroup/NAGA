
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
#include <cublas_v2.h>
#include <memory>
#include <mutex>
#include <scalix/cuda.hpp>
#include <unordered_map>

namespace naga::detail::cublas {

class handle_t {
  public:
    const cublasHandle_t& get() const { return handle_; }

    static handle_t create() {
        handle_t handle;
        int current_device = sclx::cuda::traits::current_device();
        auto error         = cublasCreate(&(handle.handle_));
        if (error != CUBLAS_STATUS_SUCCESS) {
            sclx::throw_exception<std::runtime_error>(
                "cublasCreate failed with error code " + std::to_string(error),
                "naga::detail::cublas::handle_t::"
            );
        }
        handle.device_ = current_device;
        return handle;
    }

    static handle_t create_for_device(int device) {
        int current_device = sclx::cuda::traits::current_device();
        sclx::cuda::set_device(device);
        auto handle = create();
        sclx::cuda::set_device(current_device);
        return handle;
    }

    // allow implicit conversion to cublasHandle_t
    operator cublasHandle_t() const { return get(); }

    ~handle_t() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    handle_t() = default;

    handle_t(const handle_t&)            = delete;
    handle_t& operator=(const handle_t&) = delete;

    handle_t(handle_t&& other) noexcept {
        handle_       = other.handle_;
        other.handle_ = nullptr;
    }

    handle_t& operator=(handle_t&& other) noexcept {
        handle_       = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }

  private:
    cublasHandle_t handle_{};
    int device_{};
};

}  // namespace naga::detail::cublas
