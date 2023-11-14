
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
#include <cusparse.h>
#include <scalix/cuda.hpp>

namespace naga::detail::cusparse {

class handle_t {
  public:
    const cusparseHandle_t & get() const { return handle_; }

    static handle_t create() {
        handle_t handle;
        int current_device = sclx::cuda::traits::current_device();
        auto error         = cusparseCreate(&(handle.handle_));
        if (error != CUSPARSE_STATUS_SUCCESS) {
            sclx::throw_exception<std::runtime_error>(
                "cusparseCreate failed with error code " + std::to_string(error),
                "naga::detail::cusparse::handle_t::"
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
    operator cusparseHandle_t() const { return get(); }

    ~handle_t() {
        if (handle_ != nullptr) {
            int current_device = sclx::cuda::traits::current_device();
            if (current_device != device_) {
                sclx::cuda::set_device(device_);
            }
            cusparseDestroy(handle_);
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
    cusparseHandle_t handle_{};
    int device_{};
};

class gemm_desc_t {
  public:
    const cusparseSpGEMMDescr_t & get() const { return desc_; }

    static gemm_desc_t create() {
        gemm_desc_t handle;
        int current_device = sclx::cuda::traits::current_device();
        auto error         = cusparseSpGEMM_createDescr(&(handle.desc_));
        if (error != CUSPARSE_STATUS_SUCCESS) {
            sclx::throw_exception<std::runtime_error>(
                "cusparseCreate failed with error code " + std::to_string(error),
                "naga::detail::cusparse::gemm_desc_t::"
            );
        }
        handle.device_ = current_device;
        return handle;
    }

    static gemm_desc_t create_for_device(int device) {
        int current_device = sclx::cuda::traits::current_device();
        sclx::cuda::set_device(device);
        auto handle = create();
        sclx::cuda::set_device(current_device);
        return handle;
    }

    // allow implicit conversion to cusparseSpGEMMDescr_t
    operator cusparseSpGEMMDescr_t() const { return get(); }

    ~gemm_desc_t() {
        if (desc_ != nullptr) {
            int current_device = sclx::cuda::traits::current_device();
            if (current_device != device_) {
                sclx::cuda::set_device(device_);
            }
            cusparseSpGEMM_destroyDescr(desc_);
        }
    }

    gemm_desc_t() = default;

    gemm_desc_t(const gemm_desc_t&)            = delete;
    gemm_desc_t& operator=(const gemm_desc_t&) = delete;

    gemm_desc_t(gemm_desc_t&& other) noexcept {
        desc_       = other.desc_;
        other.desc_ = nullptr;
    }

    gemm_desc_t& operator=(gemm_desc_t&& other) noexcept {
        desc_       = other.desc_;
        other.desc_ = nullptr;
        return *this;
    }

  private:
    cusparseSpGEMMDescr_t desc_{};
    int device_{};
};

}
