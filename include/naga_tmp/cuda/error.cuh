// Copyright 2023 Jack Myers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Created by jack on 1/12/23.
//

#pragma once

#include <stdexcept>

namespace naga::cuda {

class cuda_exception : public std::runtime_error {
  public:
    explicit cuda_exception(const std::string& what_arg)
        : std::runtime_error(what_arg) {}
    explicit cuda_exception(const char* what_arg)
        : std::runtime_error(what_arg) {}
};

class cuda_error {
  public:
    __host__ cuda_error() : error_code_(cudaSuccess) {}

    __host__ explicit cuda_error(const cudaError_t& error)
        : error_code_(error) {}

    __host__ bool success() const { return error_code_ == cudaSuccess; }

    __host__ void raise_if_error(const std::string& msg_prefix = "") const {
        if (!success()) {
            throw cuda_exception(msg_prefix + to_string());
        }
    }

    __host__ std::string to_string() const {
        return {cudaGetErrorString(error_code_)};
    }

    __host__ const cudaError_t& get() const { return error_code_; }

  private:
    cudaError_t error_code_;
};

__host__ inline cuda_error get_last_error() {
    return cuda_error{cudaGetLastError()};
}

__host__ inline cuda_error peek_last_error() {
    return cuda_error{cudaPeekAtLastError()};
}

}  // namespace naga::cuda