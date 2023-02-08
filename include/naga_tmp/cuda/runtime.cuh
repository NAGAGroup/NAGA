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
// Created by jack on 1/13/23.
//

#pragma once
#include "error.cuh"
#include <naga/defines.h>
#include <sstream>

namespace naga::cuda {

struct runtime_error : public std::runtime_error {
    __host__ explicit runtime_error(const std::string& msg)
        : std::runtime_error(msg) {}

    __host__ static std::string get_error_string(cudaError_t error) {
        std::stringstream ss;
        ss << "cuda runtime failed with error: " << std::endl
           << "    " << cudaGetErrorString(error);
    }

    __host__ static runtime_error augment_with_caller(
        const runtime_error& e,
        const std::string& calling_function
    ) {
        std::stringstream ss;
        ss << calling_function << " -- " << e.what();
        return runtime_error(ss.str());
    }
};

class runtime {
  public:
    runtime()                          = delete;
    runtime(const runtime&)            = delete;
    runtime(runtime&&)                 = delete;
    runtime& operator=(const runtime&) = delete;
    runtime& operator=(runtime&&)      = delete;

    static constexpr int cpu_device_id         = cudaCpuDeviceId;
    static constexpr int distributed_device_id = cudaCpuDeviceId - 1;

    __host__ static bool success(cudaError_t error) {
        return error == cudaSuccess;
    }

    __host__ static int get_device_count() {
        int device_count;
        if (!success(cudaGetDeviceCount(&device_count))) {
            throw runtime_error(
                runtime_error::get_error_string(cudaGetLastError())
            );
        }
        return device_count;
    }

    __host__ static int get_device() {
        int device;
        if (!success(cudaGetDevice(&device))) {
            throw runtime_error(
                runtime_error::get_error_string(cudaGetLastError())
            );
        }
        return device;
    }

    __host__ static int set_device(int device) {
        int old_device = get_device();
        if (device == cpu_device_id || device == distributed_device_id) {
            throw std::invalid_argument(
                std::string(NAGA_PRETTY_FUNCTION)
                + ": cpu_device_id and distributed_device_id are not valid "
                  "device ids"
            );
        }
        if (!success(cudaSetDevice(device))) {
            throw runtime_error(
                runtime_error::get_error_string(cudaGetLastError())
            );
        }
        return old_device;
    }

    __host__ static void device_reset(int device) {
        int current_device = set_device(device);
        if (!success(cudaDeviceReset())) {
            throw runtime_error(
                runtime_error::get_error_string(cudaGetLastError())
            );
        }
        set_device(current_device);
    }

    __host__ static void device_reset() { device_reset(get_device()); }

    __host__ static void system_reset() {
        for (int i = 0; i < get_device_count(); i++) {
            device_reset(i);
        }
    }
};

}  // namespace naga::cuda