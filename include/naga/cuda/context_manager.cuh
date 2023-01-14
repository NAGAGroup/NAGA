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

namespace naga::cuda {

class context_manager {
  public:
    context_manager()                                  = delete;
    context_manager(const context_manager&)            = delete;
    context_manager(context_manager&&)                 = delete;
    context_manager& operator=(const context_manager&) = delete;
    context_manager& operator=(context_manager&&)      = delete;

    static constexpr int cpu_device_id         = cudaCpuDeviceId;
    static constexpr int distributed_device_id = cudaCpuDeviceId - 1;

    __host__ static int get_device_count() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        return device_count;
    }

    __host__ static int get_device() {
        int device;
        cudaGetDevice(&device);
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
        cuda_error(cudaSetDevice(device))
            .raise_if_error(
                std::string(NAGA_PRETTY_FUNCTION) + " failed with error: "
            );
        return old_device;
    }

    __host__ static void device_reset(int device) {
        int current_device = set_device(device);
        cuda_error(cudaDeviceReset())
            .raise_if_error(
                std::string(NAGA_PRETTY_FUNCTION) + " failed with error: "
            );
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