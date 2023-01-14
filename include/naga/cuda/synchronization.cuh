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
#include "stream.cuh"
#include <vector>

namespace naga::cuda {

__host__ cuda_error inline synchronize() {
    return cuda_error{cudaDeviceSynchronize()};
}

__host__ cuda_error inline synchronize_device(int device) {
    int old_device = context_manager::set_device(device);
    cuda_error error = synchronize();
    context_manager::set_device(old_device);
    return error;
}

__host__ std::vector<cuda_error> inline synchronize_all_devices() {
    std::vector<cuda_error> errors;
    errors.reserve(context_manager::get_device_count());
    for (int i = 0; i < context_manager::get_device_count(); i++) {
        errors.push_back(synchronize_device(i));
    }
    return errors;
}

__host__ cuda_error inline synchronize_streams(const std::vector<stream> &streams) {
    cuda_error error;
    for (const stream &stream : streams) {
        cuda_error sync_error = stream.synchronize();
        if (error.success() && !sync_error.success()) {
            error = sync_error;
        }
    }
    return error;
}

} // namespace naga::cuda