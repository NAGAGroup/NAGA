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
// Created by jack on 2/7/23.
//

#pragma once

#include <naga/interface/device_accessor.cuh>

namespace naga::uvm_runtime {
template<class T>
struct device_accessor
    : interface::device_accessor<T, device_accessor<T>> {

    __device__ const T& read_impl(size_t index) const {
        static T value{};
        printf("Reading in uvm_device_accessor\n");
        return value;
    }

    __device__ bool write_impl(size_t index, const T& value) {
        printf("Writing in uvm_device_accessor\n");
        return true;
    }

    __host__ std::future<void> open_impl() {
        return std::async(std::launch::async, []() {
            //...
        });
    }

    __host__ std::future<void> close_impl() {
        return std::async(std::launch::async, []() {
            //...
        });
    }
};
}  // namespace naga::uvm_runtime