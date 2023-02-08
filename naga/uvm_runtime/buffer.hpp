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

#include <naga/interface/buffer.hpp>
#include "device_accessor.cuh"

namespace naga::uvm_runtime {

struct uvm_runtime;

template <class T>
struct buffer : interface::buffer<T, buffer<T>, uvm_runtime> {
    using runtime = uvm_runtime;

    std::string get_runtime_impl() const {
        return "uvm_runtime";
    }

    device_accessor<T> get_device_accessor_impl() {
        return device_accessor<T>();
    }

    detail::host_accessor<T> get_host_accessor_impl() {
        return detail::host_accessor<T>();
    }
};

}