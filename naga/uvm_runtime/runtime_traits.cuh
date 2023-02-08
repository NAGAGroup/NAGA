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

#include "buffer.hpp"
#include "device_accessor.cuh"
#include <naga/interface/runtime_traits.cuh>

namespace naga {

namespace uvm_runtime {
struct uvm_runtime;
}

template <>
struct runtime_traits<uvm_runtime::uvm_runtime> {
    template<class T>
    using device_accessor_type = uvm_runtime::device_accessor<T>;

    template<class T>
    using buffer_type = uvm_runtime::buffer<T>;
};

}