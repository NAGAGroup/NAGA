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

#ifdef NAGA_UVM_RUNTIME
#include "uvm_runtime/runtime.cuh"
namespace naga {
using runtime = uvm_runtime::uvm_runtime;
#endif

template <class T>
using device_accessor = typename runtime_traits<runtime>::template device_accessor_type<T>;

template <class T>
using host_accessor = typename detail::host_accessor<T>;

template <class T>
using buffer = typename runtime_traits<runtime>::template buffer_type<T>;
}