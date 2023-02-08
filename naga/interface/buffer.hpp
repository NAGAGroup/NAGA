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

#include <naga/detail/buffer.hpp>
#include <naga/detail/host_accessor.hpp>
#include "runtime_traits.cuh"

namespace naga::interface {

template <class T, class Derived, class Runtime>
struct buffer : detail::buffer {
    std::string get_runtime() const final {
        return static_cast<const Derived*>(this)->get_runtime_impl();
    }

    auto get_device_accessor() -> runtime_traits<Runtime>::template device_accessor_type<T> {
        return static_cast<Derived*>(this)->get_host_accessor_impl();
    }

    detail::host_accessor<T> get_host_accessor() {
        return static_cast<Derived*>(this)->get_host_accessor_impl();
    }
};

}