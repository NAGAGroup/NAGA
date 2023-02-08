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

#include <naga/detail/device_accessor.hpp>

namespace naga {

namespace interface {

template<class T, class Derived>
struct device_accessor : detail::device_accessor {
    __device__ const T& read(size_t index) const {
        return static_cast<const Derived*>(this)->read_impl(index);
    }

    __device__ bool write(size_t index, const T& value) {
        return static_cast<Derived*>(this)->write_impl(index);
    }

    __host__ std::future<void> open() final {
        return static_cast<Derived*>(this)->open_impl();
    }

    __host__ std::future<void> close() final {
        return static_cast<Derived*>(this)->close_impl();
    }

    using value_type = T;
};
}  // namespace interface

}  // namespace naga
