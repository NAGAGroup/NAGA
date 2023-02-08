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

#include <memory>

namespace naga::detail {

struct host_accessor_impl{
    virtual void* get(size_t index, size_t type_size) = 0;
    virtual const void* get(size_t index, size_t type_size) const = 0;
};

template <class T>
struct host_accessor {
    T& operator[](size_t index) {
        return *static_cast<T*>(accessor_impl_->get(index, sizeof(T)));
    }
    const T& operator[](size_t index) const {
        return *static_cast<const T*>(accessor_impl_->get(index, sizeof(T)));
    }

    std::shared_ptr<host_accessor_impl> accessor_impl_;
};

}