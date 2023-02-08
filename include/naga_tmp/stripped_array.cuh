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
// Created by jack on 1/14/23.
//

#pragma once
#include "dim4.cuh"

namespace naga {

using index_t = dim4;

template <class T>
class array;

template <class T>
class stripped_array {
  public:
    using value_type = T;

  private:
    template<class>
    friend class stripped_array;

    template<class>
    friend class array;

    T* data_{};
    dim4 dims_{};

    __host__ __device__ stripped_array() = default;
};

}
