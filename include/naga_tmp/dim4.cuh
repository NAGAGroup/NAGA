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
#include <stdexcept>


namespace naga {

struct dim4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;

    __host__ __device__ dim4() : x(0), y(0), z(0), w(0) {}

    __host__ __device__ dim4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
        : x(x), y(y), z(z), w(w) {}

    __host__ __device__ dim4(unsigned int x, unsigned int y, unsigned int z) : x(x), y(y), z(z), w(1) {}

    __host__ __device__ dim4(unsigned int x, unsigned int y) : x(x), y(y), z(1), w(1) {}

    __host__ __device__ dim4(unsigned int x) : x(x), y(1), z(1), w(1) {}

    __host__ __device__ dim4(const dim4& other) = default;

    __host__ __device__ dim4& operator=(const dim4& other) = default;

    __host__ __device__ dim4(dim4&& other) noexcept = default;

    __host__ __device__ dim4& operator=(dim4&& other) noexcept = default;

    __host__ __device__ ~dim4() = default;

    __host__ __device__ unsigned int operator[](unsigned int i) const {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                throw std::out_of_range("dim4 index out of range");
        }
    }

    __host__ __device__ unsigned int& operator[](unsigned int i) {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                throw std::out_of_range("dim4 index out of range");
        }
    }
};

}