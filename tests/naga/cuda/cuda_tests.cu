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
// Created by jack on 1/12/23.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "naga/cuda/cuda.hpp"

__global__ void broken_kernel() {
    int* ptr = nullptr;
    *ptr = 0;
}

TEST_CASE("naga::cuda::cuda_error") {
    CHECK(naga::cuda::cuda_error(cudaSuccess).success());
    CHECK(!naga::cuda::cuda_error(cudaErrorMemoryAllocation).success());

    CHECK_THROWS_AS(naga::cuda::cuda_error(cudaErrorMemoryAllocation).raise_if_error(), naga::cuda::cuda_exception);

    CHECK(naga::cuda::cuda_error(cudaErrorMemoryAllocation).to_string() == "out of memory");

    CHECK(naga::cuda::cuda_error(cudaErrorMemoryAllocation).get() == cudaErrorMemoryAllocation);

    CHECK(naga::cuda::cuda_error::get_last_error().success());

    broken_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    auto error = naga::cuda::cuda_error::peek_last_error();
    CHECK(!error.success());
    CHECK_THROWS_AS((error = naga::cuda::cuda_error::get_last_error()).raise_if_error(), naga::cuda::cuda_exception);
    CHECK(error.to_string() == "unspecified launch failure");
}