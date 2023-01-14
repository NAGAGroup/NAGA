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

TEST_CASE("naga::cuda::cuda_error") {
    CHECK(naga::cuda::cuda_error(cudaSuccess).success());
    CHECK(!naga::cuda::cuda_error(cudaErrorMemoryAllocation).success());

    naga::cuda::cuda_error error(cudaErrorMemoryAllocation);
    CHECK_THROWS_AS(error.raise_if_error(), naga::cuda::cuda_exception);
    CHECK(error.to_string() == "out of memory");
    CHECK(error.get() == cudaErrorMemoryAllocation);

    CHECK(naga::cuda::cuda_error::get_last_error().success());

    cudaMalloc(nullptr, 0);
    error = naga::cuda::cuda_error::peek_last_error();
    CHECK(!error.success());
    CHECK_THROWS_AS(error.raise_if_error(), naga::cuda::cuda_exception);
    CHECK(error.to_string() == "invalid argument");

    error = naga::cuda::cuda_error::get_last_error();
    CHECK(!error.success());
    CHECK_THROWS_AS(error.raise_if_error(), naga::cuda::cuda_exception);
    CHECK(error.to_string() == "invalid argument");

    error = naga::cuda::cuda_error::get_last_error();
    CHECK(error.success());
    CHECK_NOTHROW(error.raise_if_error());
}

__global__ void kernel() {printf("Hello World!\\n");}

TEST_CASE("naga::cuda::context_manager") {
    using context_manager = naga::cuda::context_manager;

    float *device_ptr;
    size_t available, total;
    cudaMemGetInfo(&available, &total);
    cudaMalloc(&device_ptr, available / 3);
    context_manager::system_reset();
    size_t available_after_reset, total_after_reset;
    cudaMemGetInfo(&available_after_reset, &total_after_reset);
    CHECK(available == available_after_reset);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    CHECK(naga::cuda::context_manager::get_device_count() == device_count);

    CHECK_THROWS_AS(
        context_manager::set_device(100),
        naga::cuda::cuda_exception
    );
    CHECK_THROWS_WITH(
        context_manager::set_device(100),
        "static int naga::cuda::context_manager::set_device(int) failed with "
        "error: invalid device ordinal"
    );

    CHECK_NOTHROW(context_manager::set_device(0));
    CHECK(context_manager::get_device() == 0);

    CHECK_THROWS_AS(context_manager::set_device(context_manager::cpu_device_id), std::invalid_argument);
    CHECK_THROWS_WITH(context_manager::set_device(context_manager::cpu_device_id), "static int naga::cuda::context_manager::set_device(int): cpu_device_id and distributed_device_id are not valid device ids");

    CHECK_THROWS_AS(context_manager::set_device(context_manager::distributed_device_id), std::invalid_argument);
    CHECK_THROWS_WITH(context_manager::set_device(context_manager::distributed_device_id), "static int naga::cuda::context_manager::set_device(int): cpu_device_id and distributed_device_id are not valid device ids");
}