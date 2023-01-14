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
// Created by jack on 1/13/23.
//

#pragma once
#include "stream.cuh"
#include <vector>

#ifdef __CLION_IDE__
__global__ void dummy_kernel(int, int, size_t) {}  // for development in CLion
#endif

namespace naga::cuda {

enum class execution_policy {
    sync,
    async,
};

template<auto Kernel, class... Args>
__host__ stream inline kernel_stream_launch(
    uint grid_size,
    uint block_size,
    size_t shared_mem,
    const stream& stream,
    Args... args
) {
    int current_device = stream.prepare();
    Kernel<<<grid_size, block_size, shared_mem, stream.get()>>>(args...);
    context_manager::set_device(current_device);
    return stream;
}

template<auto Kernel, class... Args>
__host__ stream inline kernel_launch(
    execution_policy policy,
    uint grid_size,
    uint block_size,
    size_t shared_mem,
    int device_id,
    Args... args
) {
    const stream& kernel_stream = stream::create(device_id);
    //    if (policy == execution_policy::async) {
    //        kernel_stream = stream::create(device_id);
    //    } else {
    //        kernel_stream = stream::default_stream(device_id);
    //    }
    kernel_stream_launch<Kernel, Args...>(
        grid_size,
        block_size,
        shared_mem,
        kernel_stream,
        args...
    );
    if (policy == execution_policy::sync) {
        kernel_stream.synchronize();
    }
    return kernel_stream;
}

/** Distributes a problem across all available CUDA devices.
 *
 * Every kernel must accept as its first three arguments:
 * - int device_count (the number of devices)
 * - int device_id (the id of the current device)
 * - size_t device_problem_size (the size of the problem for each device)
 *
 * The reasoning for this is that kernels often need this metadata to
 * properly distribute work across devices.
 *
 * @tparam Kernel The kernel to launch.
 * @tparam Args The types of the arguments to the kernel.
 * @param policy The execution policy to use.
 * @param total_problem_size The total size of the problem to distribute.
 * @param max_grid_size The maximum grid size to use for each kernel launch.
 * @param block_size The block size to use for each kernel launch.
 * @param shared_mem_per_device The amount of shared memory to use for each
 * kernel launch.
 * @param args The arguments to the kernel, excluding those passed by the kernel
 * launcher.
 * @return A vector of streams, one for each kernel launch.
 */
template<auto Kernel, class... Args>
__host__ std::vector<stream> inline distributed_kernel_launch(
    execution_policy policy,
    size_t total_problem_size,
    uint max_grid_size,
    uint block_size,
    size_t shared_mem_per_device,
    Args... args
) {
    std::vector<stream> streams;
    streams.reserve(context_manager::get_device_count());
    for (int dev = 0; dev < context_manager::get_device_count(); dev++) {
        size_t problem_size_per_device
            = (total_problem_size + context_manager::get_device_count() - 1)
            / context_manager::get_device_count();
        uint grid_size = std::min(
            max_grid_size,
            static_cast<uint>(problem_size_per_device + block_size - 1)
                / block_size
        );
        streams.push_back(kernel_launch<Kernel>(
            policy,
            grid_size,
            block_size,
            shared_mem_per_device,
            dev,
            context_manager::get_device_count(),
            dev,
            problem_size_per_device,
            args...
        ));
    }
    return streams;
}

#ifdef __CLION_IDE__
template std::vector<stream> distributed_kernel_launch<dummy_kernel>(
    execution_policy policy,
    size_t total_problem_size,
    uint block_size,
    uint grid_size,
    size_t shared_mem
);
#endif

}  // namespace naga::cuda
