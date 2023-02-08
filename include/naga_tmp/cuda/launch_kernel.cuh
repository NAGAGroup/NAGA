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
// Created by jack on 1/19/23.
//

#pragma once
#include "runtime.cuh"
#include <future>
#include <sstream>

namespace naga::cuda {

enum class partition_type { partitioned, not_partitioned };

namespace detail {
template<class T, size_t N>
__device__ T& static_shared_mem_get(size_t index) {
    __shared__ T data[N];
    return data[index];
}

__device__ void* dynamic_shared_mem_get(size_t byte_offset) {
    extern __shared__ char data[];
    return data + byte_offset;
}
}  // namespace detail

template<class T, size_t N>
struct static_shared_mem_t {
    __device__ T& operator[](size_t index) {
        return detail::static_shared_mem_get<T, N>(index);
    }
};

template<class T>
struct dynamic_shared_mem_t {
    __device__ T& operator[](size_t index) { return data[index]; }
    T* data;
};

class launch_handle {
    size_t problem_size_;
    size_t local_problem_size_;
    uint launch_idx_;
    uint num_launches_;
    partition_type type_;
    size_t shared_mem_offset_{};

  public:
    __host__ launch_handle(
        size_t problem_size,
        size_t local_problem_size,
        uint launch_idx,
        uint num_launches,
        partition_type type
    )
        : problem_size_(problem_size),
          local_problem_size_(local_problem_size),
          launch_idx_(launch_idx),
          num_launches_(num_launches),
          type_(type) {}

    __host__ __device__ launch_handle() = delete;

    __device__ void sync_threads() const { __syncthreads(); }

    __device__ void sync_warp() const { __syncwarp(); }

    template<class T, size_t N>
    __device__ static_shared_mem_t<T, N> static_shared_mem() {
        return {};
    }

    template<class T>
    __device__ dynamic_shared_mem_t<T> dynamic_shared_mem(size_t num_elements) {
        auto ptr = detail::dynamic_shared_mem_get(shared_mem_offset_);
        shared_mem_offset_ += num_elements * sizeof(T);
        return {reinterpret_cast<T*>(ptr)};
    }

    __device__ const size_t& global_problem_size() const {
        return problem_size_;
    }

    __device__ const size_t& local_problem_size() const {
        return local_problem_size_;
    }

    __device__ size_t global_thread_id() const {
        if (type_ == partition_type::partitioned) {
            return launch_idx_ * local_problem_size_ + blockIdx.x * blockDim.x
                 + threadIdx.x;
        } else {
            return launch_thread_id();
        }
    }

    __device__ size_t launch_thread_id() const {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ uint block_thread_id() const { return threadIdx.x; }

    __device__ uint block_id() const { return blockIdx.x; }

    __device__ const uint& launch_id() const { return launch_idx_; }

    __device__ const uint& total_launch_count() const { return num_launches_; }

    __device__ uint grid_size() const { return gridDim.x; }

    __device__ uint block_size() const { return blockDim.x; }

    __device__ uint threads_per_launch() const {
        return grid_size() * block_size();
    }

    __device__ size_t launch_range_start() const {
        if (type_ == partition_type::partitioned) {
            return launch_idx_ * local_problem_size_;
        } else {
            return 0;
        }
    }

    __device__ size_t launch_range_end() const {
        if (type_ == partition_type::partitioned) {
            return (launch_idx_ + 1) * local_problem_size_;
        } else {
            return problem_size_;
        }
    }

    __device__ bool is_in_launch_range(size_t idx) const {
        return idx >= launch_range_start() && idx < launch_range_end();
    }

    __device__ bool is_in_problem_range(size_t idx) const {
        return idx < problem_size_;
    }

    __device__ size_t recommended_write_range_start() const {
        return (problem_size_ + num_launches_ - 1) / num_launches_
             * launch_idx_;
    }

    __device__ size_t recommended_write_range_end() const {
        return (problem_size_ + num_launches_ - 1) / num_launches_
             * (launch_idx_ + 1);
    }

    __device__ bool is_in_recommended_write_range(size_t idx) const {
        return idx >= recommended_write_range_start()
            && idx < recommended_write_range_end();
    }
};

// enum class execution_policy {
//     async,
//     sync
// };
struct launch_config {
    size_t problem_size{};
    partition_type type{partition_type::not_partitioned};
    std::vector<int> device_ids{0};
    uint block_size{256};
    uint max_grid_size{std::numeric_limits<uint>::max()};
    uint shared_mem_per_launch{0};
};

struct launch_error : public std::runtime_error {
    explicit launch_error(const std::string& msg) : std::runtime_error(msg) {}

    static std::string get_error_string(
        const std::vector<std::tuple<int, cudaStream_t, cudaError_t>>& errors
    ) {
        std::stringstream ss;
        ss << "cuda launch failed on " << errors.size()
           << " devices: " << std::endl;
        for (auto& [device_id, stream, error] : errors) {
            ss << "    Device " << device_id << " failed on stream " << stream
               << " with '" << cudaGetErrorString(error) << "'" << std::endl;
        }
        return ss.str();
    }

    static launch_error augment_with_caller(
        const launch_error& error,
        const std::string& calling_function
    ) {
        std::stringstream ss;
        ss << calling_function << " -- " << error.what();
        return launch_error(ss.str());
    }
};

enum class execution_policy {
    sync,
    async,
};

template<auto Kernel, class... Args>
__host__ std::future<void> launch_kernel(
    execution_policy policy,
    const launch_config& config,
    Args&&... args
) {
    size_t local_problem_size = config.problem_size;
    int num_launches          = static_cast<int>(config.device_ids.size());
    if (config.type == partition_type::partitioned) {
        local_problem_size
            = (config.problem_size + num_launches - 1) / num_launches;
    }

    // we use a min here because if the recommended grid size doesn't fit
    // into a 32 bit int, then it is for sure too big for CUDA
    size_t recommended_grid_size = std::min(
        (local_problem_size + config.block_size - 1) / config.block_size,
        static_cast<size_t>(std::numeric_limits<uint>::max())
    );
    uint grid_size = std::min(
        static_cast<uint>(recommended_grid_size),
        config.max_grid_size
    );

    std::future<void> fut = std::async(std::launch::async, [=]() {
        std::vector<cudaStream_t> streams(num_launches);
        int current_device = runtime::get_device();
        for (int i = 0; i < num_launches; ++i) {
            runtime::set_device(config.device_ids[i]);
            if (!runtime::success(cudaStreamCreate(&streams[i]))) {
                throw runtime_error(
                    runtime_error::get_error_string(cudaGetLastError())
                );
            }
        }
        for (int i = 0; i < num_launches; ++i) {
            runtime::set_device(config.device_ids[i]);
            Kernel<<<
                grid_size,
                config.block_size,
                config.shared_mem_per_launch,
                streams[i]>>>(
                launch_handle(
                    config.problem_size,
                    local_problem_size,
                    i,
                    num_launches,
                    config.type
                ),
                args...
            );
        }
        std::vector<std::tuple<int, cudaStream_t, cudaError_t>> errors;
        for (int i = 0; i < num_launches; ++i) {
            runtime::set_device(config.device_ids[i]);
            if (!runtime::success(cudaStreamSynchronize(streams[i]))) {
                errors.emplace_back(
                    config.device_ids[i],
                    streams[i],
                    cudaGetLastError()
                );
            }
            if (!runtime::success(cudaStreamDestroy(streams[i]))) {
                runtime_error error(
                    runtime_error::get_error_string(cudaGetLastError())
                );
            }
        }
        if (!errors.empty()) {
            throw launch_error(launch_error::get_error_string(errors));
        }
        cudaSetDevice(current_device);
    });

    if (policy == execution_policy::sync) {
        fut.wait();
    }

    return fut;
}

}  // namespace naga::cuda
