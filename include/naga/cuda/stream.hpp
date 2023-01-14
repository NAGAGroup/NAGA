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
#include "context_manager.hpp"
#include <vector>
#include <memory>

namespace naga::cuda {
class cuda_stream {
  public:
    __host__ cuda_stream() = delete;
    __host__ cuda_stream(cudaStream_t stream, int device_id) : stream_(stream), device_id_(device_id) {}

    __host__ static cuda_stream create() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return {stream, context_manager::get_device()};
    }

    __host__ static cuda_stream create(int device_id) {
        int old_device = context_manager::set_device(device_id);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        context_manager::set_device(old_device);
        return {stream, device_id};
    }

    __host__ static cuda_stream default_stream() {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
        return {stream, context_manager::get_device()};
    }

    __host__ static cuda_stream default_stream(int device_id) {
        int old_device = context_manager::set_device(device_id);
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
        context_manager::set_device(old_device);
        return {stream, device_id};
    }

    __host__ static std::vector<cuda_stream> create_streams(int count) {
        std::vector<cuda_stream> streams;
        streams.reserve(count);
        for (int i = 0; i < count; i++) {
            streams.push_back(create());
        }
        return streams;
    }

    __host__ static std::vector<cuda_stream> create_streams(int count, int device_id) {
        std::vector<cuda_stream> streams;
        streams.reserve(count);
        for (int i = 0; i < count; i++) {
            streams.push_back(create(device_id));
        }
        return streams;
    }

    __host__ int prepare_for_launch() const {
        return context_manager::set_device(device_id_);
    }

    __host__ bool is_done() const {
        int current_device = context_manager::get_device();
        context_manager::set_device(device_id_);
        bool done = cudaStreamQuery(stream_) == cudaSuccess;
        context_manager::set_device(current_device);
        return done;
    }

    __host__ cuda_error synchronize() const {
        int old_device = context_manager::set_device(device_id_);
        cuda_error error = cuda_error(cudaStreamSynchronize(stream_));
        context_manager::set_device(old_device);
        return error;
    }

    __host__ const cudaStream_t &get() const { return stream_; }
    __host__ const int& associated_device_id() const { return device_id_; }

    __host__ ~cuda_stream() {
        int old_device = context_manager::set_device(device_id_);
        cudaStreamDestroy(stream_);
        context_manager::set_device(old_device);
    }

  private:
    cudaStream_t stream_;
    int device_id_;
};

} // namespace naga::cuda
