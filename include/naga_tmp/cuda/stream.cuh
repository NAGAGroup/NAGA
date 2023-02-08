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
#include "runtime.cuh"
#include <memory>
#include <vector>

namespace naga::cuda {
class stream {
  public:
    __host__ stream(cudaStream_t stream, int device_id) {
        auto* stream_copy_ptr = new cudaStream_t;
        *stream_copy_ptr      = stream;
        this->raw_stream_ptr_
            = std::shared_ptr<cudaStream_t>(stream_copy_ptr, stream_deleter);
        this->device_id_ = device_id;
    }

    __host__ stream() = default;

    __host__ static stream create(int device_id) {
        int current_device = runtime::set_device(device_id);
        auto* stream_ptr   = new cudaStream_t;
        cuda_error(cudaStreamCreate(stream_ptr))
            .raise_if_error("naga::stream_t creation failed.");
        runtime::set_device(current_device);
        return {*stream_ptr, runtime::get_device()};
    }

    __host__ static stream create() {
        return create(runtime::get_device());
    }

    __host__ static stream default_stream() {
        int device_id = runtime::get_device();
        return {cudaStreamDefault, device_id};
    }

    __host__ static stream default_stream(int device_id) {
        return {cudaStreamDefault, device_id};
    }

    __host__ cudaStream_t& get() { return *raw_stream_ptr_; }

    __host__ const cudaStream_t& get() const { return *raw_stream_ptr_; }

    __host__ const int& associated_device() const { return device_id_; }

    __host__ int prepare() const {
        return runtime::set_device(device_id_);
    }

    __host__ cuda_error synchronize() const {
        int current_device = runtime::get_device();
        runtime::set_device(device_id_);
        cuda_error err{cudaStreamSynchronize(*raw_stream_ptr_)};
        runtime::set_device(current_device);
        return err;
    }

    __host__ bool ready() const {
        int current_device = runtime::get_device();
        runtime::set_device(device_id_);
        bool is_ready = cudaStreamQuery(*raw_stream_ptr_) == cudaSuccess;
        runtime::set_device(current_device);
        return is_ready;
    }

    __host__ static std::vector<stream>
    create_streams(uint num_streams, int device_id) {
        int current_device = runtime::get_device();
        runtime::set_device(device_id);
        std::vector<stream> result;
        for (uint i = 0; i < num_streams; ++i) {
            result.emplace_back(create(device_id));
        }
        runtime::set_device(current_device);
        return result;
    }

    __host__ static std::vector<stream> create_streams(uint num_streams) {
        return create_streams(num_streams, runtime::get_device());
    }

  private:
    std::shared_ptr<cudaStream_t> raw_stream_ptr_
        = std::make_shared<cudaStream_t>(
            static_cast<cudaStream_t>(cudaStreamDefault)
        );
    int device_id_{};

    static constexpr auto stream_deleter = [](cudaStream_t* stream_ptr) {
        if (*stream_ptr != cudaStreamDefault)
            cudaStreamDestroy(*stream_ptr);
        delete stream_ptr;
    };
};

}  // namespace naga::cuda
