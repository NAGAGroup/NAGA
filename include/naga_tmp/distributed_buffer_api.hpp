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

enum class access_specifier
{
    read_only,
    write_only,
    read_write
};

template<class T>
struct device_accessor;

template <class T>
struct device_hash_map {
    bool is_one_to_one_{};
    T* data_{};
    device_accessor<T> *accessor_{};

    // computes the index to data_ for a given key
    // will also flag to the device_accessor that incorrect
    // access was made if the key is not in the map
    int compute_hash(int key) const;

    T& operator[](const int& key) {
        if (is_one_to_one_) {
            return data_[key];
        } else {
            return data_[compute_hash(key)];
        }
    }
};

template<class T>
struct device_accessor {
    access_specifier specifier;
    device_hash_map<T> extern_data_map;
    bool attempted_incorrect_access = false;
    T* local_data;
    size_t local_data_start_index;
    size_t local_data_end_index;

    bool can_write_to_index(size_t index) const;

    void write(const int& key, const T& value) {
        if (!can_write_to_index(key)) {
            attempted_incorrect_access = true;
            return ;
        }

        if (key >= local_data_start_index && key < local_data_end_index) {
            local_data[key - local_data_start_index] = value;
        } else {
            extern_data_map[key] = value;
        }
    }

    bool can_read_from_index(size_t index) const;

    T& read(const int& key) {
        if (!can_read_from_index(key)) {
            attempted_incorrect_access = true;
            return local_data[0];
        }
        if (key >= local_data_start_index && key < local_data_end_index) {
            return local_data[key - local_data_start_index];
        } else {
            return extern_data_map[key];
        }
    }
};

// used to create per-device accessors when the kernel is ready to be launched
// the kernel launcher will iterate across all devices and make a call to
// get get_accessor for the device.
//
// it also contains locks indicating whether the data is in a state where
// it cannot be accessed by the kernel and will pause execution until
// the data is ready
//
// it also manages the transfer of data between devices for reads and creates
// hash maps if the user provides a stencil for the data to be read
//
// writes can only be made to the data that the device owns
//
// often times the stencil pattern doesn't change between kernel launches, even
// though the data may change.  This class can be cached in a kernel launcher class
// and reused between kernel launches to avoid the overhead of calling
// allocation functions and creating hash maps
template <class T>
struct device_accessor_proxy {
    device_accessor<T> get_accessor(int device);
};

// a buffer containing data that is shared across multiple devices
//
// when constructed, a user will specifiy some or all of the following, not
// a comprehensive list:
// - the number of devices that will operate on the data
// - the number of elements in the data
// - which devices will operate on the data
//
// this buffer will be used to get accessors proxies for kernel launches
// the api for getting accessors is not yet defined, but will likely need
// the following:
// - the access specifier for the accessor
// - if reading, the stencil pattern for the data
template<class T>
struct buffer{};

// example of a kernel that reads from +/-1 of the current index from one array,
// sums the values, and writes the result to another array
//
// note that there is an index offset. this is provided by the kernel launcher
// and is determined by the problem size specified by the user and the number
// of devices that will be used to execute the kernel. also, some kernels
// may not need an index offset when the user specifies that the full problem
// size will be used by all devices. an example of this is when a mask is
// provided to only operate on a subset of the data. in this case, it's often
// beneficial to provide the mask as a read-only array to the kernel and
// read every index from the mask on each device. then a check can be provided
// in the kernel to only operate on the data if the mask index is writable
// for that device (example further below)
__global__ void sum_and_write(device_accessor<int> read_array,
                        device_accessor<int> write_array,
                       int offset_from_current_index) {
     int index = blockIdx.x * blockDim.x + threadIdx.x + offset_from_current_index;
     int sum = read_array.read(index - 1) + read_array.read(index + 1);
     write_array.write(index, sum);
}

__global__ void sum_and_write_from_mask(device_accessor<int> read_array,
                        device_accessor<int> write_array,
                        device_accessor<int> mask,
                        int offset_from_current_index) {
     int kernel_index = blockIdx.x * blockDim.x + threadIdx.x + offset_from_current_index;
     int index = mask.read(kernel_index);
     if (!write_array.can_write_to_index(index)) {
         return;
     }
     int sum = read_array.read(index - 1) + read_array.read(index + 1);
     write_array.write(index, sum);
}

template<uint ...>
struct index_sequence {};

template<uint N, uint ...S>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, S...> {};

template<uint ...S>
struct make_index_sequence<0, S...> : index_sequence<S...> {};

template<class T>
struct argtype {
     using type = T;
};

template<class T>
struct argtype<device_accessor_proxy<T>> {
     using type = argtype<device_accessor<T>>;
};

template <class T>
T process_arg(const T& arg) {
    return arg;
}

template <class T>
device_accessor_proxy<T> process_arg(const device_accessor<T>& arg) {
    return arg.get_accessor();
}

template<auto F, class ... Args>
struct executor_impl {
     std::tuple<typename argtype<Args>::type...> args;
    explicit executor_impl(Args... args) : args(args...) {}
};

