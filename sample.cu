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

#define NAGA_UVM_RUNTIME
#include <naga/runtime.cuh>
#include <iostream>
#include <typeinfo>

//namespace naga {
//
//class uvm_runtime;
//using runtime = uvm_runtime;
//
//// user-facing interface
//template<class Runtime, class Derived>
//struct gpu_task {
//    __host__ void prepare() {
//        static_cast<Derived*>(this)->prepare_impl();
//    }
//
//    __device__ void operator()() const {
//        static_cast<Derived*>(this)->operator_parens_impl();
//    }
//
//    __host__ void finish() const { static_cast<Derived*>(this)->finish_impl(); }
//};
//
//}  // namespace naga
//
//// recommended user definition of gpu task
//
//namespace something {
//namespace detail {
//
//template<class Runtime>
//struct my_task : naga::gpu_task<Runtime, my_task<Runtime>> {
//    using runtime = Runtime;
//    naga::device_accessor<float> input;
//
//    __host__ void prepare_impl() {
//        input.prepare().wait();
//    }
//
//    __device__ void operator_parens_impl() const {
//        //...
//    }
//
//    __host__ void finish_impl() const {
//        //...
//    }
//};
//}  // namespace detail
//
//using my_task = detail::my_task<naga::runtime>;
//
//}  // namespace something

int main() {
    naga::device_accessor<float> input;

    naga::buffer<float> buf;
    return 0;
}