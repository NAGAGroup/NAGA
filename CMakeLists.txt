cmake_minimum_required(VERSION 3.24)
project(NAGA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

#include_directories(./include)
#
#add_library(
#  NAGA
#  INTERFACE
#  include/naga/cuda/cuda.cuh
#  include/naga/cuda/error.cuh
#  include/naga/defines.h
#        include/naga/cuda/runtime.cuh
#  include/naga/cuda/stream.cuh
#  include/naga/cuda/synchronization.cuh
#  include/naga/cuda/kernel_dispatch.cuh include/naga/stripped_array.cuh include/naga/dim4.cuh include/naga/array.cuh include/naga/distributed_buffer_api.hpp include/naga/cuda/launch_kernel.cuh)
#
#add_subdirectory(tests)

add_executable(sample sample.cu)
target_include_directories(sample PRIVATE .)
