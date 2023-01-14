cmake_minimum_required(VERSION 3.24)
project(NAGA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include_directories(./include)

add_library(
  NAGA
  INTERFACE
  include/naga/cuda/cuda.cuh
  include/naga/cuda/error.cuh
  include/naga/defines.h
  include/naga/cuda/context_manager.cuh
  include/naga/cuda/stream.cuh
  include/naga/cuda/synchronization.cuh
  include/naga/cuda/kernel_dispatch.cuh)

add_subdirectory(tests)
