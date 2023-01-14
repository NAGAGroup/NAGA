cmake_minimum_required(VERSION 3.24)
project(NAGA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include_directories(./include)

add_library(
  NAGA
  INTERFACE
  include/naga/cuda/cuda.hpp
  include/naga/cuda/error.hpp
  include/naga/defines.h
  include/naga/cuda/context_manager.hpp
  include/naga/cuda/stream.hpp)

add_subdirectory(tests)
