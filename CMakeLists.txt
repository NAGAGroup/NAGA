cmake_minimum_required(VERSION 3.24)
project(NAGA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include_directories(./include)

add_library(NAGA INTERFACE include/naga/cuda/cuda.hpp include/naga/cuda/errors.hpp)

add_subdirectory(tests)
