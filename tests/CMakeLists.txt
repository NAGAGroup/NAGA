cmake_minimum_required(VERSION 3.24)
project(NAGA_TESTS CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include_directories(doctest/doctest)

enable_testing()

# CUDA_HPP_TESTS testing NAGA CPP API for CUDA
add_executable(CUDA_HPP_TESTS naga/cuda/cuda_tests.cu)
add_test(NAME CUDA_HPP_TESTS COMMAND CUDA_HPP_TESTS)
