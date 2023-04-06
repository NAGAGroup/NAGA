cmake_minimum_required(VERSION 3.20)
project(
  NAGA
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(extern/Scalix)

# naga is a header only library
add_library(naga INTERFACE)
target_include_directories(naga INTERFACE ${PROJECT_SOURCE_DIR}/.)
target_link_libraries(naga INTERFACE scalix cublas)

# naga examples
add_executable(rectangular_partitioner examples/rectangular_partitioner.cu)
target_link_libraries(rectangular_partitioner PRIVATE naga)

add_executable(batched_knn_search examples/batched_knn_search.cu)
target_link_libraries(batched_knn_search PRIVATE naga)

add_executable(batched_matrix_inversion examples/batched_matrix_inversion.cu)
target_link_libraries(batched_matrix_inversion PRIVATE naga)
