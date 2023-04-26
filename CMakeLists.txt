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
add_executable(nd_cubic_segmentation examples/nd_cubic_segmentation.cu)
target_link_libraries(nd_cubic_segmentation PRIVATE naga)

add_executable(batched_knn_search examples/batched_knn_search.cu)
target_link_libraries(batched_knn_search PRIVATE naga)

add_executable(batched_matrix_inversion examples/batched_matrix_inversion.cu)
target_link_libraries(batched_matrix_inversion PRIVATE naga)

add_executable(radial_point_interpolation examples/radial_point_interpolation.cu)
target_link_libraries(radial_point_interpolation PRIVATE naga)

add_executable(radial_point_interpolation_3d examples/radial_point_interpolation_3d.cu)
target_link_libraries(radial_point_interpolation_3d PRIVATE naga)

add_executable(nonlocal_divergence examples/nonlocal_divergence.cu)
target_link_libraries(nonlocal_divergence PRIVATE naga)