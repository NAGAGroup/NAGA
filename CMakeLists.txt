cmake_minimum_required(VERSION 3.20)
project(
  NAGA
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/eigen)
set(NANOFLANN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/nanoflann/include)

add_subdirectory(extern/Scalix)

set(SDF_USE_SYSTEM_EIGEN OFF)
add_subdirectory(extern/sdf)

# naga is a header only library
add_library(naga INTERFACE)
target_include_directories(naga INTERFACE ${PROJECT_SOURCE_DIR}/.
                                           ${EIGEN_INCLUDE_DIR} ${NANOFLANN_INCLUDE_DIR})
target_link_libraries(naga INTERFACE scalix cublas sdf)

# naga examples
add_executable(nd_cubic_segmentation examples/nd_cubic_segmentation.cu)
target_link_libraries(nd_cubic_segmentation PRIVATE naga)

add_executable(batched_knn_search examples/batched_knn_search.cu)
target_link_libraries(batched_knn_search PRIVATE naga)

add_executable(batched_matrix_inversion examples/batched_matrix_inversion.cu)
target_link_libraries(batched_matrix_inversion PRIVATE naga)

add_executable(radial_point_interpolation
               examples/radial_point_interpolation.cu)
target_link_libraries(radial_point_interpolation PRIVATE naga)

add_executable(radial_point_interpolation_3d
               examples/radial_point_interpolation_3d.cu)
target_link_libraries(radial_point_interpolation_3d PRIVATE naga)

add_executable(nonlocal_divergence examples/nonlocal_divergence.cu)
target_link_libraries(nonlocal_divergence PRIVATE naga)

add_executable(nonlocal_divergence_3d examples/nonlocal_divergence_3d.cu)
target_link_libraries(nonlocal_divergence_3d PRIVATE naga)

add_executable(nonlocal_advection examples/nonlocal_advection.cu)
target_link_libraries(nonlocal_advection PRIVATE naga)

add_executable(nonlocal_advection_3d examples/nonlocal_advection_3d.cu)
target_link_libraries(nonlocal_advection_3d PRIVATE naga)

add_executable(
  nonlocal_lattice_boltzmann examples/nonlocal_lattice_boltzmann.cu
                             examples/nonlocal_lattice_boltzmann.cu)
target_link_libraries(nonlocal_lattice_boltzmann PRIVATE naga)
