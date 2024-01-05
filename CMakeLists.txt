cmake_minimum_required(VERSION 3.20)
project(
  NAGA
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# if NAGA_EXAMPLES_RESULTS_DIR is passed as a cmake variable add it as a
# preprocessor definition
if(NAGA_EXAMPLES_RESULTS_DIR)
  add_compile_definitions(
    NAGA_EXAMPLES_RESULTS_DIR=${NAGA_EXAMPLES_RESULTS_DIR})
endif()
add_compile_definitions(NAGA_RESOURCES_DIR="${PROJECT_SOURCE_DIR}/resources")

set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/eigen)
set(NANOFLANN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/nanoflann/include)
set(AUDIOFILE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/AudioFile)

add_subdirectory(extern/Scalix)

option(SDF_USE_SYSTEM_EIGEN "" OFF)
add_subdirectory(extern/sdf)

find_package(OpenMP REQUIRED)

find_package(VTK REQUIRED COMPONENTS IOXML)

add_library(
  naga-lib STATIC
        source/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/detail/conforming_point_cloud_provider.cu
)
target_include_directories(
  naga-lib PRIVATE ${PROJECT_SOURCE_DIR}/. ${EIGEN_INCLUDE_DIR}
                   ${NANOFLANN_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/extern/tinyobjloader)
target_link_libraries(naga-lib PRIVATE scalix sdf)

# naga is a mostly header only library
add_library(naga INTERFACE)
target_include_directories(
  naga INTERFACE ${PROJECT_SOURCE_DIR}/. ${EIGEN_INCLUDE_DIR}
                 ${NANOFLANN_INCLUDE_DIR} ${AUDIOFILE_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/extern/tinyobjloader)
target_link_libraries(naga INTERFACE naga-lib scalix cublas cusparse sdf
                                     OpenMP::OpenMP_CXX ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS naga MODULES ${VTK_LIBRARIES})
set_target_properties(naga PROPERTIES CMAKE_CUDA_FLAGS
                                      "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

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

add_executable(nonlocal_lattice_boltzmann
               examples/nonlocal_lattice_boltzmann.cu)
target_link_libraries(nonlocal_lattice_boltzmann PRIVATE naga)

add_executable(nonlocal_lattice_boltzmann_3d
               examples/nonlocal_lattice_boltzmann_3d.cu)
target_link_libraries(nonlocal_lattice_boltzmann_3d PRIVATE naga)

add_executable(lbm_streaming_tests_3d examples/lbm_streaming_tests_3d.cu)
target_link_libraries(lbm_streaming_tests_3d PRIVATE naga)

add_executable(knn_validation_3d examples/knn_validation_3d.cu)
target_link_libraries(knn_validation_3d PRIVATE naga)

add_executable(ranges_example examples/ranges_example.cu)
target_link_libraries(ranges_example PRIVATE naga)

add_executable(cgal_2d_mesh examples/cgal_2d_mesh.cu)
target_link_libraries(cgal_2d_mesh PRIVATE naga)

add_executable(cgal_3d_mesh examples/cgal_3d_mesh.cu)
target_link_libraries(cgal_3d_mesh PRIVATE naga)

add_executable(nonlocal_lbm_pulse_3d examples/nonlocal_lbm_pulse_3d.cu)
target_link_libraries(nonlocal_lbm_pulse_3d PRIVATE naga)

add_executable(sparse_matrix_vector_mult examples/sparse_matrix_vector_mult.cu)
target_link_libraries(sparse_matrix_vector_mult PRIVATE naga)

add_executable(sparse_gemm examples/sparse_gemm.cu)
target_link_libraries(sparse_gemm PRIVATE naga)
