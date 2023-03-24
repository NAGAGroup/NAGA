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
target_link_libraries(naga INTERFACE scalix)

# naga examples
add_executable(sample sample.cu)
target_link_libraries(sample naga)