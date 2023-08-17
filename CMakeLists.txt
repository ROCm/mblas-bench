cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER "/opt/rocm/hip/bin/hipcc")
set(CMAKE_CXX_STANDARD 17)
project(matmul_bench LANGUAGES CXX HIP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(Threads REQUIRED) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(
    mblas_bench
    src/mblasBench.cpp
    src/genericGemm.cpp
    src/rocblas/rocblasGemm.cpp
    src/rocblas/rocblasConvert.hip
    src/rocblas/rocblasCreateAllocate.cpp
    src/rocblas/rocblasDtypeUtils.cpp
    src/rocblas/rocblasError.cpp
    src/third_party/barrier.h
    src/third_party/cxxopts.hpp
)
target_include_directories(mblas_bench PRIVATE
    src/
    src/third_party
)

target_link_libraries(mblas_bench PRIVATE
	hip::device
    hip::host
    roc::rocblas
    Threads::Threads
)
