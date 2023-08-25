cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER "/opt/rocm/hip/bin/hipcc")
#set(CMAKE_HIP_COMPILER "/opt/rocm/hip/bin/hipcc")
set(CMAKE_CXX_STANDARD 17)
project(mblas_bench LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "-fPIC -fgpu-rdc -w -D__HIP_PLATFORM_AMD__")
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


add_library(mblasrocm SHARED
    rocblasGemm.cpp
    rocblasConvert.hip
    rocblasCreateAllocate.cpp
    rocblasDtypeUtils.cpp
    rocblasError.cpp
)

set_target_properties(mblasrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(mblasrocm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(mblasrocm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(mblasrocm PRIVATE
    .
    ../third_party
    ../generic
)

target_link_libraries(mblasrocm PRIVATE
	hip::device
    hip::host
    roc::rocblas
    Threads::Threads
    mblasgeneric
)