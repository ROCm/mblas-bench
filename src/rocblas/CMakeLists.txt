cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_CXX_STANDARD 17)
project(mblas_bench LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "-fPIC -fgpu-rdc -w -D__HIP_PLATFORM_AMD__")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
# Search for rocm in common locations
list(PREPEND CMAKE_PREFIX_PATH /opt/rocm)
set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
# Find hip packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(Threads REQUIRED) 


add_library(mblasrocm SHARED
    rocblasGemm.cpp
    rocblasGemmFactory.cpp
    hipblasLtGemm.cpp
    hipblasLtGemmFactory.cpp
    rocblasConvert.hip
    rocblasCreateAllocate.cpp
    rocblasDtypeUtils.cpp
    rocblasError.cpp
    mblasHipDataType.cpp
    mblasHipComputeType.cpp
    mblasHipOperation.cpp
    mblasRocDataType.cpp
    mblasRocComputeType.cpp
    mblasRocOperation.cpp
)
target_compile_options(mblasrocm PRIVATE -Wno-unused-result)

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
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
    mblasgeneric
)