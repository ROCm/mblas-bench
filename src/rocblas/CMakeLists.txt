cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_CXX_STANDARD 17)
project(mblas_bench LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "-fPIC -fgpu-rdc -w -D__HIP_PLATFORM_AMD__ --offload-arch=gfx950")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
set(GPU_TARGETS "gfx90a,gfx942,gfx950" CACHE STRING "GPU targets to compile for")
# Search for rocm in common locations
list(PREPEND CMAKE_PREFIX_PATH /opt/rocm)
set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
# Find hip packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(Threads REQUIRED) 
find_package(OpenMP REQUIRED)


add_library(mblasrocm SHARED
	#    rocblas_gemm.cpp
	#rocblas_gemm_factory.cpp
    hipblaslt_gemm.cpp
    hipblaslt_gemm_factory.cpp
    hip_convert.hip
    hip_create_allocate.cpp
    hip_datatype_utils.cpp
    hip_error.cpp
    mblas_hip_data_type.cpp
    mblas_hipblas_compute_type.cpp
    mblas_hipblas_operation.cpp
    #mblas_rocblas_data_type.cpp
    #mblas_rocblas_compute_type.cpp
    #mblas_rocblas_operation.cpp
)
target_compile_options(mblasrocm PRIVATE -Wno-unused-result)

set_target_properties(mblasrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mblasrocm PRIVATE
    .
    ../third_party
    ../generic
)

target_link_libraries(mblasrocm PRIVATE
	hip::device
    hip::host
    #roc::rocblas
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
    OpenMP::OpenMP_CXX
    -lpthread
    -fopenmp
    mblasgeneric
)
