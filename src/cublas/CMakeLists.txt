cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

#set(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDAHOSTCXX g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#unset(CMAKE_CXX_COMPILER)
#set(CMAKE_CXX_STANDARD 17)
unset(CMAKE_CXX_STANDARD)
project(mblas_bench LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
#set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Check for CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT enum in CUDA headers
# This is more reliable than parsing version numbers since patch levels can vary
if(CUDAToolkit_INCLUDE_DIRS)
    set(CUBLAS_HEADER "${CUDAToolkit_INCLUDE_DIRS}/cublas_v2.h")
    if(EXISTS "${CUBLAS_HEADER}")
        file(READ "${CUBLAS_HEADER}" CUBLAS_HEADER_CONTENT)
        string(FIND "${CUBLAS_HEADER_CONTENT}" "CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT" ENUM_FOUND)
        if(NOT ENUM_FOUND EQUAL -1)
            message(STATUS "Found CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT in ${CUBLAS_HEADER}")
            add_definitions(-DHAS_CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)
        else()
            message(STATUS "CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT not found in ${CUBLAS_HEADER}")
        endif()
    else()
        message(WARNING "Could not find cublas_v2.h at ${CUBLAS_HEADER}")
    endif()
else()
    message(WARNING "CUDAToolkit_INCLUDE_DIRS not set, cannot check for CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT")
endif() 

add_library(mblascuda SHARED
    cublas_gemm.cpp
    cublas_gemm_factory.cpp
    cublaslt_gemm.cpp
    cublaslt_gemm_factory.cpp
    cublas_convert.cu
    cublas_create_allocate.cpp
    cublas_datatype_utils.cpp
    cuda_error.cpp
    mblas_cuda_data_type.cpp
    mblas_cuda_compute_type.cpp
    mblas_cuda_operation.cpp
)

set_target_properties(mblascuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(mblascuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mblascuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set_target_properties(mblascuda PROPERTIES CUDA_ARCHITECTURES 80 86 89 90)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=compute_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -g")

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8.0)
    add_definitions(-DENABLE_CUDA_FP4)
endif()

target_include_directories(mblascuda PRIVATE
    .
    ../third_party
    ../generic
)

target_link_libraries(mblascuda PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::cublas 
    CUDA::cublasLt
    CUDA::nvml
    mblasgeneric
)
