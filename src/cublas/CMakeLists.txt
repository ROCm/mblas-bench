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

# Extract exact CUDA version including patch number from CMake's detected version
# This comes from the "The CUDA compiler identification is NVIDIA X.Y.Z" message
if(CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH "${CMAKE_CUDA_COMPILER_VERSION}")
    if(CUDA_VERSION_MATCH)
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
        set(CUDA_VERSION_PATCH ${CMAKE_MATCH_3})
        
        # Create a numeric version like 130088 for 13.0.88
        math(EXPR CUDA_VERSION_FULL "${CUDA_VERSION_MAJOR} * 10000 + ${CUDA_VERSION_MINOR} * 100 + ${CUDA_VERSION_PATCH}")
        
        message(STATUS "Using CUDA version: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}.${CUDA_VERSION_PATCH} (MBLAS_CUDA_VERSION_FULL=${CUDA_VERSION_FULL})")
        
        # Pass as preprocessor definitions
        add_definitions(-DMBLAS_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
        add_definitions(-DMBLAS_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})
        add_definitions(-DMBLAS_CUDA_VERSION_PATCH=${CUDA_VERSION_PATCH})
        add_definitions(-DMBLAS_CUDA_VERSION_FULL=${CUDA_VERSION_FULL})
    else()
        message(WARNING "Could not parse CUDA version from CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
else()
    message(WARNING "CMAKE_CUDA_COMPILER_VERSION not set")
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
