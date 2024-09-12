cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

#set(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDAHOSTCXX g++)
set(CMAKE_CUDA_COMPILER nvcc)
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
    cublasGemm.cpp
    cublasGemmFactory.cpp
    cublasLtGemm.cpp
    cublasLtGemmFactory.cpp
    cublasConvert.cu
    cublasCreateAllocate.cpp
    cublasDtypeUtils.cpp
    cudaError.cpp
    mblasCuDataType.cpp
    mblasCuComputeType.cpp
    mblasCuOperation.cpp
)

set_target_properties(mblascuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(mblascuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mblascuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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
    mblasgeneric
)