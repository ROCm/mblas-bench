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