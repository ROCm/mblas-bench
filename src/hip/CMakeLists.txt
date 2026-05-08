cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_CXX_STANDARD 17)
project(mblaship LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "-fPIC -fgpu-rdc -w -D__HIP_PLATFORM_AMD__")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
set(GPU_TARGETS "gfx90a,gfx942,gfx950" CACHE STRING "GPU targets to compile for")
# Search for rocm in common locations
list(PREPEND CMAKE_PREFIX_PATH /opt/rocm)
set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")

# Find HIP/ROCm packages (conditionally based on enabled backends)
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

if(WITH_ROCBLAS)
    find_package(rocblas REQUIRED)
endif()

if(WITH_HIPBLASLT)
    find_package(hipblaslt REQUIRED)
endif()

# Base hip sources (always included)
set(ROCM_SOURCES
    hip_convert.hip
    hip_create_allocate.cpp
    hip_datatype_utils.cpp
    hip_error.cpp
    mblas_hip_data_type.cpp
)

# Conditionally add hipblaslt backend sources
if(WITH_HIPBLASLT)
    list(APPEND ROCM_SOURCES
        hipblaslt/hipblaslt_gemm.cpp
        hipblaslt/hipblaslt_gemm_factory.cpp
        hipblaslt/mblas_hipblas_compute_type.cpp
        hipblaslt/mblas_hipblas_operation.cpp
    )
endif()

# Conditionally add rocblas backend sources
if(WITH_ROCBLAS)
    list(APPEND ROCM_SOURCES
        rocblas/rocblas_gemm.cpp
        rocblas/rocblas_gemm_factory.cpp
        rocblas/mblas_rocblas_data_type.cpp
        rocblas/mblas_rocblas_compute_type.cpp
        rocblas/mblas_rocblas_operation.cpp
    )
endif()

add_library(mblaship SHARED ${ROCM_SOURCES})
target_compile_options(mblaship PRIVATE -Wno-unused-result)

set_target_properties(mblaship PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mblaship PUBLIC
    .
    hipblaslt
    rocblas
    ../third_party
    ../generic
)

set(ROCM_LINK_LIBS
    hip::device
    hip::host
    roc::hipblas
    Threads::Threads
    OpenMP::OpenMP_CXX
    -lpthread
    -fopenmp
    mblasgeneric
)

target_compile_definitions(mblaship PRIVATE
    MBLAS_WITH_ROCBLAS=$<BOOL:${WITH_ROCBLAS}>
    MBLAS_WITH_HIPBLASLT=$<BOOL:${WITH_HIPBLASLT}>
)

if(WITH_ROCBLAS)
    list(APPEND ROCM_LINK_LIBS roc::rocblas)
endif()

if(WITH_HIPBLASLT)
    list(APPEND ROCM_LINK_LIBS roc::hipblaslt)
endif()

target_link_libraries(mblaship PRIVATE ${ROCM_LINK_LIBS})
