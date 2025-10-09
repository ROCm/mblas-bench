cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

cmake_policy(SET CMP0028 NEW)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
project(mblas_bench LANGUAGES CXX)
#set(CMAKE_CXX_COMPILER_ID "Clang")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT DEFINED WITH_ROCM)
    set(WITH_ROCM "true")
endif()
if (NOT DEFINED WITH_CUDA)
    set(WITH_CUDA "true")
endif()
#set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake":"/opt/rocm/lib/cmake/":${CMAKE_MODULE_PATH})
#message("CMAKE_MODULE_PATH"=${CMAKE_MODULE_PATH})

# Set the GPU to compile for
#set(GPU_TARGETS "gfx908:sramecc+:xnack-" CACHE STRING "GPU targets to compile for")
# Search for rocm in common locations
#list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip packages
#find_package(hip REQUIRED)
#find_package(rocblas REQUIRED)
#find_package(Threads REQUIRED) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_subdirectory(generic)

# Compile shared libs for GPU backends
if (WITH_ROCM)
    add_subdirectory(rocblas)
    add_compile_definitions(WITH_ROCM)
endif()

if (WITH_CUDA)
    add_subdirectory(cublas)
    add_compile_definitions(WITH_CUDA)
endif()

add_executable(
    mblas-bench
    mblas_bench.cpp
)

target_include_directories(mblas-bench PUBLIC
    .
    third_party
    generic
)

include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# Add each backend to the include path
# If backend is disabled, use the dummy implementation
if (WITH_ROCM)
    target_include_directories(mblas-bench PRIVATE
        "rocblas"
    )
else ()
    target_include_directories(mblas-bench PRIVATE
        "dummy/rocblas"
    )
endif()

if (WITH_CUDA)
    target_include_directories(mblas-bench PRIVATE
        "cublas"
    )
else ()
    target_include_directories(mblas-bench PRIVATE
        "dummy/cublas"
    )
endif()

target_link_libraries(mblas-bench PUBLIC
    -lstdc++
    -lm
    -lpthread
    -fopenmp
    yaml-cpp::yaml-cpp
)

if (WITH_ROCM)
    target_link_libraries(mblas-bench PRIVATE
    mblasrocm
        -lstdc++
        -lm
        -lpthread
        -fopenmp
    )
endif()


if (WITH_CUDA)
    #link_directories(/usr/local/cuda/lib64/)
    target_link_libraries(mblas-bench PRIVATE
        mblascuda
        -lstdc++
        -lm
        -lpthread
    )
endif()
