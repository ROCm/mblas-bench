cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

cmake_policy(SET CMP0028 NEW)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
project(mblas_bench LANGUAGES CXX)
#set(CMAKE_CXX_COMPILER_ID "Clang")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(WITH_ROCM "Build with ROCm support" OFF)
option(WITH_HIPBLASLT "Build with hipblaslt support" OFF)
option(WITH_ROCBLAS "Build with rocblas support" OFF)
option(WITH_CUDA "Build with CUDA support" OFF)

if(WITH_ROCM)
    set(WITH_HIPBLASLT ON)
    set(WITH_ROCBLAS ON)
endif()

set(WITH_HIP OFF)
if(WITH_HIPBLASLT OR WITH_ROCBLAS)
    set(WITH_HIP ON)
endif()

if(NOT WITH_HIP AND NOT WITH_CUDA)
    message(FATAL_ERROR "No backends enabled. Please enable at least one backend (WITH_ROCM, WITH_HIPBLASLT, WITH_ROCBLAS, WITH_CUDA).")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_subdirectory(generic)

#Compile shared libs for GPU backends
if (WITH_HIP)
    add_subdirectory(hip)
endif()

if (WITH_CUDA)
    add_subdirectory(cublas)
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
if (WITH_HIPBLASLT)
    target_include_directories(mblas-bench PRIVATE
        "hip/hipblaslt"
    )
else ()
    target_include_directories(mblas-bench PRIVATE
        "dummy/hipblaslt"
    )
endif()

if (WITH_ROCBLAS)
    target_include_directories(mblas-bench PRIVATE
        "hip/rocblas"
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

if (WITH_HIP)
    target_link_libraries(mblas-bench PRIVATE
    mblaship
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
