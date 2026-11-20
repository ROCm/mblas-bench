
add_library(mblasgeneric STATIC
    generic_gemm.cpp
    generic_setup.cpp
    generic_init.cpp
    generic_dump.cpp
    mblas_data_type.cpp
    mblas_compute_type.cpp
    mblas_operation.cpp
)
find_package(OpenMP REQUIRED)

set_target_properties(mblasgeneric PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Not CMAKE_BUILD_TYPE=Release: its -DNDEBUG would disable the assert() calls
# that the error-check helpers depend on.
target_compile_options(mblasgeneric PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3> -g)

target_include_directories(mblasgeneric PRIVATE
    .
    ../third_party
    ..
)

# Compile flag only. Do not link OpenMP::OpenMP_CXX here: this static library is linked into
# every backend, and FindOpenMP (run under g++) would force libgomp onto the clang-built HIP
# backend. Each backend links the OpenMP runtime of its own compiler instead.
target_compile_options(mblasgeneric PRIVATE ${OpenMP_CXX_FLAGS})
