
add_library(mblasgeneric STATIC
    generic_gemm.cpp
    generic_setup.cpp
    generic_init.cpp
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

target_link_libraries(mblasgeneric PUBLIC OpenMP::OpenMP_CXX)