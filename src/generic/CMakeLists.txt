
add_library(mblasgeneric STATIC
    genericGemm.cpp
    genericSetup.cpp
    mblasDataType.cpp
    mblasComputeType.cpp
    mblasOperation.cpp
)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -g")

set_target_properties(mblasgeneric PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mblasgeneric PRIVATE
    .
    ../third_party
    ..
)

target_link_libraries(mblasgeneric PUBLIC OpenMP::OpenMP_CXX)