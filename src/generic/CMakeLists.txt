
add_library(mblasgeneric STATIC
    genericGemm.cpp
)

set_target_properties(mblasgeneric PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mblasgeneric PRIVATE
    .
    ../third_party
    ..
)
