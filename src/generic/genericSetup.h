#pragma once

#include <cassert>
#include <iostream>

uint64_t roundUp(uint64_t numToRound, uint64_t multiple);

uint64_t calculate_offsets( uint64_t rows_mem_a, uint64_t cols_mem_a, uint64_t rows_mem_b, uint64_t cols_mem_b,
                        uint64_t rows_mem_c, uint64_t cols_mem_c, uint64_t rows_mem_d, uint64_t cols_mem_d, 
                        uint64_t & a_offset, uint64_t & b_offset, uint64_t & c_offset, uint64_t & d_offset,
                        int a_type_size,  int b_type_size, int c_type_size, int d_type_size,
                        int batch_count, bool inplace
);