#include "generic_setup.h"

#include <cassert>
#include <iostream>
#include <cstdint>

/*
 * Round's a variable up to the nearest number divisible by 16
 * Used for ensuring that all our pointers fall on a 16 byte boundary
*/
uint64_t round_up(uint64_t numToRound, uint64_t multiple)
{
    //assert(multiple && ((multiple & (multiple - 1)) == 0));
    //return (numToRound + multiple - 1) & -multiple;
  uint64_t remainder = numToRound % multiple;
  uint64_t new_num = numToRound + (multiple - remainder);
  assert(new_num % multiple == 0);
  return new_num;
}

/* 
 * Calculates the offset between blocks in the rotating tensor area 
 * Returns: total_block_size 
 * Also updated: <a-d>_offset
 */
uint64_t calculate_offsets( uint64_t rows_mem_a, uint64_t cols_mem_a, uint64_t rows_mem_b, uint64_t cols_mem_b,
                        uint64_t rows_mem_c, uint64_t cols_mem_c, uint64_t rows_mem_d, uint64_t cols_mem_d, 
                        int a_type_size,  int b_type_size, int c_type_size, int d_type_size,
                        int a_type_pack,  int b_type_pack, int c_type_pack, int d_type_pack,
                        int batch_count, bool inplace
) {
  uint64_t a_size, b_size, c_size, d_size;
  uint64_t a_size_round, b_size_round, c_size_round, d_size_round;
  uint64_t total_block_size;
  a_size = ceil_division(rows_mem_a * cols_mem_a * batch_count * a_type_size, uint64_t(a_type_pack)); 
  a_size_round = round_up(a_size, 16);
  b_size = ceil_division(rows_mem_b * cols_mem_b * batch_count * b_type_size, uint64_t(b_type_pack)); 
  b_size_round = round_up(b_size, 16);
  c_size = ceil_division(rows_mem_c * cols_mem_c * batch_count * c_type_size, uint64_t(c_type_pack)); 
  c_size_round = round_up(c_size, 16);
  if (!inplace) {
    d_size = ceil_division(rows_mem_d * cols_mem_d * batch_count * d_type_size, uint64_t(d_type_pack)); 
    d_size_round = round_up(d_size, 16);
  } else {
    d_size = c_size;
    d_size_round = 0;
  }

  total_block_size = a_size_round + b_size_round + c_size_round + d_size_round;
  return total_block_size;
}