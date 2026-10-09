#include <generic_init.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

template <typename Keep>
constexpr size_t count_codes(int bits, Keep keep) {
  size_t n = 0;
  for (int c = 0; c < (1 << bits); ++c)
    if (keep(static_cast<uint8_t>(c))) ++n;
  return n;
}

template <size_t N, typename Keep>
constexpr std::array<uint8_t, N> make_codes(int bits, Keep keep) {
  std::array<uint8_t, N> codes{};
  size_t n = 0;
  for (int c = 0; c < (1 << bits); ++c)
    if (keep(static_cast<uint8_t>(c))) codes[n++] = static_cast<uint8_t>(c);
  return codes;
}

// OCP E4M3: 0x7F and 0xFF are NaN.
constexpr auto e4m3_keep = [](uint8_t c) { return (c & 0x7F) != 0x7F; };
// OCP E5M2: exponent all ones (0x7C-0x7F, 0xFC-0xFF) is Inf or NaN.
constexpr auto e5m2_keep = [](uint8_t c) { return (c & 0x7C) != 0x7C; };
constexpr auto keep_all  = [](uint8_t) { return true; };

constexpr auto e4m3_codes = make_codes<count_codes(8, e4m3_keep)>(8, e4m3_keep);
constexpr auto e5m2_codes = make_codes<count_codes(8, e5m2_keep)>(8, e5m2_keep);
constexpr auto fp6_codes  = make_codes<count_codes(6, keep_all)>(6, keep_all);
constexpr auto fp4_codes  = make_codes<count_codes(4, keep_all)>(4, keep_all);

static_assert(e4m3_codes.size() == 254 && e4m3_codes[126] == 0x7E && e4m3_codes[127] == 0x80);
static_assert(e5m2_codes.size() == 248 && e5m2_codes[123] == 0x7B && e5m2_codes[124] == 0x80);
static_assert(fp6_codes.size() == 64 && fp4_codes.size() == 16);

struct code_table {
  const uint8_t *codes;
  int count;
};

code_table native_code_table(const mblas_data_type &type) {
  switch (static_cast<mblas_data_type_enum>(type)) {
    case MBLAS_R_8F_E4M3:
      return {e4m3_codes.data(), static_cast<int>(e4m3_codes.size())};
    case MBLAS_R_8F_E5M2:
      return {e5m2_codes.data(), static_cast<int>(e5m2_codes.size())};
    case MBLAS_R_6F_E2M3:
    case MBLAS_R_6F_E3M2:
      return {fp6_codes.data(), static_cast<int>(fp6_codes.size())};
    case MBLAS_R_4F_E2M1:
      return {fp4_codes.data(), static_cast<int>(fp4_codes.size())};
    default:
      throw std::invalid_argument("Error: uniform_native not supported for " + type.to_string() +
                                  "; supported types are OCP E4M3, E5M2, E3M2, E2M3, and E2M1");
  }
}

}  // namespace

void fill_host_uniform_native(const mblas_data_type &type, void **ptr_array, long x, long y, int batch,
                              long long stride, int flush_batch_count) {
  const long long n_elements = stride * (batch - 1) + static_cast<long long>(x) * y;
  if (batch * x * y == 0) {
    // Matrix not used, nothing is copied to the device
    return;
  }
  const code_table table = native_code_table(type);
  const int packing = type.get_packing_count();
  const long long n_bytes = (n_elements + packing - 1) / packing;

  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, table.count - 1);
    #pragma omp for collapse(2)
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (long long i = 0; i < n_bytes; i++) {
        uint8_t *A = static_cast<uint8_t *>(ptr_array[flush_idx]);
        if (packing == 2) {
          uint8_t lo = table.codes[dist(gen)];
          uint8_t hi = table.codes[dist(gen)];
          A[i] = static_cast<uint8_t>(lo | (hi << 4));
        } else {
          A[i] = table.codes[dist(gen)];
        }
      }
    }
  }
}
