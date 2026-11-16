#include "generic_dump.h"

#include <fstream>
#include <stdexcept>

namespace {
std::string json_escape(const std::string &s) {
  std::string out;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    out += ch;
  }
  return out;
}
}  // namespace

std::string write_matrix_dump(const std::string &prefix,
                              const matrix_dump_info &info,
                              const std::vector<uint8_t> &bytes) {
  const std::string bin_path = prefix + "_" + info.matrix + ".bin";
  const std::string json_path = bin_path + ".json";

  std::ofstream bin(bin_path, std::ios::binary);
  if (!bin) throw std::runtime_error("Cannot open " + bin_path + " for writing");
  bin.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  bin.close();
  if (!bin) throw std::runtime_error("Failed writing " + bin_path);

  const long long num_elements =
      info.stride * (info.batch - 1) + static_cast<long long>(info.rows_mem) * info.cols_mem;
  std::ofstream js(json_path);
  if (!js) throw std::runtime_error("Cannot open " + json_path + " for writing");
  js << "{\n"
     << "  \"matrix\": \"" << info.matrix << "\",\n"
     << "  \"file\": \"" << json_escape(bin_path) << "\",\n"
     << "  \"type\": \"" << json_escape(info.type_name) << "\",\n"
     << "  \"packing\": " << info.packing << ",\n"
     << "  \"bytes\": " << bytes.size() << ",\n"
     << "  \"num_elements\": " << num_elements << ",\n"
     << "  \"rows\": " << info.rows << ",\n"
     << "  \"cols\": " << info.cols << ",\n"
     << "  \"rows_mem\": " << info.rows_mem << ",\n"
     << "  \"cols_mem\": " << info.cols_mem << ",\n"
     << "  \"ld\": " << info.ld << ",\n"
     << "  \"batch\": " << info.batch << ",\n"
     << "  \"stride\": " << info.stride << ",\n"
     << "  \"layout\": \"column-major\",\n"
     << "  \"init\": \"" << json_escape(info.init) << "\",\n"
     << "  \"device\": " << info.device << ",\n"
     << "  \"stage\": \"" << info.stage << "\"\n"
     << "}\n";
  js.close();
  if (!js) throw std::runtime_error("Failed writing " + json_path);
  return bin_path;
}
