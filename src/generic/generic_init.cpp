#include <generic_init.h>

// Helper function to detect and parse normal distribution patterns
bool parseNormalDistribution(const std::string& initialization, float& mean, float& std_dev) {
  std::vector<std::string> patterns = {"normal_float", "norm_float", "norm_dist"};
  
  for (const auto& pattern : patterns) {
    if (initialization.find(pattern) == 0) {
      mean = 0.0f; 
      std_dev = 1.0f;  // defaults
      
      if (initialization.length() > pattern.length() && initialization[pattern.length()] == '_') {
        // Parse additional parameters like "norm_float_5_2"
        std::string params = initialization.substr(pattern.length() + 1);
        std::istringstream ss(params);
        std::string mean_str, std_str;
        
        if (std::getline(ss, mean_str, '_') && std::getline(ss, std_str)) {
          try {
            mean = std::stof(mean_str);
            std_dev = std::stof(std_str);
          } catch (const std::exception&) {
            // If parsing fails, use defaults
            mean = 0.0f;
            std_dev = 1.0f;
          }
        } else if (!mean_str.empty()) {
          // Only mean provided, use default std
          try {
            mean = std::stof(mean_str);
          } catch (const std::exception&) {
            mean = 0.0f;
          }
        }
      }
      return true;
    }
  }
  return false;
}