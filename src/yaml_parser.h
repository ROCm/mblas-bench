#pragma once
#include "third_party/cxxopts.hpp"
#include <string>
#include <vector>
#include <optional>
#include <regex>

std::optional<cxxopts::ParseResult> parse_line(const std::string& line, const cxxopts::ParseResult& default_opts)
{
    // Expected formats:
    // - {key1: value1, key2: value2, ...}
    // {key1: value1, key2: value2, ...}
    
    // Check if the line is commented out or empty
    std::string trimmed_line = std::regex_replace(line, std::regex(R"(^\s+|\s+$)"), "");
    if (trimmed_line.empty() || trimmed_line[0] == '#') {
        return std::nullopt; // Ignore comments and empty lines
    }

    // Extract the content within braces
    std::regex yaml_regex(R"(\s*\{([^}]*)\}\s*)");
    std::smatch match;
    if (!std::regex_match(line, match, yaml_regex)) {
        return std::nullopt; // Not a valid YAML line
    }

    // Parse key-value pairs
    std::string content = match[1].str();
    std::regex pair_regex(R"(\s*([^:]+)\s*:\s*([^,]+)\s*,?)");
    auto pairs_begin = std::sregex_iterator(content.begin(), content.end(), pair_regex);
    auto pairs_end = std::sregex_iterator();
    cxxopts::ParseResult result = default_opts;
    for (std::sregex_iterator i = pairs_begin; i != pairs_end; ++i) {
        std::smatch pair_match = *i;
        std::string key = std::regex_replace(pair_match[1].str(), std::regex(R"(^\s+|\s+$)"), "");
        std::string value = std::regex_replace(pair_match[2].str(), std::regex(R"(^\s+|\s+$)"), "");
        result[key] = value;
    }
    return result;
}


std::vector<cxxopts::ParseResult> parse_yaml_file(const std::string& filename, const cxxopts::ParseResult& default_opts)
{
    std::vector<cxxopts::ParseResult> results;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open YAML file: " << filename << std::endl;
        return results;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto parsed = parse_line(line, default_opts);
        if (parsed) {
            results.push_back(*parsed);
        }
    }
    file.close();
    return results;
}
