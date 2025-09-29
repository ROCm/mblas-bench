#pragma once

#include "third_party/cxxopts.hpp"
#include <string>
#include <vector>
#include <optional>
#include <regex>
#include <fstream>

std::optional<std::unordered_map<std::string, std::string>> parse_line(const std::string& line)
{
    // Expected formats:
    // - {key1: value1, key2: value2, ...}
    // {key1: value1, key2: value2, ...}
    
    // Check if the line is commented out or empty
    std::string trimmed_line = line;
    trimmed_line.erase(0, trimmed_line.find_first_not_of(" \t\n\r"));
    if (trimmed_line.empty() || trimmed_line[0] == '#') {
        return std::nullopt; // Ignore comments and empty lines
    }

    // Extract the content within braces
    size_t start = trimmed_line.find('{');
    size_t end = trimmed_line.find('}');
    if (start == std::string::npos || end == std::string::npos || start >= end) {
        return std::nullopt; // Not a valid YAML line
    }
    trimmed_line = trimmed_line.substr(start + 1, end - start - 1);

    // Parse key-value pairs
    std::string content = trimmed_line;
    std::regex pair_regex(R"(\s*([^:]+)\s*:\s*([^,]+)\s*,?)");
    auto pairs_begin = std::sregex_iterator(content.begin(), content.end(), pair_regex);
    auto pairs_end = std::sregex_iterator();

    std::unordered_map<std::string, std::string> result;
    for (std::sregex_iterator i = pairs_begin; i != pairs_end; ++i) {
        std::smatch pair_match = *i;
        std::string key = std::regex_replace(pair_match[1].str(), std::regex(R"(^\s+|\s+$)"), "");
        std::string value = std::regex_replace(pair_match[2].str(), std::regex(R"(^\s+|\s+$)"), "");
        result[key] = value;
    }
    return result;
}


std::vector<cxxopts::ParseResult> parse_yaml_file(const std::string& filename, cxxopts::Options& opts, int argc, char** argv)
{
    // Read the YAML file and parse each line
    // For each parsed line, convert it to command-line arguments and parse with cxxopts so that the original code can remain unchanged
    std::vector<cxxopts::ParseResult> results;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open YAML file: " << filename << std::endl;
        return results;
    }

    std::string line;
    while (std::getline(file, line)) {
        const auto& parsed = parse_line(line);
        if (parsed) {
            std::vector<std::string> args;
            for (int i = 0; i < argc; ++i) {
                args.push_back(argv[i]);
            }
            for (const auto& [key, value]: *parsed)
            {
                const std::string arg_key = key.size() == 1 ? "-" + key : "--" + key;
                args.push_back(arg_key);
                args.push_back(value);
            }
            std::vector<const char*> cstr_args;
            for (const auto& arg: args) {
                cstr_args.push_back(arg.c_str());
            }

            auto result = opts.parse(static_cast<int>(cstr_args.size()), cstr_args.data());
            results.push_back(result);
            std::cout << result.arguments_string() << std::endl; // For debugging for now
        }
    }
    file.close();
    return results;
}
