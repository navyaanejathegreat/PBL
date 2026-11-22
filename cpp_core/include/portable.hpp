// std::stoi / std::stod are missing in old MinGW.org g++ (6.x); these work everywhere.
#pragma once
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nn {
inline int to_int(const std::string& s) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str()) throw std::invalid_argument("not an integer: " + s);
    return static_cast<int>(v);
}
inline double to_double(const std::string& s) {
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) throw std::invalid_argument("not a number: " + s);
    return v;
}
inline std::string to_str(long v) {   // std::to_string is missing there too
    std::ostringstream o;
    o << v;
    return o.str();
}
}  // namespace nn
