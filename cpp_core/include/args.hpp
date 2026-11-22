// Tiny "--key value" argument parser shared by the command-line tools.
#pragma once
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "portable.hpp"

namespace nn {

struct Args {
    std::string command;
    std::map<std::string, std::string> kv;

    Args(int argc, char** argv) {
        int i = 1;
        if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0) command = argv[i++];
        for (; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument " + k);
            k = k.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv[k] = argv[++i];
            else kv[k] = "1";   // flag
        }
    }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string str(const std::string& k, const std::string& d) const { return has(k) ? kv.at(k) : d; }
    double num(const std::string& k, double d) const { return has(k) ? to_double(kv.at(k)) : d; }
    int integer(const std::string& k, int d) const { return has(k) ? to_int(kv.at(k)) : d; }
    std::vector<int> ints(const std::string& k, const std::vector<int>& d) const {
        if (!has(k)) return d;
        std::vector<int> out;
        std::stringstream ss(kv.at(k));
        std::string t;
        while (std::getline(ss, t, ',')) out.push_back(to_int(t));
        return out;
    }
};

}  // namespace nn
