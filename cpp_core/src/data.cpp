#include "data.hpp"
#include "portable.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace nn {

Dataset load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::vector<int> labels;
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        labels.push_back(to_int(cell));
        std::vector<double> r;
        while (std::getline(ss, cell, ',')) r.push_back(to_double(cell));
        rows.push_back(std::move(r));
    }
    if (rows.empty()) throw std::runtime_error("no rows in " + path);
    Dataset d;
    d.X.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(rows[0].size()));
    d.y.resize(static_cast<Eigen::Index>(rows.size()));
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != rows[0].size()) throw std::runtime_error("ragged row in " + path);
        for (size_t j = 0; j < rows[i].size(); ++j) d.X(i, j) = rows[i][j];
        d.y(i) = labels[i];
    }
    d.n_classes = d.y.maxCoeff() + 1;
    return d;
}

std::vector<std::string> load_class_names(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) {
            if (line.back() == '\r') line.pop_back();
            names.push_back(line);
        }
    return names;
}

void Standardizer::fit(const Eigen::MatrixXd& X) {
    mean = X.colwise().mean();
    Eigen::MatrixXd c = X.rowwise() - mean;                // broadcast subtract
    stdev = (c.array().square().colwise().sum() / X.rows()).sqrt().matrix();
    stdev = stdev.cwiseMax(1e-8);
}

Eigen::MatrixXd Standardizer::transform(const Eigen::MatrixXd& X) const {
    return ((X.rowwise() - mean).array().rowwise() / stdev.array()).matrix();
}

Eigen::MatrixXd hflip_rows(const Eigen::MatrixXd& X, int side) {
    Eigen::MatrixXd out(X.rows(), X.cols());
    for (Eigen::Index i = 0; i < X.rows(); ++i)
        for (int r = 0; r < side; ++r)
            for (int c = 0; c < side; ++c) out(i, r * side + c) = X(i, r * side + (side - 1 - c));
    return out;
}

std::vector<int> permutation(int n, std::mt19937& rng) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

Eigen::MatrixXd take_rows(const Eigen::MatrixXd& X, const std::vector<int>& idx, int from, int to) {
    Eigen::MatrixXd out(to - from, X.cols());
    for (int i = from; i < to; ++i) out.row(i - from) = X.row(idx[i]);
    return out;
}

Eigen::VectorXi take_rows(const Eigen::VectorXi& y, const std::vector<int>& idx, int from, int to) {
    Eigen::VectorXi out(to - from);
    for (int i = from; i < to; ++i) out(i - from) = y(idx[i]);
    return out;
}

}  // namespace nn
