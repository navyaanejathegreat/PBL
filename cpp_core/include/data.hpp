// Dataset loading and preparation for the from-scratch MLP.
// CSV format written by python/export_crops.py:
//   label,p0,p1,...,p1023      (32x32 grayscale crop, values 0..1)
#pragma once
#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>

namespace nn {

struct Dataset {
    Eigen::MatrixXd X;          // n x d
    Eigen::VectorXi y;          // n
    int n_classes = 0;
};

Dataset load_csv(const std::string& path);
std::vector<std::string> load_class_names(const std::string& path);   // one per line

// Standardization: statistics come from TRAIN only, then applied to every split.
struct Standardizer {
    Eigen::RowVectorXd mean, stdev;
    void fit(const Eigen::MatrixXd& X);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
};

// Horizontal flip of square images stored row-major in each row of X.
Eigen::MatrixXd hflip_rows(const Eigen::MatrixXd& X, int side);

// Random permutation of 0..n-1.
std::vector<int> permutation(int n, std::mt19937& rng);

Eigen::MatrixXd take_rows(const Eigen::MatrixXd& X, const std::vector<int>& idx, int from, int to);
Eigen::VectorXi take_rows(const Eigen::VectorXi& y, const std::vector<int>& idx, int from, int to);

}  // namespace nn
