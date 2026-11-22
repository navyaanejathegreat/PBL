// Unit II tensor sandbox: creation, broadcasting, inner products, reshaping,
// a programmatic expression with a hand-derived gradient checked numerically,
// and SVD of the e-waste crop matrix (spectrum, low-rank compression, eigen-images).
//
//   sandbox [--data crops/train.csv] [--out results] [--side 32]
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "args.hpp"
#include "data.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

static void section(const char* t) { std::cout << "\n== " << t << " ==\n"; }

int main(int argc, char** argv) {
    nn::Args args(argc, argv);
    const std::string out = args.str("out", "results");
    const int side = args.integer("side", 32);
    Eigen::IOFormat fmt(4, 0, ", ", "\n", "  [", "]");

    section("1. Tensor creation");
    MatrixXd A(2, 3);
    A << 1, 2, 3, 4, 5, 6;
    MatrixXd Z = MatrixXd::Zero(2, 3), I = MatrixXd::Identity(3, 3);
    std::cout << "A (2x3) =\n" << A.format(fmt) << "\nzeros(2x3) and identity(3x3) also created; "
              << "A.rows()=" << A.rows() << " A.cols()=" << A.cols() << "\n";

    section("2. Broadcasting");
    Eigen::RowVector3d bias(10, 20, 30);
    Eigen::Vector2d scale(1, -1);
    MatrixXd B = A.rowwise() + bias;                       // add a row to every row
    MatrixXd C = A.array().colwise() * scale.array();      // scale each row
    std::cout << "A + bias(row, broadcast down) =\n" << B.format(fmt)
              << "\nA * scale(column, broadcast across) =\n" << C.format(fmt) << "\n";

    section("3. Inner products and matrix products");
    VectorXd u(3), v(3);
    u << 1, 2, 3;
    v << 4, 5, 6;
    std::cout << "u.v = " << u.dot(v) << "   |u| = " << u.norm()
              << "   cos(u,v) = " << u.dot(v) / (u.norm() * v.norm()) << "\n";
    std::cout << "A * I (2x3 * 3x3) =\n" << (A * I).format(fmt) << "\n";

    section("4. Reshaping (no copy: Eigen::Map views the same memory)");
    VectorXd flat(6);
    flat << 1, 2, 3, 4, 5, 6;
    Eigen::Map<MatrixXd> as3x2(flat.data(), 3, 2);         // column-major view
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> as2x3(flat.data());
    std::cout << "flat(6) viewed as 3x2 (column-major) =\n" << as3x2.format(fmt)
              << "\nflat(6) viewed as 2x3 (row-major) =\n" << as2x3.format(fmt) << "\n";
    as2x3(0, 0) = 100;
    std::cout << "after as2x3(0,0)=100, flat[0] = " << flat(0) << "  (shared memory)\n";

    section("5. Programmatic expression + gradient check");
    // f(x) = sum_i sigmoid((W x)_i)^2 ;  df/dx = W^T [ 2 s (1-s) s ]   with s = sigmoid(Wx)
    std::mt19937 rng(0);
    std::normal_distribution<double> N(0, 1);
    MatrixXd W(4, 3);
    VectorXd x(3);
    for (int i = 0; i < W.size(); ++i) W(i) = N(rng);
    for (int i = 0; i < 3; ++i) x(i) = N(rng);
    auto f = [&](const VectorXd& xx) {
        VectorXd s = (1.0 / (1.0 + (-(W * xx)).array().exp())).matrix();
        return s.squaredNorm();
    };
    VectorXd s = (1.0 / (1.0 + (-(W * x)).array().exp())).matrix();
    VectorXd analytic = W.transpose() * (2.0 * s.array().square() * (1.0 - s.array())).matrix();
    VectorXd numeric(3);
    const double eps = 1e-6;
    for (int i = 0; i < 3; ++i) {
        VectorXd xp = x, xm = x;
        xp(i) += eps;
        xm(i) -= eps;
        numeric(i) = (f(xp) - f(xm)) / (2 * eps);
    }
    double rel = (analytic - numeric).norm() / (analytic.norm() + numeric.norm());
    std::cout << "f(x) = " << f(x) << "\nanalytic grad = " << analytic.transpose()
              << "\nnumeric  grad = " << numeric.transpose() << "\nrelative error = " << rel
              << (rel < 1e-6 ? "  (PASS)" : "  (CHECK)") << "\n";

    section("6. SVD of the crop matrix");
    MatrixXd X;
    if (args.has("data")) {
        X = nn::load_csv(args.str("data", "")).X;
        if (X.rows() > 3000) X = X.topRows(3000).eval();   // keep the SVD quick
        std::cout << "using " << X.rows() << " crops of " << side << "x" << side << " from "
                  << args.str("data", "") << "\n";
    } else {
        X = MatrixXd::Random(500, side * side);
        std::cout << "no --data given: using random 500x" << side * side << " matrix\n";
    }
    Eigen::RowVectorXd mean = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - mean;
    Eigen::BDCSVD<MatrixXd> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd sv = svd.singularValues();
    double total = sv.array().square().sum();

    std::ofstream spec(out + "/svd_spectrum.csv");
    if (!spec) {
        std::cerr << "Cannot write to '" << out << "'. Create the folder first.\n";
        return 1;
    }
    spec << "k,singular_value,cumulative_energy\n";
    double cum = 0;
    int k90 = -1, k99 = -1;
    for (int k = 0; k < sv.size(); ++k) {
        cum += sv(k) * sv(k);
        spec << k + 1 << "," << sv(k) << "," << cum / total << "\n";
        if (k90 < 0 && cum / total >= 0.90) k90 = k + 1;
        if (k99 < 0 && cum / total >= 0.99) k99 = k + 1;
    }
    std::cout << "components for 90% of variance: " << k90 << ", for 99%: " << k99 << " (of "
              << sv.size() << ")\n";

    std::ofstream rec(out + "/svd_reconstruction.csv");
    rec << "rank,relative_error,compression_ratio\n";
    const MatrixXd& U = svd.matrixU();
    const MatrixXd& V = svd.matrixV();
    for (int r : {1, 2, 5, 10, 20, 50, 100, 200}) {
        if (r > sv.size()) break;
        MatrixXd Xr = U.leftCols(r) * sv.head(r).asDiagonal() * V.leftCols(r).transpose();
        double err = (Xc - Xr).norm() / Xc.norm();
        double ratio = double(X.rows() * X.cols()) / double(r * (X.rows() + X.cols() + 1));
        rec << r << "," << err << "," << ratio << "\n";
        std::cout << "  rank " << r << ": reconstruction error " << err << ", storage "
                  << ratio << "x smaller\n";
    }

    std::ofstream eig(out + "/eigen_images.csv");   // mean image + top 16 components
    eig << "component";
    for (int j = 0; j < X.cols(); ++j) eig << ",p" << j;
    eig << "\nmean";
    for (int j = 0; j < X.cols(); ++j) eig << "," << mean(j);
    for (int k = 0; k < std::min<int>(16, static_cast<int>(V.cols())); ++k) {
        eig << "\n" << k + 1;
        for (int j = 0; j < X.cols(); ++j) eig << "," << V(j, k);
    }
    eig << "\n";
    std::cout << "Wrote svd_spectrum.csv, svd_reconstruction.csv, eigen_images.csv to " << out << "\n";
    return 0;
}
