// From-scratch multilayer perceptron: forward pass, softmax cross-entropy,
// backpropagation, and four optimizers (batch GD, SGD, Momentum, Nesterov).
// Eigen is used only as the array type; every gradient is derived by hand here.
#pragma once
#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>

namespace nn {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using RowVec = Eigen::RowVectorXd;

enum class Act { Sigmoid, Tanh, ReLU };
enum class Init { Xavier, He, SmallRandom };
enum class Opt { BatchGD, SGD, Momentum, Nesterov };

Act parse_act(const std::string& s);
Init parse_init(const std::string& s);
Opt parse_opt(const std::string& s);
std::string to_string(Act a);
std::string to_string(Init i);
std::string to_string(Opt o);

Mat activate(const Mat& z, Act a);
Mat activate_grad(const Mat& z, Act a);   // d(act)/dz evaluated at z
Mat softmax_rows(const Mat& z);           // row-wise, numerically stable

struct Layer {
    Mat W;          // in x out
    RowVec b;       // 1 x out
    Mat vW;         // velocity (Momentum / Nesterov)
    RowVec vb;
};

struct Grads {
    std::vector<Mat> dW;
    std::vector<RowVec> db;
};

struct Hyper {                       // hyperparameters: chosen, never learned
    std::vector<int> hidden{128};
    Act act = Act::ReLU;
    Init init = Init::He;
    Opt opt = Opt::Momentum;
    double lr = 0.05;
    double momentum = 0.9;
    double l2 = 0.0;
    int batch = 64;
    int epochs = 30;
    unsigned seed = 0;
};

class MLP {
public:
    MLP(int in_dim, int n_classes, const Hyper& h);

    Mat forward(const Mat& X);                        // caches Z/A; returns probabilities
    double loss(const Mat& P, const Eigen::VectorXi& y) const;   // CE + L2
    Grads backward(const Eigen::VectorXi& y) const;   // uses cache from last forward
    void step(const Grads& g);                        // one optimizer update
    Eigen::VectorXi predict(const Mat& X);

    long n_parameters() const;                        // learned parameters
    std::vector<Layer>& layers() { return L_; }
    const Hyper& hyper() const { return h_; }

    bool save(const std::string& path) const;

private:
    Hyper h_;
    std::vector<Layer> L_;
    std::vector<Mat> Z_, A_;   // A_[0] = input
};

// Relative error between analytic and central-difference gradients
// on a small random problem. Returns the worst layer's relative error.
double gradient_check(MLP& net, const Mat& X, const Eigen::VectorXi& y,
                      double eps = 1e-5, bool verbose = true);

}  // namespace nn
