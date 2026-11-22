#include "mlp.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace nn {

// ------------------------------------------------------------ names
Act parse_act(const std::string& s) {
    if (s == "sigmoid") return Act::Sigmoid;
    if (s == "tanh") return Act::Tanh;
    if (s == "relu") return Act::ReLU;
    throw std::invalid_argument("act must be sigmoid|tanh|relu, got " + s);
}
Init parse_init(const std::string& s) {
    if (s == "xavier") return Init::Xavier;
    if (s == "he") return Init::He;
    if (s == "small") return Init::SmallRandom;
    throw std::invalid_argument("init must be xavier|he|small, got " + s);
}
Opt parse_opt(const std::string& s) {
    if (s == "gd") return Opt::BatchGD;
    if (s == "sgd") return Opt::SGD;
    if (s == "momentum") return Opt::Momentum;
    if (s == "nesterov") return Opt::Nesterov;
    throw std::invalid_argument("opt must be gd|sgd|momentum|nesterov, got " + s);
}
std::string to_string(Act a) { return a == Act::Sigmoid ? "sigmoid" : a == Act::Tanh ? "tanh" : "relu"; }
std::string to_string(Init i) { return i == Init::Xavier ? "xavier" : i == Init::He ? "he" : "small"; }
std::string to_string(Opt o) {
    switch (o) {
        case Opt::BatchGD: return "gd";
        case Opt::SGD: return "sgd";
        case Opt::Momentum: return "momentum";
        default: return "nesterov";
    }
}

// ------------------------------------------------------------ activations
Mat activate(const Mat& z, Act a) {
    switch (a) {
        case Act::Sigmoid: return (1.0 / (1.0 + (-z.array()).exp())).matrix();
        case Act::Tanh: return z.array().tanh().matrix();
        default: return z.cwiseMax(0.0);
    }
}

Mat activate_grad(const Mat& z, Act a) {
    switch (a) {
        case Act::Sigmoid: {
            Mat s = activate(z, Act::Sigmoid);
            return (s.array() * (1.0 - s.array())).matrix();
        }
        case Act::Tanh: {
            Mat t = z.array().tanh().matrix();
            return (1.0 - t.array().square()).matrix();
        }
        default: return (z.array() > 0.0).cast<double>().matrix();
    }
}

Mat softmax_rows(const Mat& z) {
    // subtract each row's max (broadcast) for numerical stability
    Mat shifted = z.colwise() - z.rowwise().maxCoeff();
    Mat e = shifted.array().exp().matrix();
    Vec sums = e.rowwise().sum();
    return e.array().colwise() / sums.array();
}

// ------------------------------------------------------------ model
MLP::MLP(int in_dim, int n_classes, const Hyper& h) : h_(h) {
    std::mt19937 rng(h.seed);
    std::vector<int> dims{in_dim};
    dims.insert(dims.end(), h.hidden.begin(), h.hidden.end());
    dims.push_back(n_classes);

    for (size_t l = 0; l + 1 < dims.size(); ++l) {
        int fin = dims[l], fout = dims[l + 1];
        double sd;
        switch (h.init) {
            case Init::Xavier: sd = std::sqrt(2.0 / (fin + fout)); break;
            case Init::He: sd = std::sqrt(2.0 / fin); break;
            default: sd = 0.01; break;
        }
        std::normal_distribution<double> N(0.0, sd);
        Layer layer;
        layer.W = Mat(fin, fout);
        for (int i = 0; i < fin; ++i)
            for (int j = 0; j < fout; ++j) layer.W(i, j) = N(rng);
        layer.b = RowVec::Zero(fout);
        layer.vW = Mat::Zero(fin, fout);
        layer.vb = RowVec::Zero(fout);
        L_.push_back(std::move(layer));
    }
}

Mat MLP::forward(const Mat& X) {
    Z_.clear();
    A_.clear();
    A_.push_back(X);
    for (size_t l = 0; l < L_.size(); ++l) {
        Mat z = A_.back() * L_[l].W;
        z.rowwise() += L_[l].b;                 // broadcasting the bias over the batch
        Z_.push_back(z);
        bool last = (l + 1 == L_.size());
        A_.push_back(last ? softmax_rows(z) : activate(z, h_.act));
    }
    return A_.back();
}

double MLP::loss(const Mat& P, const Eigen::VectorXi& y) const {
    double ce = 0.0;
    for (int i = 0; i < P.rows(); ++i) ce -= std::log(std::max(P(i, y(i)), 1e-12));
    ce /= P.rows();
    double reg = 0.0;
    if (h_.l2 > 0)
        for (const auto& layer : L_) reg += layer.W.squaredNorm();
    return ce + 0.5 * h_.l2 * reg;
}

Grads MLP::backward(const Eigen::VectorXi& y) const {
    const int n = static_cast<int>(y.size());
    const size_t nl = L_.size();
    Grads g;
    g.dW.resize(nl);
    g.db.resize(nl);

    // softmax + cross-entropy: dL/dz = (P - onehot(y)) / n
    Mat dZ = A_.back();
    for (int i = 0; i < n; ++i) dZ(i, y(i)) -= 1.0;
    dZ /= n;

    for (int l = static_cast<int>(nl) - 1; l >= 0; --l) {
        g.dW[l] = A_[l].transpose() * dZ + h_.l2 * L_[l].W;
        g.db[l] = dZ.colwise().sum();
        if (l > 0) {
            Mat dA = dZ * L_[l].W.transpose();
            dZ = dA.cwiseProduct(activate_grad(Z_[l - 1], h_.act));
        }
    }
    return g;
}

void MLP::step(const Grads& g) {
    const double lr = h_.lr, mu = h_.momentum;
    for (size_t l = 0; l < L_.size(); ++l) {
        Layer& p = L_[l];
        switch (h_.opt) {
            case Opt::BatchGD:
            case Opt::SGD:   // same update; they differ in batch size (set by the trainer)
                p.W -= lr * g.dW[l];
                p.b -= lr * g.db[l];
                break;
            case Opt::Momentum:
                p.vW = mu * p.vW - lr * g.dW[l];
                p.vb = mu * p.vb - lr * g.db[l];
                p.W += p.vW;
                p.b += p.vb;
                break;
            case Opt::Nesterov: {
                // Sutskever et al. (2013) form: the stored weights are the look-ahead point.
                Mat vW_prev = p.vW;
                RowVec vb_prev = p.vb;
                p.vW = mu * p.vW - lr * g.dW[l];
                p.vb = mu * p.vb - lr * g.db[l];
                p.W += -mu * vW_prev + (1.0 + mu) * p.vW;
                p.b += -mu * vb_prev + (1.0 + mu) * p.vb;
                break;
            }
        }
    }
}

Eigen::VectorXi MLP::predict(const Mat& X) {
    Mat P = forward(X);
    Eigen::VectorXi out(P.rows());
    for (int i = 0; i < P.rows(); ++i) {
        Eigen::Index j;
        P.row(i).maxCoeff(&j);
        out(i) = static_cast<int>(j);
    }
    return out;
}

long MLP::n_parameters() const {
    long n = 0;
    for (const auto& layer : L_) n += layer.W.size() + layer.b.size();
    return n;
}

bool MLP::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int nl = static_cast<int>(L_.size());
    f.write(reinterpret_cast<const char*>(&nl), sizeof nl);
    for (const auto& layer : L_) {
        int r = static_cast<int>(layer.W.rows()), c = static_cast<int>(layer.W.cols());
        f.write(reinterpret_cast<const char*>(&r), sizeof r);
        f.write(reinterpret_cast<const char*>(&c), sizeof c);
        f.write(reinterpret_cast<const char*>(layer.W.data()), sizeof(double) * layer.W.size());
        f.write(reinterpret_cast<const char*>(layer.b.data()), sizeof(double) * layer.b.size());
    }
    return static_cast<bool>(f);
}

// ------------------------------------------------------------ gradient check
double gradient_check(MLP& net, const Mat& X, const Eigen::VectorXi& y, double eps, bool verbose) {
    net.forward(X);
    Grads g = net.backward(y);
    double worst = 0.0;
    auto& Ls = net.layers();
    for (size_t l = 0; l < Ls.size(); ++l) {
        Mat num = Mat::Zero(Ls[l].W.rows(), Ls[l].W.cols());
        for (int i = 0; i < num.rows(); ++i) {
            for (int j = 0; j < num.cols(); ++j) {
                double keep = Ls[l].W(i, j);
                Ls[l].W(i, j) = keep + eps;
                double lp = net.loss(net.forward(X), y);
                Ls[l].W(i, j) = keep - eps;
                double lm = net.loss(net.forward(X), y);
                Ls[l].W(i, j) = keep;
                num(i, j) = (lp - lm) / (2 * eps);
            }
        }
        double rel = (num - g.dW[l]).norm() / std::max(1e-12, num.norm() + g.dW[l].norm());
        worst = std::max(worst, rel);
        if (verbose)
            std::cout << "  layer " << l << " W " << Ls[l].W.rows() << "x" << Ls[l].W.cols()
                      << "  relative error = " << rel << "\n";
    }
    return worst;
}

}  // namespace nn
