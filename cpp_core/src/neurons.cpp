#include "neurons.hpp"

#include <algorithm>
#include <cmath>

namespace nn {

int MPNeuron::fire(const std::vector<int>& x) const {
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) s += w[i] * x[i];
    return s >= threshold ? 1 : 0;
}

double step(double z) { return z >= 0.0 ? 1.0 : 0.0; }
double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
double tanh_act(double z) { return std::tanh(z); }

double SigmoidNeuron::predict(const std::vector<double>& x) const {
    double z = b;
    for (size_t i = 0; i < x.size(); ++i) z += w[i] * x[i];
    return sigmoid(z);
}

std::vector<double> SigmoidNeuron::train(const std::vector<std::vector<double>>& X,
                                         const std::vector<double>& y, double lr, int epochs) {
    if (w.empty()) w.assign(X[0].size(), 0.0);
    std::vector<double> history;
    const double n = static_cast<double>(X.size());
    for (int e = 0; e < epochs; ++e) {
        std::vector<double> gw(w.size(), 0.0);
        double gb = 0.0, loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            double p = predict(X[i]);
            double err = p - y[i];                 // dBCE/dz for a sigmoid output
            for (size_t j = 0; j < w.size(); ++j) gw[j] += err * X[i][j];
            gb += err;
            p = std::min(std::max(p, 1e-12), 1.0 - 1e-12);
            loss -= y[i] * std::log(p) + (1 - y[i]) * std::log(1 - p);
        }
        for (size_t j = 0; j < w.size(); ++j) w[j] -= lr * gw[j] / n;
        b -= lr * gb / n;
        history.push_back(loss / n);
    }
    return history;
}

}  // namespace nn
