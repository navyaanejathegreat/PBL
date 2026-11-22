// Unit I demo: McCulloch-Pitts logic gates, why one neuron cannot do XOR,
// a two-layer MP network that can, a sigmoid neuron learning AND, and a
// step / sigmoid / tanh comparison table written to CSV for plotting.
//
//   neurons [out_dir]          (default: results)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "neurons.hpp"

using namespace nn;

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "results";
    const std::vector<std::vector<int>> inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    MPNeuron AND{{1, 1}, 2}, OR{{1, 1}, 1}, NAND{{-1, -1}, -1};
    MPNeuron NOT{{-1}, 0};

    std::cout << "== McCulloch-Pitts neurons (fixed weights, threshold) ==\n";
    std::cout << " x1 x2 | AND OR NAND | XOR = AND(OR, NAND)\n";
    for (const auto& x : inputs) {
        int o = OR.fire(x), na = NAND.fire(x);
        int xr = AND.fire({o, na});                 // second layer
        std::cout << "  " << x[0] << "  " << x[1] << " |  " << AND.fire(x) << "   " << o
                  << "   " << na << "  |  " << xr << "\n";
    }
    std::cout << " NOT: 0 -> " << NOT.fire({0}) << ", 1 -> " << NOT.fire({1}) << "\n";
    std::cout << "A single threshold unit draws one straight line, so it cannot separate XOR;\n"
                 "two layers (OR and NAND, then AND) can.\n\n";

    std::cout << "== Sigmoid neuron learning AND by gradient descent ==\n";
    std::vector<std::vector<double>> X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y{0, 0, 0, 1};
    SigmoidNeuron s;
    auto hist = s.train(X, y, 1.0, 2000);
    std::cout << " loss: epoch 1 = " << hist.front() << ", epoch 2000 = " << hist.back() << "\n";
    for (size_t i = 0; i < X.size(); ++i)
        std::cout << "  " << X[i][0] << " " << X[i][1] << " -> " << std::fixed << std::setprecision(3)
                  << s.predict(X[i]) << "\n";
    std::cout << " learned w = (" << s.w[0] << ", " << s.w[1] << "), b = " << s.b << "\n\n";

    std::ofstream a(out + "/activations.csv");
    std::ofstream l(out + "/sigmoid_neuron_loss.csv");
    if (!a || !l) {
        std::cerr << "Cannot write to '" << out << "'. Create the folder first.\n";
        return 1;
    }
    a << "z,step,sigmoid,tanh,d_sigmoid,d_tanh\n";
    for (int i = -60; i <= 60; ++i) {
        double z = i / 10.0, sg = sigmoid(z), th = tanh_act(z);
        a << z << "," << step(z) << "," << sg << "," << th << "," << sg * (1 - sg) << "," << 1 - th * th << "\n";
    }
    l << "epoch,loss\n";
    for (size_t e = 0; e < hist.size(); ++e) l << e + 1 << "," << hist[e] << "\n";
    std::cout << "Wrote " << out << "/activations.csv and " << out << "/sigmoid_neuron_loss.csv\n";
    std::cout << "Step has zero gradient everywhere it is defined, so it cannot be trained by\n"
                 "backprop; sigmoid saturates (max slope 0.25); tanh is zero-centred (max slope 1).\n";
    return 0;
}
