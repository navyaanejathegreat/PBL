// Single-neuron models from Unit I: McCulloch-Pitts threshold unit and a
// sigmoid neuron trained by gradient descent.
#pragma once
#include <vector>

namespace nn {

// McCulloch-Pitts: binary inputs, fixed weights, fires (1) when the weighted
// sum reaches the threshold. Nothing is learned.
struct MPNeuron {
    std::vector<double> w;
    double threshold;
    int fire(const std::vector<int>& x) const;
};

double step(double z);
double sigmoid(double z);
double tanh_act(double z);

// Sigmoid neuron trained with batch gradient descent on binary cross-entropy.
struct SigmoidNeuron {
    std::vector<double> w;
    double b = 0.0;
    double predict(const std::vector<double>& x) const;
    // Returns the loss after each epoch.
    std::vector<double> train(const std::vector<std::vector<double>>& X,
                              const std::vector<double>& y, double lr, int epochs);
};

}  // namespace nn
