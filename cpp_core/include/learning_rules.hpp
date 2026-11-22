// Unsupervised learning rules (Unit III): Hebbian / Oja, competitive
// (winner-take-all) learning, and a restricted Boltzmann machine trained
// with contrastive divergence. All hand-written; Eigen is only the array type.
#pragma once
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace nn {

// ---------- Hebbian ----------
// Plain Hebb:  dw = eta * y * x            (weights grow without bound)
// Oja's rule:  dw = eta * y * (x - y * w)  (normalised; converges to the first
//                                           principal component of the data)
struct HebbResult {
    Eigen::VectorXd w;
    std::vector<double> norm;          // |w| after each epoch
    std::vector<double> cos_to_pc1;    // |cos(w, PC1)| after each epoch
};
HebbResult train_hebbian(const Eigen::MatrixXd& X, bool oja, double eta, int epochs,
                         unsigned seed, const Eigen::VectorXd& pc1);

// ---------- Competitive learning ----------
// k prototypes; for each sample only the closest prototype (the winner)
// moves towards it:  w_win += eta * (x - w_win).
struct CompetitiveResult {
    Eigen::MatrixXd W;                 // k x d prototypes
    std::vector<double> quant_error;   // mean squared distance to winner per epoch
    Eigen::VectorXi assign;            // winner for each sample after training
    std::vector<int> wins;             // how many samples each prototype won
};
CompetitiveResult train_competitive(const Eigen::MatrixXd& X, int k, double eta0, int epochs,
                                    unsigned seed);
// Fraction of samples whose cluster's majority label equals their own label.
double cluster_purity(const Eigen::VectorXi& assign, const Eigen::VectorXi& labels, int k);

// ---------- Restricted Boltzmann machine ----------
// Binary visible and hidden units, energy E(v,h) = -a.v - b.h - v'Wh.
// Trained with CD-1: dW = eta * (v0 h0' - v1 h1') / n.
class RBM {
public:
    RBM(int n_visible, int n_hidden, unsigned seed);
    double train_epoch(const Eigen::MatrixXd& V, double eta, int batch);   // returns recon error
    Eigen::MatrixXd hidden_probs(const Eigen::MatrixXd& V) const;
    Eigen::MatrixXd visible_probs(const Eigen::MatrixXd& H) const;
    Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& V) const;           // v -> p(h) -> p(v)
    Eigen::MatrixXd gibbs_sample(int n, int steps);                        // samples from the model
    double free_energy(const Eigen::MatrixXd& V) const;                    // mean F(v)
    const Eigen::MatrixXd& W() const { return W_; }

private:
    Eigen::MatrixXd W_;        // n_visible x n_hidden
    Eigen::RowVectorXd a_, b_; // visible and hidden biases
    std::mt19937 rng_;
    Eigen::MatrixXd sample(const Eigen::MatrixXd& P);
};

}  // namespace nn
