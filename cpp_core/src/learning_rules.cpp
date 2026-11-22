#include "learning_rules.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace nn {

namespace {
std::vector<int> shuffled(int n, std::mt19937& rng) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
    return (1.0 / (1.0 + (-z.array()).exp())).matrix();
}
}  // namespace

// ------------------------------------------------------------------ Hebbian
HebbResult train_hebbian(const Eigen::MatrixXd& X, bool oja, double eta, int epochs,
                         unsigned seed, const Eigen::VectorXd& pc1) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    HebbResult r;
    r.w = Eigen::VectorXd(X.cols());
    for (int j = 0; j < r.w.size(); ++j) r.w(j) = N(rng);
    r.w.normalize();                                  // start from a random unit vector
    for (int e = 0; e < epochs; ++e) {
        for (int i : shuffled(static_cast<int>(X.rows()), rng)) {
            Eigen::VectorXd x = X.row(i).transpose();
            double y = r.w.dot(x);
            if (oja) r.w += eta * y * (x - y * r.w);
            else r.w += eta * y * x;
            if (!std::isfinite(r.w.norm())) break;   // plain Hebb can overflow
        }
        double n = r.w.norm();
        r.norm.push_back(n);
        r.cos_to_pc1.push_back(std::isfinite(n) && n > 0 ? std::abs(r.w.dot(pc1)) / (n * pc1.norm()) : 0.0);
    }
    return r;
}

// -------------------------------------------------------------- competitive
CompetitiveResult train_competitive(const Eigen::MatrixXd& X, int k, double eta0, int epochs,
                                    unsigned seed) {
    std::mt19937 rng(seed);
    const int n = static_cast<int>(X.rows());
    CompetitiveResult r;
    r.W.resize(k, X.cols());
    auto start = shuffled(n, rng);                    // initialise on random samples
    for (int c = 0; c < k; ++c) r.W.row(c) = X.row(start[c % n]);

    auto winner = [&](const Eigen::RowVectorXd& x, double* dist) {
        Eigen::Index best;
        Eigen::VectorXd d = (r.W.rowwise() - x).rowwise().squaredNorm();
        *dist = d.minCoeff(&best);
        return static_cast<int>(best);
    };

    for (int e = 0; e < epochs; ++e) {
        double eta = eta0 * (1.0 - static_cast<double>(e) / epochs);   // decaying rate
        double qe = 0;
        for (int i : shuffled(n, rng)) {
            double d;
            int w = winner(X.row(i), &d);
            r.W.row(w) += eta * (X.row(i) - r.W.row(w));
            qe += d;
        }
        r.quant_error.push_back(qe / n);
    }
    r.assign.resize(n);
    r.wins.assign(k, 0);
    for (int i = 0; i < n; ++i) {
        double d;
        r.assign(i) = winner(X.row(i), &d);
        r.wins[r.assign(i)]++;
    }
    return r;
}

double cluster_purity(const Eigen::VectorXi& assign, const Eigen::VectorXi& labels, int k) {
    std::vector<std::map<int, int>> counts(k);
    for (int i = 0; i < assign.size(); ++i) counts[assign(i)][labels(i)]++;
    int agree = 0;
    for (const auto& m : counts) {
        int best = 0;
        for (const auto& kv : m) best = std::max(best, kv.second);
        agree += best;
    }
    return assign.size() ? static_cast<double>(agree) / assign.size() : 0.0;
}

// ---------------------------------------------------------------------- RBM
RBM::RBM(int n_visible, int n_hidden, unsigned seed) : rng_(seed) {
    std::normal_distribution<double> N(0.0, 0.01);
    W_.resize(n_visible, n_hidden);
    for (int i = 0; i < W_.size(); ++i) W_(i) = N(rng_);
    a_ = Eigen::RowVectorXd::Zero(n_visible);
    b_ = Eigen::RowVectorXd::Zero(n_hidden);
}

Eigen::MatrixXd RBM::sample(const Eigen::MatrixXd& P) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Eigen::MatrixXd S(P.rows(), P.cols());
    for (int i = 0; i < P.size(); ++i) S(i) = U(rng_) < P(i) ? 1.0 : 0.0;
    return S;
}

Eigen::MatrixXd RBM::hidden_probs(const Eigen::MatrixXd& V) const {
    Eigen::MatrixXd z = V * W_;
    z.rowwise() += b_;
    return sigmoid(z);
}

Eigen::MatrixXd RBM::visible_probs(const Eigen::MatrixXd& H) const {
    Eigen::MatrixXd z = H * W_.transpose();
    z.rowwise() += a_;
    return sigmoid(z);
}

Eigen::MatrixXd RBM::reconstruct(const Eigen::MatrixXd& V) const {
    return visible_probs(hidden_probs(V));
}

double RBM::train_epoch(const Eigen::MatrixXd& V, double eta, int batch) {
    const int n = static_cast<int>(V.rows());
    auto order = shuffled(n, rng_);
    double err = 0;
    for (int from = 0; from < n; from += batch) {
        int to = std::min(n, from + batch), m = to - from;
        Eigen::MatrixXd v0(m, V.cols());
        for (int i = from; i < to; ++i) v0.row(i - from) = V.row(order[i]);
        Eigen::MatrixXd ph0 = hidden_probs(v0);          // positive phase
        Eigen::MatrixXd h0 = sample(ph0);
        Eigen::MatrixXd pv1 = visible_probs(h0);         // one Gibbs step
        Eigen::MatrixXd ph1 = hidden_probs(pv1);         // negative phase
        W_ += eta * (v0.transpose() * ph0 - pv1.transpose() * ph1) / m;
        a_ += eta * (v0 - pv1).colwise().mean();
        b_ += eta * (ph0 - ph1).colwise().mean();
        err += (v0 - pv1).squaredNorm();
    }
    return err / (static_cast<double>(n) * V.cols());
}

Eigen::MatrixXd RBM::gibbs_sample(int n, int steps) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Eigen::MatrixXd v(n, W_.rows());
    for (int i = 0; i < v.size(); ++i) v(i) = U(rng_) < 0.5 ? 1.0 : 0.0;
    Eigen::MatrixXd pv = v;
    for (int s = 0; s < steps; ++s) {
        Eigen::MatrixXd h = sample(hidden_probs(v));
        pv = visible_probs(h);
        v = sample(pv);
    }
    return pv;
}

double RBM::free_energy(const Eigen::MatrixXd& V) const {
    // F(v) = -a.v - sum_j log(1 + exp(b_j + v.W_j))
    Eigen::MatrixXd z = V * W_;
    z.rowwise() += b_;
    Eigen::VectorXd soft = z.unaryExpr([](double t) {
        return t > 30 ? t : std::log1p(std::exp(t));
    }).rowwise().sum();
    Eigen::VectorXd lin = V * a_.transpose();
    return (-lin - soft).mean();
}

}  // namespace nn
