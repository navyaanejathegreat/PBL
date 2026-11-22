// Minimal self-contained unit tests (no framework needed). Run: unit_tests  or  ctest
#include <cmath>
#include <iostream>
#include <random>

#include "data.hpp"
#include "learning_rules.hpp"
#include "mlp.hpp"
#include "neurons.hpp"

using namespace nn;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (cond) std::cout << "  ok    " << msg << "\n";            \
        else { std::cout << "  FAIL  " << msg << "\n"; ++failures; }  \
    } while (0)

int main() {
    std::cout << "McCulloch-Pitts gates\n";
    MPNeuron AND{{1, 1}, 2}, OR{{1, 1}, 1};
    CHECK(AND.fire({1, 1}) == 1 && AND.fire({1, 0}) == 0 && AND.fire({0, 0}) == 0, "AND truth table");
    CHECK(OR.fire({0, 0}) == 0 && OR.fire({0, 1}) == 1 && OR.fire({1, 1}) == 1, "OR truth table");

    std::cout << "Sigmoid neuron learns AND\n";
    SigmoidNeuron s;
    auto hist = s.train({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {0, 0, 0, 1}, 1.0, 2000);
    CHECK(hist.back() < hist.front(), "loss decreases");
    CHECK(s.predict({1, 1}) > 0.5 && s.predict({0, 1}) < 0.5, "correct predictions");

    std::cout << "Softmax\n";
    Mat z(2, 3);
    z << 1, 2, 3, 1000, 1000, 1000;   // second row tests overflow handling
    Mat p = softmax_rows(z);
    CHECK(std::abs(p.row(0).sum() - 1) < 1e-12 && std::abs(p.row(1).sum() - 1) < 1e-12, "rows sum to 1");
    CHECK(std::abs(p(1, 0) - 1.0 / 3) < 1e-12, "stable for large inputs");

    std::cout << "Backprop gradient check\n";
    std::mt19937 rng(1);
    std::normal_distribution<double> N(0, 1);
    Mat X(12, 5);
    for (int i = 0; i < X.size(); ++i) X(i) = N(rng);
    Eigen::VectorXi y(12);
    for (int i = 0; i < 12; ++i) y(i) = i % 4;
    for (const char* act : {"sigmoid", "tanh", "relu"}) {
        Hyper h;
        h.hidden = {7};
        h.act = parse_act(act);
        h.init = Init::Xavier;
        h.l2 = 1e-2;
        MLP net(5, 4, h);
        CHECK(gradient_check(net, X, y, 1e-5, false) < 1e-6, std::string("gradients match (") + act + ")");
    }

    std::cout << "Optimizers reduce loss on a separable toy problem\n";
    for (const char* o : {"sgd", "momentum", "nesterov"}) {
        Hyper h;
        h.hidden = {8};
        h.opt = parse_opt(o);
        h.lr = 0.05;
        MLP net(5, 4, h);
        double before = net.loss(net.forward(X), y);
        for (int it = 0; it < 300; ++it) {
            net.forward(X);
            net.step(net.backward(y));
        }
        double after = net.loss(net.forward(X), y);
        CHECK(after < 0.5 * before, std::string(o) + " halves the loss");
    }

    std::cout << "Data helpers\n";
    Mat img(1, 4);
    img << 1, 2, 3, 4;   // 2x2 image [[1,2],[3,4]]
    Mat f = hflip_rows(img, 2);
    CHECK(f(0, 0) == 2 && f(0, 1) == 1 && f(0, 2) == 4 && f(0, 3) == 3, "horizontal flip");
    Standardizer st;
    st.fit(X);
    Mat Xs = st.transform(X);
    CHECK(Xs.colwise().mean().cwiseAbs().maxCoeff() < 1e-12, "standardized columns have mean 0");

    std::cout << "Unsupervised learning rules\n";
    {
        std::mt19937 g(3);
        std::normal_distribution<double> Nn(0, 1);
        Mat T(300, 2);
        for (int i = 0; i < 300; ++i) { double u = Nn(g) * 2, v = Nn(g) * 0.3; T(i, 0) = u; T(i, 1) = v; }
        Eigen::VectorXd pc(2); pc << 1, 0;
        HebbResult oja = train_hebbian(T, true, 0.01, 30, 1, pc);
        CHECK(oja.cos_to_pc1.back() > 0.99, "Oja's rule finds the first principal component");
        CHECK(std::abs(oja.w.norm() - 1.0) < 0.1, "Oja keeps |w| near 1");
        HebbResult hebb = train_hebbian(T, false, 0.01, 5, 1, pc);
        CHECK(hebb.norm.back() > 10, "plain Hebb grows without bound");

        Mat B(200, 2);
        Eigen::VectorXi lab(200);
        for (int i = 0; i < 200; ++i) { int c = i % 2; B(i, 0) = (c ? 5 : -5) + Nn(g) * 0.3; B(i, 1) = Nn(g) * 0.3; lab(i) = c; }
        CompetitiveResult cr = train_competitive(B, 2, 0.1, 10, 1);
        CHECK(cluster_purity(cr.assign, lab, 2) > 0.99, "competitive learning separates two clusters");
        CHECK(cr.quant_error.back() <= cr.quant_error.front(), "quantisation error does not increase");

        Mat V(120, 8);
        for (int i = 0; i < 120; ++i) for (int j = 0; j < 8; ++j) V(i, j) = (i % 2) ? (j < 4) : (j >= 4);
        RBM rbm(8, 4, 1);
        double e0 = rbm.train_epoch(V, 0.1, 10), e1 = e0;
        for (int e = 0; e < 60; ++e) e1 = rbm.train_epoch(V, 0.1, 10);
        CHECK(e1 < 0.5 * e0, "RBM reconstruction error falls with CD-1");
    }

    std::cout << (failures ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return failures ? 1 : 0;
}
