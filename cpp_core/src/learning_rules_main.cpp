// Unit III demo on e-waste crops: Hebbian vs Oja, competitive learning, and a
// restricted Boltzmann machine trained with contrastive divergence (CD-1).
//
//   learning_rules --data crops --out results [--side 32] [--epochs 15]
//
// Writes CSVs to the out folder; python/plot_results.py turns them into figures.
#include <Eigen/SVD>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "args.hpp"
#include "data.hpp"
#include "learning_rules.hpp"
#include "portable.hpp"

using namespace nn;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static void write_rows(const std::string& path, const MatrixXd& M, const std::string& first_col,
                       const std::vector<std::string>& labels) {
    std::ofstream f(path);
    f << first_col;
    for (int j = 0; j < M.cols(); ++j) f << ",p" << j;
    f << "\n";
    for (int i = 0; i < M.rows(); ++i) {
        f << (i < static_cast<int>(labels.size()) ? labels[i] : to_str(i));
        for (int j = 0; j < M.cols(); ++j) f << "," << M(i, j);
        f << "\n";
    }
}

// Average-pool square images (row-major) by a factor of 2.
static MatrixXd pool2(const MatrixXd& X, int side) {
    int s = side / 2;
    MatrixXd out(X.rows(), s * s);
    for (int i = 0; i < X.rows(); ++i)
        for (int r = 0; r < s; ++r)
            for (int c = 0; c < s; ++c)
                out(i, r * s + c) = 0.25 * (X(i, 2 * r * side + 2 * c) + X(i, 2 * r * side + 2 * c + 1) +
                                            X(i, (2 * r + 1) * side + 2 * c) + X(i, (2 * r + 1) * side + 2 * c + 1));
    return out;
}

// Binarise each image at its own mean brightness (object vs background).
static MatrixXd binarise(const MatrixXd& X) {
    MatrixXd B(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        double m = X.row(i).mean();
        for (int j = 0; j < X.cols(); ++j) B(i, j) = X(i, j) > m ? 1.0 : 0.0;
    }
    return B;
}

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        const std::string dir = a.str("data", "crops"), out = a.str("out", "results");
        const int side = a.integer("side", 32), epochs = a.integer("epochs", 15);
        Dataset tr = load_csv(dir + "/train.csv"), te = load_csv(dir + "/test.csv");
        const int k = std::max(tr.n_classes, te.n_classes);
        std::cout << "train " << tr.X.rows() << " crops, test " << te.X.rows() << ", " << k << " classes\n";

        // ============================================================ 1. Hebbian
        std::cout << "\n== 1. Hebbian learning vs Oja's rule ==\n";
        MatrixXd Xc = tr.X.rowwise() - tr.X.colwise().mean();
        Eigen::BDCSVD<MatrixXd> svd(Xc, Eigen::ComputeThinV);
        VectorXd pc1 = svd.matrixV().col(0);
        // scale so the variance along PC1 is 1: one learning rate then suits any crop size
        MatrixXd Xh = Xc / (svd.singularValues()(0) / std::sqrt(static_cast<double>(Xc.rows())));
        HebbResult hebb = train_hebbian(Xh, false, 2e-3, epochs, 1, pc1);
        HebbResult oja = train_hebbian(Xh, true, 2e-3, std::max(epochs, 30), 1, pc1);
        std::ofstream hl(out + "/hebbian_log.csv");
        if (!hl) { std::cerr << "Cannot write to '" << out << "'. Create the folder first.\n"; return 1; }
        hl << "epoch,hebb_norm,hebb_cos_pc1,oja_norm,oja_cos_pc1\n";
        for (int e = 0; e < epochs; ++e)
            hl << e + 1 << "," << hebb.norm[e] << "," << hebb.cos_to_pc1[e] << "," << oja.norm[e] << ","
               << oja.cos_to_pc1[e] << "\n";
        std::cout << "  plain Hebb: |w| = " << hebb.norm.back() << " (keeps growing)\n"
                  << "  Oja:        |w| = " << oja.norm.back() << " (stays near 1), |cos(w, PC1)| = "
                  << oja.cos_to_pc1.back() << "\n";
        MatrixXd ojaimg(2, Xh.cols());
        ojaimg.row(0) = oja.w.transpose() / oja.w.norm();
        ojaimg.row(1) = pc1.transpose() * (pc1.dot(oja.w) < 0 ? -1.0 : 1.0);
        write_rows(out + "/oja_vs_pc1.csv", ojaimg, "image", {"oja_w", "pc1_svd"});

        // 2-D toy data: shows the weight vector lining up with the data's main axis
        std::mt19937 rng(7);
        std::normal_distribution<double> N(0, 1);
        MatrixXd T(400, 2);
        for (int i = 0; i < 400; ++i) {
            double u = N(rng) * 1.6, v = N(rng) * 0.4;
            T(i, 0) = 0.8 * u - 0.6 * v;
            T(i, 1) = 0.6 * u + 0.8 * v;
        }
        Eigen::JacobiSVD<MatrixXd> s2(T, Eigen::ComputeThinV);
        VectorXd t_pc1 = s2.matrixV().col(0);
        HebbResult t_oja = train_hebbian(T, true, 0.01, 30, 3, t_pc1);
        std::ofstream t2(out + "/hebbian_2d.csv");
        t2 << "x,y\n";
        for (int i = 0; i < 400; ++i) t2 << T(i, 0) << "," << T(i, 1) << "\n";
        std::ofstream t2w(out + "/hebbian_2d_w.csv");
        t2w << "name,wx,wy\noja," << t_oja.w(0) << "," << t_oja.w(1) << "\npc1," << t_pc1(0) << "," << t_pc1(1) << "\n";

        // ======================================================= 2. Competitive
        std::cout << "\n== 2. Competitive (winner-take-all) learning, k = " << k << " ==\n";
        CompetitiveResult comp = train_competitive(tr.X, k, 0.1, epochs, 2);
        double purity = cluster_purity(comp.assign, tr.y, k);
        std::vector<int> per_class(k, 0);
        for (int i = 0; i < tr.y.size(); ++i) per_class[tr.y(i)]++;
        double baseline = static_cast<double>(*std::max_element(per_class.begin(), per_class.end())) / tr.y.size();
        std::ofstream cl(out + "/competitive_log.csv");
        cl << "epoch,quantisation_error\n";
        for (int e = 0; e < epochs; ++e) cl << e + 1 << "," << comp.quant_error[e] << "\n";
        std::vector<std::string> plabels;
        for (int c = 0; c < k; ++c) plabels.push_back("unit" + to_str(c) + "_won" + to_str(comp.wins[c]));
        write_rows(out + "/competitive_prototypes.csv", comp.W, "prototype", plabels);
        std::ofstream cs(out + "/competitive_summary.csv");
        cs << "metric,value\ncluster_purity," << purity << "\nmajority_class_baseline," << baseline
           << "\nfinal_quantisation_error," << comp.quant_error.back() << "\n";
        std::cout << "  quantisation error " << comp.quant_error.front() << " -> " << comp.quant_error.back()
                  << "\n  cluster purity vs true labels " << purity << " (majority-class baseline "
                  << baseline << "); no labels were used in training\n";

        // ================================================================ 3. RBM
        std::cout << "\n== 3. Restricted Boltzmann machine (CD-1) ==\n";
        MatrixXd Vtr = binarise(pool2(tr.X, side)), Vte = binarise(pool2(te.X, side));
        const int nv = static_cast<int>(Vtr.cols()), nh = a.integer("hidden", 64);
        RBM rbm(nv, nh, 5);
        std::ofstream rl(out + "/rbm_log.csv");
        rl << "epoch,recon_error,free_energy_train,free_energy_test\n";
        for (int e = 1; e <= epochs; ++e) {
            double err = rbm.train_epoch(Vtr, 0.05, 32);
            double ftr = rbm.free_energy(Vtr), fte = rbm.free_energy(Vte);
            rl << e << "," << err << "," << ftr << "," << fte << "\n";
            std::cout << "  epoch " << std::setw(2) << e << "  reconstruction error " << std::fixed
                      << std::setprecision(4) << err << "  free energy train " << ftr << " test " << fte << "\n";
        }
        MatrixXd filt = rbm.W().transpose().topRows(std::min(nh, 32));
        write_rows(out + "/rbm_filters.csv", filt, "hidden_unit", {});
        int m = std::min<int>(8, static_cast<int>(Vte.rows()));
        MatrixXd rec(2 * m, nv);
        rec.topRows(m) = Vte.topRows(m);
        rec.bottomRows(m) = rbm.reconstruct(Vte.topRows(m));
        std::vector<std::string> rlab;
        for (int i = 0; i < m; ++i) rlab.push_back("test_" + to_str(i));
        for (int i = 0; i < m; ++i) rlab.push_back("recon_" + to_str(i));
        write_rows(out + "/rbm_reconstructions.csv", rec, "image", rlab);
        write_rows(out + "/rbm_samples.csv", rbm.gibbs_sample(8, 500), "sample", {});
        std::cout << "  a test-vs-train free-energy gap that keeps widening would signal overfitting\n";

        std::cout << "\nWrote hebbian_*, oja_vs_pc1, competitive_*, rbm_* CSVs to " << out << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
