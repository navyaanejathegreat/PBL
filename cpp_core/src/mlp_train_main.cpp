// From-scratch MLP trainer for e-waste crops (Units I and III).
//
// Commands
//   mlp_train gradcheck
//   mlp_train train       --data crops [options]           one run
//   mlp_train compare-opt --data crops [options]           GD vs SGD vs Momentum vs Nesterov
//   mlp_train ablation    --data crops [options]           one change at a time vs baseline
//
// Options (hyperparameters)
//   --hidden 256,128  --act relu|sigmoid|tanh  --init he|xavier|small
//   --opt gd|sgd|momentum|nesterov  --lr 0.05  --momentum 0.9  --l2 0
//   --batch 64  --epochs 30  --seed 0  --no-norm  --augment  --out results
//
// The data folder (written by python/export_crops.py) holds
//   train.csv, val.csv, test.csv, classes.txt
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "args.hpp"
#include "data.hpp"
#include "mlp.hpp"

using namespace nn;

namespace {

struct Flags {
    bool normalize = true;
    bool augment = false;
    int side = 32;
};

struct Result {
    std::string name;
    double train_loss = 0, train_acc = 0, val_loss = 0, val_acc = 0;
    double best_val_acc = 0, test_acc = 0, test_macro_f1 = 0, seconds = 0;
    int best_epoch = 0;
    long params = 0;
};

double accuracy(const Eigen::VectorXi& p, const Eigen::VectorXi& y) {
    return (p.array() == y.array()).cast<double>().mean();
}

Eigen::MatrixXi confusion(const Eigen::VectorXi& p, const Eigen::VectorXi& y, int k) {
    Eigen::MatrixXi C = Eigen::MatrixXi::Zero(k, k);
    for (int i = 0; i < y.size(); ++i) C(y(i), p(i)) += 1;
    return C;
}

double macro_f1(const Eigen::MatrixXi& C) {
    double sum = 0;
    int counted = 0;
    for (int c = 0; c < C.rows(); ++c) {
        double tp = C(c, c), fp = C.col(c).sum() - tp, fn = C.row(c).sum() - tp;
        if (tp + fn == 0) continue;               // class absent from this split
        double prec = tp + fp > 0 ? tp / (tp + fp) : 0, rec = tp / (tp + fn);
        sum += prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
        ++counted;
    }
    return counted ? sum / counted : 0;
}

Hyper hyper_from(const Args& a) {
    Hyper h;
    h.hidden = a.ints("hidden", {256, 128});
    h.act = parse_act(a.str("act", "relu"));
    h.init = parse_init(a.str("init", h.act == Act::ReLU ? "he" : "xavier"));
    h.opt = parse_opt(a.str("opt", "momentum"));
    h.lr = a.num("lr", 0.05);
    h.momentum = a.num("momentum", 0.9);
    h.l2 = a.num("l2", 0.0);
    h.batch = a.integer("batch", 64);
    h.epochs = a.integer("epochs", 30);
    h.seed = static_cast<unsigned>(a.integer("seed", 0));
    return h;
}

void write_config(const std::string& path, const std::string& name, const Hyper& h,
                  const Flags& fl, long params) {
    std::ofstream f(path);
    f << "run," << name << "\n"
      << "# hyperparameters (chosen by us)\n"
      << "hidden,";
    for (size_t i = 0; i < h.hidden.size(); ++i) f << (i ? "-" : "") << h.hidden[i];
    f << "\nactivation," << to_string(h.act) << "\ninit," << to_string(h.init)
      << "\noptimizer," << to_string(h.opt) << "\nlr," << h.lr << "\nmomentum," << h.momentum
      << "\nl2," << h.l2 << "\nbatch," << h.batch << "\nepochs," << h.epochs << "\nseed," << h.seed
      << "\nnormalize," << fl.normalize << "\naugment_hflip," << fl.augment
      << "\n# parameters (learned by backprop)\nlearned_parameters," << params << "\n";
}

Result run(const std::string& name, Hyper h, const Flags& fl, const Dataset& tr_raw,
           const Dataset& va_raw, const Dataset& te_raw, const std::string& out) {
    Dataset tr = tr_raw, va = va_raw, te = te_raw;
    if (fl.normalize) {
        Standardizer s;
        s.fit(tr.X);                                // train statistics only
        tr.X = s.transform(tr.X);
        va.X = s.transform(va.X);
        te.X = s.transform(te.X);
    }
    const int n = static_cast<int>(tr.X.rows());
    if (h.opt == Opt::BatchGD) h.batch = n;         // full-batch gradient descent

    MLP net(static_cast<int>(tr.X.cols()), tr.n_classes, h);
    std::mt19937 rng(h.seed + 1);
    std::bernoulli_distribution coin(0.5);

    Result r;
    r.name = name;
    r.params = net.n_parameters();
    write_config(out + "/" + name + "_config.csv", name, h, fl, r.params);
    std::ofstream log(out + "/" + name + "_log.csv");
    log << "epoch,train_loss,train_acc,val_loss,val_acc,seconds\n";

    auto t0 = std::chrono::steady_clock::now();
    for (int e = 1; e <= h.epochs; ++e) {
        auto perm = permutation(n, rng);
        for (int from = 0; from < n; from += h.batch) {
            int to = std::min(n, from + h.batch);
            Eigen::MatrixXd xb = take_rows(tr.X, perm, from, to);
            Eigen::VectorXi yb = take_rows(tr.y, perm, from, to);
            if (fl.augment && coin(rng)) xb = hflip_rows(xb, fl.side);
            net.forward(xb);
            net.step(net.backward(yb));
        }
        Eigen::MatrixXd Ptr = net.forward(tr.X);
        r.train_loss = net.loss(Ptr, tr.y);
        r.train_acc = accuracy(net.predict(tr.X), tr.y);
        Eigen::MatrixXd Pva = net.forward(va.X);
        r.val_loss = net.loss(Pva, va.y);
        r.val_acc = accuracy(net.predict(va.X), va.y);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        log << e << "," << r.train_loss << "," << r.train_acc << "," << r.val_loss << "," << r.val_acc
            << "," << secs << "\n";
        if (r.val_acc > r.best_val_acc) {
            r.best_val_acc = r.val_acc;
            r.best_epoch = e;
        }
        std::cout << "  [" << name << "] epoch " << std::setw(3) << e << "  train loss "
                  << std::fixed << std::setprecision(4) << r.train_loss << " acc " << r.train_acc
                  << " | val loss " << r.val_loss << " acc " << r.val_acc << "\n";
    }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    Eigen::VectorXi pte = net.predict(te.X);
    r.test_acc = accuracy(pte, te.y);
    Eigen::MatrixXi C = confusion(pte, te.y, tr.n_classes);
    r.test_macro_f1 = macro_f1(C);
    std::ofstream cf(out + "/" + name + "_confusion.csv");
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) cf << (j ? "," : "") << C(i, j);
        cf << "\n";
    }
    net.save(out + "/" + name + "_weights.bin");
    return r;
}

void write_summary(const std::string& path, const std::vector<Result>& rs) {
    std::ofstream f(path);
    f << "run,params,final_train_loss,final_train_acc,final_val_loss,final_val_acc,"
         "best_val_acc,best_epoch,test_acc,test_macro_f1,seconds\n";
    for (const auto& r : rs)
        f << r.name << "," << r.params << "," << r.train_loss << "," << r.train_acc << ","
          << r.val_loss << "," << r.val_acc << "," << r.best_val_acc << "," << r.best_epoch << ","
          << r.test_acc << "," << r.test_macro_f1 << "," << r.seconds << "\n";
    std::cout << "\n" << std::left << std::setw(22) << "run" << std::setw(12) << "train acc"
              << std::setw(12) << "val acc" << std::setw(12) << "test acc" << "macro F1\n";
    for (const auto& r : rs)
        std::cout << std::setw(22) << r.name << std::setw(12) << r.train_acc << std::setw(12)
                  << r.val_acc << std::setw(12) << r.test_acc << r.test_macro_f1 << "\n";
    std::cout << "Summary written to " << path << "\n";
}

int gradcheck() {
    std::cout << "Gradient check on a random 20x8 batch, 8-6-5-3 network\n";
    std::mt19937 rng(0);
    std::normal_distribution<double> N(0, 1);
    Eigen::MatrixXd X(20, 8);
    for (int i = 0; i < X.size(); ++i) X(i) = N(rng);
    Eigen::VectorXi y(20);
    for (int i = 0; i < 20; ++i) y(i) = i % 3;
    int fails = 0;
    for (const char* act : {"sigmoid", "tanh", "relu"}) {
        Hyper h;
        h.hidden = {6, 5};
        h.act = parse_act(act);
        h.init = Init::Xavier;
        h.l2 = 1e-3;
        MLP net(8, 3, h);
        std::cout << act << ":\n";
        double worst = gradient_check(net, X, y);
        bool ok = worst < 1e-6;
        fails += !ok;
        std::cout << "  worst relative error " << worst << (ok ? "  PASS\n" : "  FAIL\n");
    }
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        if (a.command == "gradcheck") return gradcheck();
        if (a.command != "train" && a.command != "compare-opt" && a.command != "ablation") {
            std::cerr << "usage: mlp_train gradcheck | train | compare-opt | ablation --data <folder> [options]\n";
            return 2;
        }
        const std::string dir = a.str("data", "crops"), out = a.str("out", "results");
        Dataset tr = load_csv(dir + "/train.csv"), va = load_csv(dir + "/val.csv"),
                te = load_csv(dir + "/test.csv");
        int k = std::max({tr.n_classes, va.n_classes, te.n_classes});
        tr.n_classes = va.n_classes = te.n_classes = k;
        std::cout << "train " << tr.X.rows() << ", val " << va.X.rows() << ", test " << te.X.rows()
                  << " samples; " << tr.X.cols() << " features; " << k << " classes\n";

        Hyper h = hyper_from(a);
        Flags fl;
        fl.normalize = !a.has("no-norm");
        fl.augment = a.has("augment");
        fl.side = a.integer("side", 32);

        std::vector<Result> rs;
        if (a.command == "train") {
            rs.push_back(run(a.str("name", "mlp"), h, fl, tr, va, te, out));
            write_summary(out + "/train_summary.csv", rs);
        } else if (a.command == "compare-opt") {
            const std::map<std::string, double> lr_default{
                {"gd", 0.5}, {"sgd", 0.05}, {"momentum", 0.01}, {"nesterov", 0.01}};
            for (const char* o : {"gd", "sgd", "momentum", "nesterov"}) {
                Hyper hh = h;
                hh.opt = parse_opt(o);
                hh.lr = a.has("lr") ? h.lr : lr_default.at(o);
                rs.push_back(run(std::string("opt_") + o, hh, fl, tr, va, te, out));
            }
            write_summary(out + "/compare_optimizers.csv", rs);
        } else {   // ablation: baseline, then change exactly one thing
            rs.push_back(run("baseline", h, fl, tr, va, te, out));
            Hyper small = h;
            small.init = Init::SmallRandom;
            rs.push_back(run("init_small", small, fl, tr, va, te, out));
            Hyper xav = h;
            xav.init = h.init == Init::He ? Init::Xavier : Init::He;
            rs.push_back(run(std::string("init_") + to_string(xav.init), xav, fl, tr, va, te, out));
            Flags nonorm = fl;
            nonorm.normalize = false;
            rs.push_back(run("no_normalization", h, nonorm, tr, va, te, out));
            Hyper reg = h;
            reg.l2 = h.l2 > 0 ? 0.0 : 1e-3;
            rs.push_back(run(reg.l2 > 0 ? "l2_1e-3" : "l2_off", reg, fl, tr, va, te, out));
            Flags aug = fl;
            aug.augment = !fl.augment;
            rs.push_back(run(aug.augment ? "augment_hflip" : "augment_off", h, aug, tr, va, te, out));
            for (const char* act : {"sigmoid", "tanh"}) {
                Hyper ha = h;
                ha.act = parse_act(act);
                ha.init = Init::Xavier;
                rs.push_back(run(std::string("act_") + act, ha, fl, tr, va, te, out));
            }
            write_summary(out + "/ablation.csv", rs);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
