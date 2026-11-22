"""
Turn the CSV outputs of the C++ tools into figures for the report.

    python plot_results.py --results ..\\results --classes ..\\crops\\classes.txt

Makes (whichever inputs exist):
    activations.png        step / sigmoid / tanh and their slopes
    sigmoid_neuron.png     loss of the single sigmoid neuron learning AND
    svd_spectrum.png       singular values and cumulative variance
    svd_reconstruction.png reconstruction error vs rank
    eigen_images.png       mean crop + top principal components as 32x32 images
    optimizers.png         GD vs SGD vs Momentum vs Nesterov (loss and val acc)
    ablation.png           train / val / test accuracy per ablation
    confusion_<run>.png    confusion matrix of the chosen run
    hebbian*.png, oja_vs_pc1.png, competitive_*.png, rbm_*.png   learning_rules outputs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--classes", default="crops/classes.txt")
    ap.add_argument("--confusion-run", default="baseline")
    a = ap.parse_args()
    r = Path(a.results)

    if (r / "activations.csv").exists():
        d = pd.read_csv(r / "activations.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for c in ["step", "sigmoid", "tanh"]:
            ax[0].plot(d.z, d[c], label=c)
        ax[0].set_title("Activation functions"); ax[0].legend(); ax[0].set_xlabel("z")
        ax[1].plot(d.z, d.d_sigmoid, label="sigmoid'")
        ax[1].plot(d.z, d.d_tanh, label="tanh'")
        ax[1].plot(d.z, 0 * d.z, label="step' (0 almost everywhere)")
        ax[1].set_title("Slopes used by backprop"); ax[1].legend(); ax[1].set_xlabel("z")
        save(fig, r / "activations.png")

    if (r / "sigmoid_neuron_loss.csv").exists():
        d = pd.read_csv(r / "sigmoid_neuron_loss.csv")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(d.epoch, d.loss); ax.set_yscale("log")
        ax.set_title("Sigmoid neuron learning AND"); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
        save(fig, r / "sigmoid_neuron.png")

    if (r / "svd_spectrum.csv").exists():
        d = pd.read_csv(r / "svd_spectrum.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogy(d.k, d.singular_value); ax[0].set_title("Singular values of the crop matrix")
        ax[0].set_xlabel("component")
        ax[1].plot(d.k, d.cumulative_energy); ax[1].axhline(0.9, ls="--", c="grey")
        ax[1].axhline(0.99, ls=":", c="grey"); ax[1].set_title("Cumulative variance explained")
        ax[1].set_xlabel("components kept")
        save(fig, r / "svd_spectrum.png")

    if (r / "svd_reconstruction.csv").exists():
        d = pd.read_csv(r / "svd_reconstruction.csv")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(d["rank"], d.relative_error, marker="o")
        for _, row in d.iterrows():
            ax.annotate(f"{row.compression_ratio:.0f}x", (row["rank"], row.relative_error),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xscale("log"); ax.set_title("Low-rank compression (labels: storage saving)")
        ax.set_xlabel("rank"); ax.set_ylabel("relative reconstruction error")
        save(fig, r / "svd_reconstruction.png")

    if (r / "eigen_images.csv").exists():
        d = pd.read_csv(r / "eigen_images.csv")
        side = int(round(np.sqrt(d.shape[1] - 1)))
        n = len(d)
        cols = 6
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
        for i, ax in enumerate(np.ravel(axs)):
            ax.axis("off")
            if i < n:
                img = d.iloc[i, 1:].to_numpy(float).reshape(side, side)
                ax.imshow(img, cmap="gray" if i == 0 else "RdBu")
                ax.set_title("mean" if i == 0 else f"PC {d.iloc[i, 0]}", fontsize=8)
        fig.suptitle("Eigen-images of e-waste crops (SVD)")
        save(fig, r / "eigen_images.png")

    logs = {o: r / f"opt_{o}_log.csv" for o in ["gd", "sgd", "momentum", "nesterov"]}
    if any(p.exists() for p in logs.values()):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for o, p in logs.items():
            if p.exists():
                d = pd.read_csv(p)
                ax[0].plot(d.epoch, d.train_loss, label=o)
                ax[1].plot(d.epoch, d.val_acc, label=o)
        ax[0].set_title("Training loss"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_xlabel("epoch")
        ax[1].set_title("Validation accuracy"); ax[1].legend(); ax[1].set_xlabel("epoch")
        save(fig, r / "optimizers.png")

    if (r / "ablation.csv").exists():
        d = pd.read_csv(r / "ablation.csv")
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (col, lab) in enumerate([("final_train_acc", "train"), ("final_val_acc", "val"),
                                        ("test_acc", "test")]):
            ax.bar(x + (i - 1) * 0.27, d[col], 0.27, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(d.run, rotation=30, ha="right")
        ax.set_ylabel("accuracy"); ax.set_title("Ablation: one change at a time vs baseline")
        ax.legend()
        save(fig, r / "ablation.png")


    # ---------------- unsupervised learning rules (learning_rules) ----------------
    def grid(rows, title, path, cmap="gray", labels=None, cols=8):
        side = int(round(np.sqrt(rows.shape[1])))
        n = len(rows)
        cols = min(cols, n)
        r = int(np.ceil(n / cols))
        fig, axs = plt.subplots(r, cols, figsize=(cols * 1.4, r * 1.55))
        for i, ax in enumerate(np.ravel(axs)):
            ax.axis("off")
            if i < n:
                ax.imshow(rows[i].reshape(side, side), cmap=cmap)
                if labels is not None:
                    ax.set_title(labels[i], fontsize=7)
        fig.suptitle(title)
        save(fig, path)

    if (r / "hebbian_log.csv").exists():
        d = pd.read_csv(r / "hebbian_log.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogy(d.epoch, d.hebb_norm, label="plain Hebb")
        ax[0].semilogy(d.epoch, d.oja_norm, label="Oja")
        ax[0].set_title("Weight norm |w|"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(d.epoch, d.hebb_cos_pc1, label="plain Hebb")
        ax[1].plot(d.epoch, d.oja_cos_pc1, label="Oja")
        ax[1].set_ylim(0, 1.05); ax[1].set_title("|cos(w, first principal component)|")
        ax[1].set_xlabel("epoch"); ax[1].legend()
        save(fig, r / "hebbian.png")
    if (r / "hebbian_2d.csv").exists():
        d = pd.read_csv(r / "hebbian_2d.csv"); w = pd.read_csv(r / "hebbian_2d_w.csv").set_index("name")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(d.x, d.y, s=6, alpha=.4, color="grey")
        for name, c in [("pc1", "black"), ("oja", "tab:green")]:
            v = w.loc[name].to_numpy(float); v = v / np.linalg.norm(v) * 3
            ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2.5 if name == "oja" else 1.5))
            ax.text(v[0] * 1.08, v[1] * 1.08, "Oja weight" if name == "oja" else "PC1 (SVD)", color=c)
        ax.set_aspect("equal"); ax.set_title("Oja's rule finds the main axis of the data")
        save(fig, r / "hebbian_2d.png")
    if (r / "oja_vs_pc1.csv").exists():
        d = pd.read_csv(r / "oja_vs_pc1.csv")
        grid(d.iloc[:, 1:].to_numpy(float), "Oja weight vector vs first SVD component (crops)",
             r / "oja_vs_pc1.png", cmap="RdBu", labels=["Oja", "PC1 (SVD)"])
    if (r / "competitive_prototypes.csv").exists():
        d = pd.read_csv(r / "competitive_prototypes.csv")
        labs = [s_.split("_won")[-1] + " wins" for s_ in d.iloc[:, 0]]
        grid(d.iloc[:, 1:].to_numpy(float), "Competitive learning: learned prototypes",
             r / "competitive_prototypes.png", labels=labs)
        q = pd.read_csv(r / "competitive_log.csv")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(q.epoch, q.quantisation_error, marker="o")
        ax.set_title("Competitive learning: quantisation error"); ax.set_xlabel("epoch")
        save(fig, r / "competitive_error.png")
    if (r / "rbm_log.csv").exists():
        d = pd.read_csv(r / "rbm_log.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(d.epoch, d.recon_error, marker="o"); ax[0].set_title("RBM reconstruction error (CD-1)")
        ax[0].set_xlabel("epoch")
        ax[1].plot(d.epoch, d.free_energy_train, label="train"); ax[1].plot(d.epoch, d.free_energy_test, label="test")
        ax[1].set_title("Mean free energy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        save(fig, r / "rbm_training.png")
        grid(pd.read_csv(r / "rbm_filters.csv").iloc[:, 1:].to_numpy(float),
             "RBM hidden-unit filters", r / "rbm_filters.png", cmap="RdBu")
        rec = pd.read_csv(r / "rbm_reconstructions.csv").iloc[:, 1:].to_numpy(float)
        grid(rec, "Top row: test crops (binarised). Bottom row: RBM reconstructions",
             r / "rbm_reconstructions.png", cols=len(rec) // 2)
        grid(pd.read_csv(r / "rbm_samples.csv").iloc[:, 1:].to_numpy(float),
             "Samples drawn from the RBM (500 Gibbs steps)", r / "rbm_samples.png")

    cm_path = r / f"{a.confusion_run}_confusion.csv"
    if cm_path.exists():
        C = np.loadtxt(cm_path, delimiter=",")
        names = Path(a.classes).read_text(encoding="utf-8").split("\n") if Path(a.classes).exists() else []
        names = [n for n in names if n]
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        if len(names) == len(C):
            ax.set_xticks(range(len(C))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(C))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix (row-normalised): {a.confusion_run}")
        fig.colorbar(im)
        save(fig, r / f"confusion_{a.confusion_run}.png")


if __name__ == "__main__":
    main()
