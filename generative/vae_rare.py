"""Conditional VAE for rare e-waste classes + latent-space visualisation.

Data: the 32x32 grayscale crops in cpp_core/crops (10 classes, label + 1024 pixels in [0,1]).

Experiment (all numbers on the untouched test split):
  1. Make one well-populated class artificially rare (Computer-Mouse: 600 -> 60 train crops),
     so the rare class still has 147 test crops and its score is reliable.
     Christmas-Lights is the naturally rare class (94 train, 4 test) and is reported too.
  2. Train a conditional VAE on this imbalanced training set (it never sees the removed crops).
  3. Train the same small CNN classifier four ways:
       baseline       imbalanced data as is
       oversample     rare-class crops repeated (with flips) up to 600
       vae            rare classes topped up to 600 with VAE samples
       vae+oversample half repeats, half VAE samples
     over 3 seeds, and compare rare-class recall/F1 and macro F1.
  4. Plot the latent space (PCA of the posterior means), samples per class, reconstructions.

Usage:
  python vae_rare.py --crops ../cpp_core/crops --out ../cpp_core/results/generative
Runs in ~10-20 min on CPU, ~2 min on a GPU.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT = 16


def load(crops):
    names = [l.strip() for l in open(crops / "classes.txt") if l.strip()]
    out = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(crops / f"{s}.csv").values.astype(np.float32)
        out[s] = (d[:, 1:].reshape(-1, 1, 32, 32), d[:, 0].astype(np.int64))
    return names, out


class CVAE(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.nc = n_classes
        self.enc = nn.Sequential(
            nn.Conv2d(1 + n_classes, 32, 4, 2, 1), nn.LeakyReLU(0.2),   # 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),  # 8
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 4
            nn.Flatten())
        self.mu = nn.Linear(128 * 16, LATENT)
        self.logvar = nn.Linear(128 * 16, LATENT)
        self.fc = nn.Linear(LATENT + n_classes, 128 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def onehot(self, y):
        return F.one_hot(y, self.nc).float()

    def encode(self, x, y):
        maps = self.onehot(y)[:, :, None, None].expand(-1, -1, 32, 32)
        h = self.enc(torch.cat([x, maps], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y):
        h = self.fc(torch.cat([z, self.onehot(y)], 1)).view(-1, 128, 4, 4)
        return self.dec(h)  # logits

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z, y), mu, logvar


def vae_loss(logits, x, mu, logvar, beta):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / x.shape[0]
    return rec + beta * kl, rec, kl


def train_vae(x, y, nc, epochs, dev, log):
    torch.manual_seed(0)
    m = CVAE(nc).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    X, Y = torch.tensor(x), torch.tensor(y)
    # balance batches so the rare classes are seen often enough
    counts = np.bincount(y, minlength=nc)
    w = torch.tensor(1.0 / counts[y], dtype=torch.double)
    for ep in range(epochs):
        m.train()
        idx = torch.multinomial(w, len(y), replacement=True)
        beta = min(1.0, (ep + 1) / (epochs * 0.3))  # KL warm-up
        tot = rec_t = kl_t = 0.0
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            xb, yb = X[b].to(dev), Y[b].to(dev)
            logits, mu, lv = m(xb, yb)
            loss, rec, kl = vae_loss(logits, xb, mu, lv, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); rec_t += rec.item() * len(b); kl_t += kl.item() * len(b)
        n = len(idx)
        log.append({"epoch": ep + 1, "loss": tot / n, "recon": rec_t / n, "kl": kl_t / n, "beta": beta})
        print(f"  vae epoch {ep+1:>2}/{epochs}  loss {tot/n:7.2f}  recon {rec_t/n:7.2f}  kl {kl_t/n:6.2f}")
    return m


@torch.no_grad()
def sample(m, cls, n, dev, temp=1.0):
    m.eval()
    z = torch.randn(n, LATENT, device=dev) * temp
    y = torch.full((n,), cls, dtype=torch.long, device=dev)
    return torch.sigmoid(m.decode(z, y)).cpu().numpy()


class SmallCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, nc))

    def forward(self, x):
        return self.f(x)


def train_cls(x, y, nc, seed, dev, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SmallCNN(nc).to(dev)
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * ((len(y) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)  # smooth end -> stable final model
    X, Y = torch.tensor(x), torch.tensor(y)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Y))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = X[b].to(dev)
            flip = torch.rand(len(b), device=dev) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            loss = F.cross_entropy(m(xb), Y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


@torch.no_grad()
def predict(m, x, dev):
    m.eval()
    return torch.cat([m(torch.tensor(x[i:i + 512]).to(dev)).argmax(1).cpu()
                      for i in range(0, len(x), 512)]).numpy()


def f1_per_class(y, p, nc):
    out = []
    for c in range(nc):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    return out


def topup(x, y, nc, target, how, vae, dev, rng):
    xs, ys = [x], [y]
    counts = np.bincount(y, minlength=nc)
    for c in range(nc):
        need = target - counts[c]
        if need <= 0:
            continue
        n_rep = need if how == "oversample" else (need // 2 if how == "vae+oversample" else 0)
        n_gen = need - n_rep
        if n_rep:
            src = x[y == c][rng.integers(0, counts[c], n_rep)]
            flip = rng.random(n_rep) < 0.5
            src = np.where(flip[:, None, None, None], src[..., ::-1], src)
            xs.append(src.astype(np.float32)); ys.append(np.full(n_rep, c))
        if n_gen:
            xs.append(sample(vae, c, n_gen, dev).astype(np.float32)); ys.append(np.full(n_gen, c))
    return np.concatenate(xs), np.concatenate(ys)


def grid(imgs, rows, cols, path, titles=None, title=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2 + (0.4 if title else 0)))
    for i, ax in enumerate(np.array(axes).ravel()):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i].reshape(32, 32), cmap="gray", vmin=0, vmax=1)
        if titles is not None and i % cols == 0 and i // cols < len(titles):
            ax.set_title(titles[i // cols], fontsize=6, loc="left")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="../cpp_core/crops")
    ap.add_argument("--out", default="../cpp_core/results/generative")
    ap.add_argument("--rare", default="Computer-Mouse")
    ap.add_argument("--rare-n", type=int, default=60)
    ap.add_argument("--vae-epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--load-vae", action="store_true", help="reuse cvae.pt from --out instead of retraining")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names, d = load(Path(a.crops))
    nc = len(names)
    rare = names.index(a.rare)
    natural_rare = int(np.argmin(np.bincount(d["train"][1], minlength=nc)))

    # 1. imbalanced training set
    rng = np.random.default_rng(0)
    xtr, ytr = d["train"]
    keep = np.ones(len(ytr), bool)
    rare_idx = np.where(ytr == rare)[0]
    keep[rng.choice(rare_idx, len(rare_idx) - a.rare_n, replace=False)] = False
    xtr, ytr = xtr[keep], ytr[keep]
    counts = np.bincount(ytr, minlength=nc)
    print("train counts:", dict(zip(names, counts.tolist())))

    # 2. conditional VAE
    if a.load_vae:
        vae = CVAE(nc).to(dev); vae.load_state_dict(torch.load(out / "cvae.pt", map_location=dev))
        log = pd.read_csv(out / "vae_log.csv").to_dict("records")
    else:
        log = []
        vae = train_vae(xtr, ytr, nc, a.vae_epochs, dev, log)
        pd.DataFrame(log).to_csv(out / "vae_log.csv", index=False)
        torch.save(vae.state_dict(), out / "cvae.pt")

    # 3. classifier comparison
    xte, yte = d["test"]
    rows = []
    for how in ("baseline", "oversample", "vae", "vae+oversample"):
        for seed in range(a.seeds):
            if how == "baseline":
                xs, ys = xtr, ytr
            else:
                xs, ys = topup(xtr, ytr, nc, int(counts.max()), how, vae, dev, np.random.default_rng(seed))
            m = train_cls(xs, ys, nc, seed, dev)
            p = predict(m, xte, dev)
            f = f1_per_class(yte, p, nc)
            rows.append({"method": how, "seed": seed, "accuracy": float(np.mean(p == yte)),
                         "macro_f1": float(np.mean([v[2] for v in f])),
                         "rare_recall": f[rare][1], "rare_f1": f[rare][2],
                         "xmas_recall": f[natural_rare][1], "xmas_f1": f[natural_rare][2]})
            print(f"  {how:<15} seed {seed}: acc {rows[-1]['accuracy']:.3f}  macroF1 {rows[-1]['macro_f1']:.3f}  "
                  f"{a.rare} F1 {f[rare][2]:.3f}  {names[natural_rare]} F1 {f[natural_rare][2]:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(out / "rare_class_runs.csv", index=False)
    summary = res.groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed")
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.round(4).to_csv(out / "rare_class_summary.csv")
    print("\n", summary.round(3).to_string())

    # 4. plots
    lg = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(lg.epoch, lg.recon); ax[0].set_title("Reconstruction loss (BCE per image)")
    ax[1].plot(lg.epoch, lg.kl, c="C1"); ax[1].set_title("KL divergence")
    for x_ in ax: x_.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out / "vae_training.png", dpi=150); plt.close(fig)

    # latent space: posterior means of val crops, projected to 2-D with PCA
    xv, yv = d["val"]
    vae.eval()
    with torch.no_grad():
        mu = torch.cat([vae.encode(torch.tensor(xv[i:i + 512]).to(dev), torch.tensor(yv[i:i + 512]).to(dev))[0].cpu()
                        for i in range(0, len(yv), 512)]).numpy()
        # unconditional view: encode every crop with the SAME label so the label channel cannot separate classes
        mu0 = torch.cat([vae.encode(torch.tensor(xv[i:i + 512]).to(dev),
                                    torch.zeros(len(xv[i:i + 512]), dtype=torch.long, device=dev))[0].cpu()
                         for i in range(0, len(yv), 512)]).numpy()
    gen = np.concatenate([sample(vae, c, 40, dev) for c in range(nc)])
    with torch.no_grad():
        mug = vae.encode(torch.tensor(gen).to(dev), torch.zeros(len(gen), dtype=torch.long, device=dev))[0].cpu().numpy()
    for tag, M, extra in (("latent_space", mu0, mug), ("latent_space_conditional", mu, None)):
        c0 = M.mean(0); _, _, vt = np.linalg.svd(M - c0, full_matrices=False)
        P = (M - c0) @ vt[:2].T
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for c in range(nc):
            s = yv == c
            ax.scatter(P[s, 0], P[s, 1], s=6, alpha=0.6, label=f"{names[c]} ({s.sum()})")
        if extra is not None:
            G = (extra - c0) @ vt[:2].T
            gl = np.repeat(np.arange(nc), 40)
            for c in (rare, natural_rare):
                ax.scatter(G[gl == c, 0], G[gl == c, 1], s=28, marker="x", c="k" if c == rare else "r",
                           label=f"VAE samples: {names[c]}")
        ax.legend(fontsize=6, markerscale=2, loc="best"); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title("VAE latent space (posterior means, PCA)" if extra is not None
                     else "Latent means given the true label (class info lives in the label)")
        fig.tight_layout(); fig.savefig(out / f"{tag}.png", dpi=150); plt.close(fig)

    grid(gen.reshape(nc, 40, 32, 32)[:, :10].reshape(-1, 1, 32, 32), nc, 10, out / "vae_samples.png",
         titles=names, title="Conditional VAE samples (one row per class)")
    sel = np.concatenate([np.where(yv == c)[0][:2] for c in range(nc)])
    with torch.no_grad():
        logits, _, _ = vae(torch.tensor(xv[sel]).to(dev), torch.tensor(yv[sel]).to(dev))
    rec = torch.sigmoid(logits).cpu().numpy()
    pair = np.stack([xv[sel], rec], 1).reshape(-1, 1, 32, 32)
    grid(pair, 4, len(sel) // 2, out / "vae_reconstructions.png", title="Validation crops (odd) and VAE reconstructions (even)")
    real_rare = xtr[ytr == rare][:10]
    grid(np.concatenate([real_rare, sample(vae, rare, 10, dev)]), 2, 10, out / "vae_rare_real_vs_generated.png",
         titles=[f"real {a.rare}", "generated"], title=f"{a.rare}: real training crops vs VAE samples")

    m = summary
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = list(m.index); x_ = np.arange(len(labels))
    for i, (col, lab) in enumerate((("macro_f1", "macro F1"), ("rare_f1", f"{a.rare} F1 (made rare)"),
                                    ("xmas_f1", f"{names[natural_rare]} F1 (naturally rare)"))):
        ax.bar(x_ + (i - 1) * 0.27, m[f"{col}_mean"], 0.27, yerr=m[f"{col}_std"], capsize=3, label=lab)
    ax.set_xticks(x_); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_title(f"Rare-class augmentation, test set, mean ± sd over {a.seeds} seeds")
    fig.tight_layout(); fig.savefig(out / "rare_class_comparison.png", dpi=150); plt.close(fig)

    json.dump({"rare_class": a.rare, "rare_train_n": a.rare_n, "natural_rare": names[natural_rare],
               "natural_rare_train_n": int(counts[natural_rare]), "device": dev,
               "vae_final": log[-1]}, open(out / "vae_summary.json", "w"), indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
