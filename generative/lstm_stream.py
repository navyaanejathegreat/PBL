"""LSTM stream monitor for a recycling conveyor.

Problem
  A bin is meant to collect one WEEE stream (e.g. "small IT & telecom"). Items pass the camera one
  by one; the detector labels each item. Occasionally a mis-sorted load arrives and, for a stretch of
  items, many items belong to other streams (a contamination episode). The monitor must raise an alarm
  during those episodes, but NOT for the isolated stray item or for detector mistakes, because each
  alarm stops the line.

Data (simulated from your real detector)
  * item classes are drawn with the class frequencies of the v2 test set;
  * every item is passed through a detector model built from the per-class precision/recall of the
    final YOLO26n on the test set (eval output text file): a correct label with probability = recall,
    otherwise missed or confused with another class, with a lower confidence;
  * episodes: 0-3 per 120-item sequence, 6-20 items long, 40-70 % foreign items inside;
  * background: 3 % stray foreign items everywhere.

Models (all tuned on a validation set, scored on a separate test set)
  rule-1     alarm when the current detected item is off-stream
  window-k   alarm when >= t of the last k detected items are off-stream (k, t tuned)
  LSTM       1-layer LSTM (hidden 32) over per-item features, per-item sigmoid output

Metrics: per-item F1 of the alarm, episode recall (episode flagged at least once), mean delay to the
first alarm in an episode, false alarms per 1,000 clean items.

Usage:
  python lstm_stream.py --eval-log test_eval_v2.txt --out ../cpp_core/results/generative
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STREAM_OF = {
    "temp": "Air-Conditioner Refrigerator Freezer Cooling-Display Water-Dispenser",
    "screen": "CRT-Monitor CRT-TV Flat-Panel-Monitor Flat-Panel-TV Laptop Patient-Monitoring-System",
    "lamp": "Compact-Fluorescent-Lamps Straight-Tube-Fluorescent-Lamp LED-Bulb",
    "large": "Washing-Machine Dishwasher Oven Stove Range-Hood Boiler Photovoltaic-Panel Server Rotary-Mower "
             "Street-Lamp Ceiling-Fan Electric-Bicycle",
    "it": "Smartphone Bar-Phone Telephone-Set Router Network-Switch Computer-Keyboard Computer-Mouse Desktop-PC "
          "Printer Calculator HDD SSD USB-Flash-Drive PCB Power-Adapter TV-Remote-Control Headphone",
    "battery": "Battery",
}
STREAMS = ["temp", "screen", "lamp", "large", "small", "it", "battery"]
SEQ = 120


def stream_of(c):
    for s, names in STREAM_OF.items():
        if c in names.split():
            return s
    return "small"


def parse_eval(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"\s*([A-Za-z][\w-]*)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m and m.group(1) != "all":
            rows.append({"cls": m.group(1), "inst": int(m.group(3)), "P": float(m.group(4)), "R": float(m.group(5))})
    df = pd.DataFrame(rows)
    df["stream"] = df.cls.map(stream_of)
    return df


class World:
    def __init__(self, df, rng):
        self.df, self.rng = df, rng
        self.p = (df.inst / df.inst.sum()).values
        self.sidx = df.stream.map(STREAMS.index).values
        self.by_stream = {s: np.where(self.sidx == s)[0] for s in range(len(STREAMS))}

    def draw(self, stream, n):
        pool = self.by_stream[stream]
        w = self.p[pool] / self.p[pool].sum()
        return self.rng.choice(pool, n, p=w)

    def sequence(self):
        r = self.rng
        target = r.integers(len(STREAMS) - 1)  # batteries are never a target bin
        true = self.draw(target, SEQ)
        episode = np.zeros(SEQ, bool)
        for _ in range(r.integers(0, 4)):
            L = r.integers(6, 21); s = r.integers(0, SEQ - L)
            episode[s:s + L] = True
        others = [s for s in range(len(STREAMS)) if s != target]
        foreign_rate = np.where(episode, r.uniform(0.4, 0.7), 0.03)
        foreign = r.random(SEQ) < foreign_rate
        for i in np.where(foreign)[0]:
            true[i] = self.draw(r.choice(others), 1)[0]
        # detector
        R = self.df.R.values[true]
        ok = r.random(SEQ) < R
        miss = ~ok & (r.random(SEQ) < 0.5)
        det = true.copy()
        wrong = ~ok & ~miss
        det[wrong] = r.choice(len(self.p), wrong.sum(), p=self.p)
        conf = np.where(ok, r.beta(8, 2, SEQ), r.beta(3, 3, SEQ))
        det_stream = np.where(miss, -1, self.sidx[det])
        return target, det_stream, conf, episode, self.sidx[true] != target


def features(target, det_stream, conf):
    n = len(det_stream)
    x = np.zeros((n, len(STREAMS) + 4), np.float32)
    seen = det_stream >= 0
    x[np.where(seen)[0], det_stream[seen]] = 1
    x[:, len(STREAMS)] = ~seen                                   # nothing detected
    x[:, len(STREAMS) + 1] = seen & (det_stream != target)       # off-stream
    x[:, len(STREAMS) + 2] = np.where(seen, conf, 0)
    x[:, len(STREAMS) + 3] = seen & (det_stream == STREAMS.index("battery"))  # hazard
    return x


def make_set(world, n):
    X, Y, OFF = [], [], []
    for _ in range(n):
        t, ds, cf, ep, _ = world.sequence()
        X.append(features(t, ds, cf)); Y.append(ep); OFF.append((ds >= 0) & (ds != t))
    return np.stack(X), np.stack(Y).astype(np.float32), np.stack(OFF)


class Monitor(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0]).squeeze(-1)


def window_rule(off, k, t):
    c = np.cumsum(np.pad(off.astype(int), ((0, 0), (1, 0))), 1)
    lo = np.maximum(np.arange(1, off.shape[1] + 1) - k, 0)
    return (c[:, 1:] - c[np.arange(off.shape[0])[:, None], lo]) >= t


def metrics(pred, y):
    pred, y = pred.astype(bool), y.astype(bool)
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    caught, delays, n_ep = 0, [], 0
    for p, t in zip(pred, y):
        i = 0
        while i < len(t):
            if t[i]:
                j = i
                while j < len(t) and t[j]:
                    j += 1
                n_ep += 1
                hit = np.where(p[i:j])[0]
                if len(hit):
                    caught += 1; delays.append(hit[0])
                i = j
            else:
                i += 1
    # false alarms: alarm onsets outside episodes
    onsets = pred & ~np.pad(pred, ((0, 0), (1, 0)))[:, :-1]
    fa = (onsets & ~y).sum() / max((~y).sum(), 1) * 1000
    return {"precision": prec, "recall": rec, "f1": f1, "episode_recall": caught / max(n_ep, 1),
            "mean_delay_items": float(np.mean(delays)) if delays else float("nan"),
            "false_alarms_per_1000": fa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-log", required=True, help="text output of eval_test.py for the final model")
    ap.add_argument("--out", default="../cpp_core/results/generative")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = parse_eval(a.eval_log)
    print(f"detector model from {len(df)} classes; mean recall {df.R.mean():.3f}")
    Xtr, Ytr, _ = make_set(World(df, np.random.default_rng(1)), 3000)
    Xva, Yva, OFFva = make_set(World(df, np.random.default_rng(2)), 600)
    Xte, Yte, OFFte = make_set(World(df, np.random.default_rng(3)), 1000)

    results = {}
    results["rule-1 (current item off-stream)"] = metrics(OFFte, Yte)
    best = max(((k, t) for k in range(3, 13) for t in range(2, k + 1)),
               key=lambda kt: metrics(window_rule(OFFva, *kt), Yva)["f1"])
    results[f"window rule (>= {best[1]} of last {best[0]} off-stream)"] = metrics(window_rule(OFFte, *best), Yte)

    torch.manual_seed(0)
    m = Monitor(Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    pos_w = torch.tensor((1 - Ytr.mean()) / Ytr.mean())
    log = []
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[b]), Yt[b], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(torch.tensor(Xva))).numpy()
        f1v = metrics(pv > 0.5, Yva)["f1"]
        log.append({"epoch": ep + 1, "train_loss": tot / len(Xt), "val_f1": f1v})
        print(f"  lstm epoch {ep+1:>2}/{a.epochs}  loss {tot/len(Xt):.4f}  val F1 {f1v:.3f}")
    thr = max(np.arange(0.2, 0.95, 0.05), key=lambda t: metrics(pv > t, Yva)["f1"])
    with torch.no_grad():
        pt = torch.sigmoid(m(torch.tensor(Xte))).numpy()
    results[f"LSTM (threshold {thr:.2f})"] = metrics(pt > thr, Yte)
    torch.save(m.state_dict(), out / "lstm_monitor.pt")

    table = pd.DataFrame(results).T.round(3)
    table.index.name = "monitor"
    table.to_csv(out / "lstm_vs_rules.csv")
    pd.DataFrame(log).to_csv(out / "lstm_log.csv", index=False)
    print("\n", table.to_string())

    # one example sequence
    i = int(np.argmax(Yte.sum(1) > 20))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = np.arange(SEQ)
    ax.fill_between(t, 0, Yte[i], step="mid", color="#F0675C", alpha=0.25, label="true contamination episode")
    ax.scatter(t[OFFte[i]], np.full(OFFte[i].sum(), 1.05), marker="|", c="k", label="detected off-stream item")
    ax.plot(t, pt[i], c="#2E9E6E", label="LSTM alarm probability")
    ax.plot(t, window_rule(OFFte[i:i + 1], *best)[0] * 0.95, c="#6A7FDB", ls="--", label="window rule alarm")
    ax.axhline(thr, c="#2E9E6E", ls=":", lw=1)
    ax.set_ylim(-0.05, 1.15); ax.set_xlabel("item on the belt"); ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Stream monitor on one test sequence")
    fig.tight_layout(); fig.savefig(out / "lstm_example.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = ["f1", "episode_recall"]
    xs = np.arange(len(table))
    for j, c in enumerate(cols):
        ax.bar(xs + (j - 0.5) * 0.38, table[c], 0.38, label=c.replace("_", " "))
    for k, v in enumerate(table["false_alarms_per_1000"]):
        ax.text(k, 1.02, f"{v:.0f} false\nalarms/1k", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([s.split(" (")[0] for s in table.index]); ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Contamination monitors on 1,000 test sequences")
    fig.tight_layout(); fig.savefig(out / "lstm_vs_rules.png", dpi=150); plt.close(fig)
    json.dump({"window_rule": {"k": best[0], "t": best[1]}, "lstm_threshold": float(thr),
               "train_sequences": 3000, "test_sequences": 1000, "seq_len": SEQ}, open(out / "lstm_summary.json", "w"), indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
