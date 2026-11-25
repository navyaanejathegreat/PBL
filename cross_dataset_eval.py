"""Cross-dataset test: run the trained e-waste detector on an independent dataset.

Dataset: "E-Waste" by azizul-abedin-azmi (Roboflow Universe, CC BY 4.0; the Bangladeshi e-waste
dataset, Mendeley DOI 10.17632/77383kmdnw.1). 12 classes; 6 of them overlap with ours.
Its labels are polygons; they are turned into boxes here.

Only the 6 shared groups are scored:
    Battery    <- our Battery
    Keyboard   <- our Computer-Keyboard
    Light bulb <- our LED-Bulb, Compact-Fluorescent-Lamps
    Mobile     <- our Smartphone, Bar-Phone
    Mouse      <- our Computer-Mouse
    PCB        <- our PCB
Their non-e-waste classes (glass, paper, plastic, ...) are ignored, and so are our predictions of
classes outside the 6 groups (the other dataset has no labels for them).

Writes, for the C++ app:  classes.csv, gt.csv, preds.csv, timing.csv   (then run qc_app --dir <out>)
Also writes confusion.csv: for every real object, what our model called it (any of the 66 classes).

Usage (project folder):
  python cross_dataset_eval.py --weights D:\\ewaste_runs\\final_v2\\weights\\best.pt
"""
from pathlib import Path
import argparse
import collections
import csv
import time

import yaml

GROUPS = {   # group name -> (their class names, our class names, recycling stream)
    "Battery": (["Battery_Waste"], ["Battery"], "battery"),
    "Keyboard": (["Keyboard"], ["Computer-Keyboard"], "it"),
    "Light-Bulb": (["Light_Bulb"], ["LED-Bulb", "Compact-Fluorescent-Lamps"], "lamp"),
    "Mobile": (["Mobile"], ["Smartphone", "Bar-Phone"], "it"),
    "Mouse": (["Mouse"], ["Computer-Mouse"], "it"),
    "PCB": (["PCB"], ["PCB"], "it"),
}
GROUP_NAMES = list(GROUPS)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_box(parts):
    """YOLO label row -> (cx, cy, w, h); accepts boxes and polygons."""
    v = [float(x) for x in parts]
    if len(v) == 4:
        return tuple(v)
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\E-Waste Dataset (2)")
    ap.add_argument("--weights", default=r"D:\ewaste_runs\final_v2\weights\best.pt")
    ap.add_argument("--splits", nargs="+", default=["test", "valid"],
                    help="its train split holds 3 augmented copies of each photo, so it is left out")
    ap.add_argument("--out", default=r"cpp_core\qc_input\cross_bangladesh")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    their = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]
    their_to_group = {their.index(t): gi for gi, g in enumerate(GROUP_NAMES) for t in GROUPS[g][0] if t in their}

    # ground truth (only images that contain at least one shared class)
    images, gt = [], []
    for split in a.splits:
        for img in sorted((src / split / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = src / split / "labels" / (img.stem + ".txt")
            boxes = []
            if lab.exists():
                for line in lab.read_text().split("\n"):
                    p = line.split()
                    if len(p) >= 5 and int(p[0]) in their_to_group:
                        boxes.append((their_to_group[int(p[0])],) + to_box(p[1:]))
            if boxes:
                for b in boxes:
                    gt.append((len(images),) + b)
                images.append(img)
    print(f"{len(images)} images with {len(gt)} objects of the 6 shared classes "
          f"({', '.join(f'{g} {sum(1 for x in gt if x[1] == i)}' for i, g in enumerate(GROUP_NAMES))})")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    ours = model.names
    our_to_group = {i: gi for i, n in ours.items() for gi, g in enumerate(GROUP_NAMES) if n in GROUPS[g][1]}
    missing = [n for g in GROUP_NAMES for n in GROUPS[g][1] if n not in ours.values()]
    if missing:
        print("warning: these class names are not in the model:", missing)

    model.predict(str(images[0]), imgsz=a.imgsz, device=a.device, verbose=False)  # warm-up
    preds, raw, inf_ms = [], [], 0.0
    t0 = time.perf_counter()
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        for j, r in enumerate(model.predict([str(p) for p in batch], imgsz=a.imgsz, device=a.device,
                                            conf=0.001, max_det=300, verbose=False)):
            inf_ms += r.speed["inference"]
            for c, s, xywh in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                raw.append((start + j, int(c), s, tuple(xywh)))
                if int(c) in our_to_group:
                    preds.append((start + j, our_to_group[int(c)], s) + tuple(xywh))
    total_ms = (time.perf_counter() - t0) * 1000

    with open(out / "classes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "name", "stream"])
        for i, g in enumerate(GROUP_NAMES):
            w.writerow([i, g, GROUPS[g][2]])
    with open(out / "gt.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "cls", "cx", "cy", "w", "h"])
        for row in gt:
            w.writerow([row[0], row[1]] + [f"{v:.6f}" for v in row[2:]])
    with open(out / "preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "cls", "conf", "cx", "cy", "w", "h"])
        for row in preds:
            w.writerow([row[0], row[1], f"{row[2]:.5f}"] + [f"{v:.6f}" for v in row[3:]])
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["images", "total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"])
        w.writerow([len(images), f"{total_ms:.1f}", 0, f"{inf_ms:.1f}", 0])

    # what did the model call each real object? (best-overlapping prediction with conf >= 0.25)
    by_img = collections.defaultdict(list)
    for r in raw:
        if r[2] >= 0.25:
            by_img[r[0]].append(r)
    conf_tab = collections.Counter()
    for g in gt:
        cands = [(iou(g[2:], r[3]), r) for r in by_img[g[0]]]
        cands = [c for c in cands if c[0] >= 0.5]
        said = ours[max(cands, key=lambda c: c[0])[1][1]] if cands else "(nothing found)"
        conf_tab[(GROUP_NAMES[g[1]], said)] += 1
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["real_class", "model_said", "count"])
        for (real, said), n in sorted(conf_tab.items(), key=lambda kv: (kv[0][0], -kv[1])):
            w.writerow([real, said, n])
    print("\nWhat the model called each real object (top 3 per class):")
    for g in GROUP_NAMES:
        rows = sorted([(n, said) for (real, said), n in conf_tab.items() if real == g], reverse=True)
        tot = sum(n for n, _ in rows)
        print(f"  {g:<11} ({tot:>3}): " + ", ".join(f"{said} {n / tot:.0%}" for n, said in rows[:3]))
    print(f"\nSaved to {out}. Now run:\n  cpp_core\\bin\\qc_app --dir {out} --out cpp_core\\results\\qc_cross_bangladesh")


if __name__ == "__main__":
    main()
