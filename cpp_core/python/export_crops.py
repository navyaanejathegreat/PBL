"""
Cut labelled objects out of the YOLO dataset into small grayscale crops for the
from-scratch C++ MLP and the tensor/SVD sandbox.

Keeps the dataset's own train/valid/test split, so the crops are leakage-safe as
long as the source dataset is (use D:\\ewaste_v2 once it is built).

    python export_crops.py --src D:\\ewaste_v2 --out ..\\crops --classes 10 --per-class 600

Writes
    crops/train.csv, crops/val.csv, crops/test.csv   label,p0..p1023  (values 0..1)
    crops/classes.txt                                 one class name per line
    crops/export_report.csv                           crops per class and split
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\ewaste_v2")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--classes", type=int, default=10,
                    help="use the N classes with the most training boxes")
    ap.add_argument("--names", nargs="*", default=None,
                    help="or list class names explicitly (overrides --classes)")
    ap.add_argument("--per-class", type=int, default=600, help="max train crops per class")
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--min-px", type=int, default=24, help="skip boxes smaller than this")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    cfg = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in names.items()}

    split_dirs = {"train": "train", "val": "valid", "test": "test"}
    boxes = defaultdict(lambda: defaultdict(list))      # split -> class -> [(img, box)]
    for split, folder in split_dirs.items():
        img_dir, lab_dir = src / folder / "images", src / folder / "labels"
        for img in img_dir.iterdir():
            if img.suffix.lower() not in EXTS:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            for line in lab.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    boxes[split][int(p[0])].append((img, tuple(map(float, p[1:]))))

    if a.names:
        by_name = {v: k for k, v in names.items()}
        chosen = [by_name[n] for n in a.names]
    else:
        chosen = sorted(boxes["train"], key=lambda c: -len(boxes["train"][c]))[: a.classes]
        chosen = [c for c in chosen if boxes["val"][c] and boxes["test"][c]]
    new_id = {c: i for i, c in enumerate(chosen)}
    print("classes:", ", ".join(names[c] for c in chosen))

    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(names[c] for c in chosen) + "\n", encoding="utf-8")
    rng = random.Random(a.seed)
    caps = {"train": a.per_class, "val": max(1, a.per_class // 4), "test": max(1, a.per_class // 4)}
    report = Counter()
    header = ["label"] + [f"p{i}" for i in range(a.side * a.side)]

    for split in split_dirs:
        rows = []
        for c in chosen:
            items = boxes[split][c][:]
            rng.shuffle(items)
            kept = 0
            for img_path, (cx, cy, w, h) in items:
                if kept >= caps[split]:
                    break
                with Image.open(img_path) as im:
                    W, H = im.size
                    x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                    x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
                    if (x1 - x0) < a.min_px or (y1 - y0) < a.min_px:
                        continue
                    # square crop around the box so the object is not stretched
                    s = max(x1 - x0, y1 - y0)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    crop = im.convert("L").crop((int(mx - s / 2), int(my - s / 2),
                                                 int(mx + s / 2), int(my + s / 2)))
                    crop = crop.resize((a.side, a.side), Image.BILINEAR)
                px = [round(v / 255.0, 4) for v in crop.tobytes()]
                rows.append([new_id[c]] + px)
                kept += 1
            report[(split, names[c])] = kept
        rng.shuffle(rows)
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"{split}: {len(rows)} crops")

    with open(out / "export_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "val", "test"])
        for c in chosen:
            n = names[c]
            w.writerow([n, report[("train", n)], report[("val", n)], report[("test", n)]])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
