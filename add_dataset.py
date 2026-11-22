"""Add another labelled dataset to the TRAINING split of an existing dataset.

  * The base dataset (e.g. ewaste_v2) is copied as it is (hard links: instant, no extra space),
    so its validation and test images stay identical and every model stays comparable.
  * Only the extra dataset's training images are added, with labels renamed to our classes
    by a mapping; boxes of classes we do not have are dropped; polygons become boxes.
  * Any added photo that looks like (dHash within 6 bits, flips included) a protected image is
    skipped: the base's valid/test, and the extra dataset's own valid/test (used for the
    cross-dataset test), so nothing leaks into any test.
  * Chain it: --base D:\\ewaste_v4 --extra <next dataset> --dst D:\\ewaste_v5, and so on.

Usage:
  python add_dataset.py --base D:\\ewaste_v2 --extra "D:\\E-Waste Dataset (2)" --map bangladesh --dst D:\\ewaste_v4
  python add_dataset.py ... --map my_mapping.json      (JSON: {"their name": "our name", ...})
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import yaml

from group_and_split import hashes, hamming

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DIST = 6
PRESETS = {
    "bangladesh": {"Battery_Waste": "Battery", "Keyboard": "Computer-Keyboard", "Light_Bulb": "LED-Bulb",
                   "Mobile": "Smartphone", "Mouse": "Computer-Mouse", "PCB": "PCB"},
}


def load_names(folder):
    with open(Path(folder) / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def images_in(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.exists() else []


def to_box(vals):
    v = [float(x) for x in vals]
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = max(0.0, min(xs)), min(1.0, max(xs)), max(0.0, min(ys)), min(1.0, max(ys))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"D:\ewaste_v2")
    ap.add_argument("--extra", required=True)
    ap.add_argument("--map", required=True, help="a preset name (" + ", ".join(PRESETS) + ") or a JSON file")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--prefix", default=None, help="file-name prefix for added images (default: from --map)")
    a = ap.parse_args()
    base, extra, dst = Path(a.base), Path(a.extra), Path(a.dst)
    if dst.exists():
        sys.exit(f"{dst} already exists. Rename or remove it first.")
    mapping = PRESETS[a.map] if a.map in PRESETS else json.load(open(a.map, encoding="utf-8"))
    prefix = (a.prefix or Path(a.map).stem) + "_"

    ours, theirs = load_names(base), load_names(extra)
    bad = [o for o in mapping.values() if o not in ours]
    if bad:
        sys.exit(f"These target classes are not in {base}: {bad}")
    to_ours = {theirs.index(t): ours.index(o) for t, o in mapping.items() if t in theirs}
    print("mapping:", {theirs[k]: ours[v] for k, v in to_ours.items()})

    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)
            for p in (base / split / sub).iterdir():
                if p.is_file() and p.suffix != ".cache":
                    place(p, dst / split / sub / p.name)
        print(f"{split}: copied from {base.name}")

    protected = []
    for folder in [base / "valid" / "images", base / "test" / "images",
                   extra / "valid" / "images", extra / "test" / "images"]:
        for p in images_in(folder):
            protected.append(hashes(p)[0])
    print(f"hashed {len(protected)} protected validation/test images")

    added = dup = empty = 0
    boxes = {}
    for img in images_in(extra / "train" / "images"):
        lab = extra / "train" / "labels" / (img.stem + ".txt")
        lines = []
        if lab.exists():
            for line in lab.read_text().split("\n"):
                p = line.split()
                if len(p) >= 5 and int(p[0]) in to_ours:
                    cx, cy, w, h = to_box(p[1:])
                    if w > 0.002 and h > 0.002:
                        lines.append(f"{to_ours[int(p[0])]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            empty += 1
            continue
        h1, hm = hashes(img)
        if any(min(hamming(h1, e), hamming(hm, e)) <= MAX_DIST for e in protected):
            dup += 1
            continue
        place(img, dst / "train" / "images" / (prefix + img.name))
        (dst / "train" / "labels" / (prefix + img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        added += 1
        for l in lines:
            c = int(l.split()[0]); boxes[c] = boxes.get(c, 0) + 1

    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(dst), "train": "train/images", "val": "valid/images", "test": "test/images",
                        "names": dict(enumerate(ours)), "nc": len(ours)}, f, sort_keys=False, allow_unicode=True)
    print(f"\nAdded {added} training images (skipped {dup} look-alikes of validation/test images, "
          f"{empty} with none of the mapped classes)")
    for c in sorted(boxes, key=lambda c: -boxes[c]):
        print(f"  {ours[c]:<24} +{boxes[c]} boxes")
    n = {s: len(images_in(dst / s / "images")) for s in ("train", "valid", "test")}
    print(f"\n{dst}: train {n['train']}, valid {n['valid']}, test {n['test']}")
    print(f"Train with: data=r'{dst / 'data.yaml'}'")


if __name__ == "__main__":
    main()
