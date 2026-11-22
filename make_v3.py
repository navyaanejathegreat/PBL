"""Build ewaste_v3 = ewaste_v2 (unchanged) + Open Images photos added to TRAIN only.

Why train only: valid/test stay exactly as in v2, so the v2 and v3 models are
scored on the same test images and the comparison is fair.

Safety: any downloaded photo that looks like (dHash within 6 bits, flips
included) a v2 valid/test photo is skipped, so nothing leaks into evaluation.

Labels: Open Images labels use ewaste_final's 77 class ids; they are converted
to v2's 66 classes by name. Boxes of classes v2 dropped are removed.

Usage:
    python make_v3.py
    python make_v3.py --v2 D:\\ewaste_v2 --extra D:\\extra_data\\openimages --dst D:\\ewaste_v3
"""
from pathlib import Path
import argparse
import os
import shutil
import sys

import yaml

from group_and_split import hashes, hamming

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DIST = 6


def load_names(folder):
    with open(Path(folder) / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def place(src, dst):
    """Hard link (instant, no extra disk) and fall back to copying."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def images_in(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=r"D:\ewaste_v2")
    ap.add_argument("--extra", default=r"D:\extra_data\openimages")
    ap.add_argument("--dst", default=r"D:\ewaste_v3")
    a = ap.parse_args()
    v2, extra, dst = Path(a.v2), Path(a.extra), Path(a.dst)

    if dst.exists():
        sys.exit(f"{dst} already exists. Rename or remove it first.")
    v2_names, oi_names = load_names(v2), load_names(extra)
    to_v2 = {i: v2_names.index(n) for i, n in enumerate(oi_names) if n in v2_names}

    # 1. copy v2 as it is
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)
            for p in (v2 / split / sub).iterdir():
                if p.is_file() and p.suffix != ".cache":
                    place(p, dst / split / sub / p.name)
        print(f"{split}: copied from v2")

    # 2. hashes of v2 evaluation images
    eval_hashes = []
    for split in ("valid", "test"):
        for p in images_in(v2 / split / "images"):
            eval_hashes.append(hashes(p)[0])
    print(f"hashed {len(eval_hashes)} v2 valid/test images")

    # 3. add downloaded photos to train
    added = skipped_dup = skipped_empty = 0
    boxes_added = {}
    taken = {p.name for p in (dst / "train" / "images").iterdir()}
    for img in images_in(extra / "train" / "images"):
        lab = extra / "train" / "labels" / (img.stem + ".txt")
        lines = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) in to_v2:
                    c = to_v2[int(parts[0])]
                    lines.append(" ".join([str(c)] + parts[1:]))
                    boxes_added[c] = boxes_added.get(c, 0) + 1
        if not lines:
            skipped_empty += 1
            continue
        h, hm = hashes(img)
        if any(min(hamming(h, e), hamming(hm, e)) <= MAX_DIST for e in eval_hashes):
            skipped_dup += 1
            continue
        name = "oi_" + img.name
        if name in taken:
            continue
        place(img, dst / "train" / "images" / name)
        (dst / "train" / "labels" / ("oi_" + img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        added += 1

    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(dst), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "names": dict(enumerate(v2_names)),
                        "nc": len(v2_names)}, f, sort_keys=False, allow_unicode=True)

    print(f"\nAdded {added} Open Images photos to train "
          f"(skipped {skipped_dup} look-alikes of valid/test, {skipped_empty} with no usable boxes)")
    for c in sorted(boxes_added, key=lambda c: -boxes_added[c]):
        print(f"  {v2_names[c]:<28} +{boxes_added[c]} boxes")
    n = {s: len(images_in(dst / s / "images")) for s in ("train", "valid", "test")}
    print(f"\n{dst}: train {n['train']}, valid {n['valid']}, test {n['test']}")
    print(f"Train with: data=r'{dst / 'data.yaml'}'")


if __name__ == "__main__":
    main()
