"""
Build dataset v2: group near-duplicate images, decide classes, re-split by group.

Two stages, run in order:

  1) REPORT  (reads only, changes nothing)
       python group_and_split.py report
     Writes into the source folder:
       groups.csv            one row per image: its group, current split
       class_report.csv      per class: boxes, images, source groups, per-split groups
       class_decisions.csv   a template you edit: keep / merge / drop per class
     Prints how many groups are split across train/valid/test today (= leakage).

  2) BUILD   (writes a NEW folder, never touches the source)
       python group_and_split.py build
     Reads class_decisions.csv, remaps class IDs, drops images that contain a
     dropped class, assigns every group to ONE split (rarest classes first,
     about 80/10/10), and writes D:\\ewaste_v2 with data.yaml and split_report.csv.

How images are grouped (union-find):
  - same flip-invariant difference hash (catches Roboflow's flipped copies and
    exact re-uploads under another name), and
  - same Roboflow source stem (text before ".rf.") AND hashes within
    MAX_STEM_DISTANCE bits (generic names like "12_jpg" come from different
    photos, so the name alone is not enough).

Needs only Pillow and PyYAML (both come with ultralytics).
"""

from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import random
import shutil
import sys

import yaml
from PIL import Image, ImageOps

SPLITS = ("train", "valid", "test")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_STEM_DISTANCE = 10      # bits of 64, for images sharing a source stem
DEFAULT_MIN_GROUPS = 25     # suggested drop/merge threshold in the template


# ---------------------------------------------------------------- hashing

def dhash(image, size=8):
    """64-bit difference hash."""
    g = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.tobytes())  # one byte per pixel in "L" mode
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hashes(path):
    with Image.open(path) as im:
        im.load()
        return dhash(im), dhash(ImageOps.mirror(im))


def hamming(a, b):
    return bin(a ^ b).count("1")


def source_stem(stem):
    return stem.split(".rf.")[0]


# ---------------------------------------------------------------- loading

def load_names(src):
    with open(src / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def index_images(src):
    items = []
    for split in SPLITS:
        img_dir, lab_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in EXTS:
                lab = lab_dir / f"{img.stem}.txt"
                if not lab.exists():
                    sys.exit(f"Missing label for {img}. Run validate_final_dataset.py first.")
                items.append((split, img, lab))
    return items


def read_boxes(label_path):
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), parts[1:]))
    return rows


def load_hash_cache(path):
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cache[r["path"]] = (int(r["size"]), float(r["mtime"]), int(r["h"]), int(r["hm"]))
    return cache


def compute_hashes(items, cache_path):
    cache = load_hash_cache(cache_path)
    out = []
    fresh = 0
    for i, (_, img, _) in enumerate(items, 1):
        st = img.stat()
        c = cache.get(str(img))
        if c and c[0] == st.st_size and abs(c[1] - st.st_mtime) < 1e-3:
            h, hm = c[2], c[3]
        else:
            h, hm = hashes(img)
            cache[str(img)] = (st.st_size, st.st_mtime, h, hm)
            fresh += 1
        out.append((h, hm))
        if i % 4000 == 0:
            print(f"  checked {i}/{len(items)} images ({fresh} newly hashed)")
    with cache_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "mtime", "h", "hm"])
        for p, (sz, mt, h, hm) in cache.items():
            w.writerow([p, sz, mt, h, hm])
    return out


# ---------------------------------------------------------------- grouping

def build_groups(items, hs):
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 1) identical flip-invariant hash
    by_hash = {}
    for i, (h, hm) in enumerate(hs):
        key = min(h, hm)
        if key in by_hash:
            union(i, by_hash[key])
        else:
            by_hash[key] = i

    # 2) same source stem and visually close (either orientation)
    by_stem = defaultdict(list)
    for i, (_, img, _) in enumerate(items):
        by_stem[source_stem(img.stem)].append(i)
    for members in by_stem.values():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                ha, hma = hs[a]
                hb, _ = hs[b]
                if min(hamming(ha, hb), hamming(hma, hb)) <= MAX_STEM_DISTANCE:
                    union(a, b)

    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    # stable, readable group ids
    ordered = sorted(groups.values(), key=lambda m: str(items[m[0]][1].name))
    return {gid: members for gid, members in enumerate(ordered)}


# ---------------------------------------------------------------- report

def report(src, min_groups):
    names = load_names(src)
    items = index_images(src)
    print(f"Images found: {len(items)}")
    print("Hashing (cached in hash_cache.csv, so re-runs are fast)...")
    hs = compute_hashes(items, src / "hash_cache.csv")
    groups = build_groups(items, hs)

    boxes, images, grp = Counter(), Counter(), defaultdict(set)
    grp_split = defaultdict(lambda: defaultdict(set))
    mixed = 0
    with (src / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "split", "image", "classes"])
        for gid, members in groups.items():
            if len({items[i][0] for i in members}) > 1:
                mixed += 1
            for i in members:
                split, img, lab = items[i]
                cls = [c for c, _ in read_boxes(lab)]
                w.writerow([gid, split, img.name, " ".join(map(str, sorted(set(cls))))])
                for c in cls:
                    boxes[c] += 1
                for c in set(cls):
                    images[c] += 1
                    grp[c].add(gid)
                    grp_split[c][split].add(gid)

    multi = sum(1 for m in groups.values() if len(m) > 1)
    print(f"\nGroups (distinct source images): {len(groups)}")
    print(f"Groups with more than one image: {multi}")
    print(f"Groups spread over 2+ splits today (leakage): {mixed}")

    with (src / "class_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "boxes", "images", "groups",
                    "train_groups", "valid_groups", "test_groups"])
        for c in sorted(names):
            w.writerow([c, names[c], boxes[c], images[c], len(grp[c]),
                        len(grp_split[c]["train"]), len(grp_split[c]["valid"]),
                        len(grp_split[c]["test"])])

    dec = src / "class_decisions.csv"
    if dec.exists():
        print(f"\n{dec.name} already exists; not overwritten.")
    else:
        with dec.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "name", "groups", "action", "merge_into", "note"])
            for c in sorted(names):
                n = len(grp[c])
                action = "keep" if n >= min_groups else "drop"
                w.writerow([c, names[c], n, action, "",
                            "" if action == "keep" else f"under {min_groups} groups: drop, or merge"])
        print(f"\nWrote {dec.name}. Edit the 'action' column (keep / merge / drop);")
        print("for merge, put the TARGET CLASS NAME in 'merge_into' (an existing or new name).")

    print("\nSmallest classes by distinct source groups:")
    for c in sorted(names, key=lambda c: len(grp[c]))[:25]:
        print(f"  {names[c]:<32} groups={len(grp[c]):>4}  boxes={boxes[c]:>5}")
    print(f"\nReports written to {src}")


# ---------------------------------------------------------------- build

def load_decisions(src, names):
    path = src / "class_decisions.csv"
    if not path.exists():
        sys.exit("class_decisions.csv not found. Run the 'report' step first.")
    new_names, remap, dropped = [], {}, set()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:                                   # keeps first, in old order
        if r["action"].strip().lower() == "keep":
            new_names.append(r["name"].strip())
    for r in rows:
        c = int(r["class_id"])
        act = r["action"].strip().lower()
        if act == "keep":
            remap[c] = new_names.index(r["name"].strip())
        elif act == "merge":
            target = r["merge_into"].strip()
            if not target:
                sys.exit(f"Class {c} ({names[c]}) is 'merge' but merge_into is empty.")
            if target not in new_names:
                new_names.append(target)
            remap[c] = new_names.index(target)
        elif act == "drop":
            dropped.add(c)
        else:
            sys.exit(f"Unknown action '{r['action']}' for class {c}. Use keep / merge / drop.")
    missing = set(names) - set(remap) - dropped
    if missing:
        sys.exit(f"No decision for class IDs: {sorted(missing)}")
    return new_names, remap, dropped


def build(src, dst, ratios, seed):
    if dst.exists():
        sys.exit(f"{dst} already exists. Rename or remove it first.")
    names = load_names(src)
    new_names, remap, dropped = load_decisions(src, names)
    items = index_images(src)
    hs = compute_hashes(items, src / "hash_cache.csv")
    groups = build_groups(items, hs)

    # drop whole groups that contain a dropped class (no half-labelled images)
    keep_groups, removed = {}, []
    for gid, members in groups.items():
        cls = {c for i in members for c, _ in read_boxes(items[i][2])}
        if cls & dropped:
            removed.append((gid, len(members), sorted(cls & dropped)))
        else:
            keep_groups[gid] = {remap[c] for c in cls}

    freq = Counter(c for cs in keep_groups.values() for c in cs)
    order = sorted(keep_groups, key=lambda g: (min((freq[c] for c in keep_groups[g]), default=10**9), g))
    rng = random.Random(seed)
    seen = {s: Counter() for s in SPLITS}
    size = Counter()
    total = len(keep_groups)
    assign = {}
    for g in order:
        cs = keep_groups[g]
        # Iterative stratification: place the group where its RAREST class is most
        # under-represented. (Summing over all classes let common classes outvote
        # rare ones and pushed every mixed group into train.)
        rare = min(cs, key=lambda c: (freq[c], c)) if cs else None

        def deficit(s):
            need = ratios[s] * freq[rare] - seen[s][rare] if rare is not None else 0.0
            overall = ratios[s] * total - size[s]
            return (need, overall, rng.random())

        s = max(SPLITS, key=deficit)
        assign[g] = s
        seen[s].update(cs)
        size[s] += 1

    counts = Counter()
    for g, s in assign.items():
        for i in groups[g]:
            _, img, lab = items[i]
            (dst / s / "images").mkdir(parents=True, exist_ok=True)
            (dst / s / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / s / "images" / img.name)
            lines = [f"{remap[c]} {' '.join(xywh)}" for c, xywh in read_boxes(lab)]
            (dst / s / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[s] += 1

    cfg = {"path": str(dst), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": dict(enumerate(new_names)), "nc": len(new_names)}
    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with (dst / "split_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "train_groups", "valid_groups", "test_groups"])
        for c, n in enumerate(new_names):
            w.writerow([c, n, seen["train"][c], seen["valid"][c], seen["test"][c]])
    with (dst / "dropped_groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "images", "dropped_classes"])
        w.writerows(removed)

    print(f"Classes: {len(names)} -> {len(new_names)}  (dropped {len(dropped)})")
    print(f"Groups dropped because they contain a dropped class: {len(removed)}")
    print(f"Images per split: " + ", ".join(f"{s}={counts[s]}" for s in SPLITS))
    thin = [n for c, n in enumerate(new_names) if min(seen[s][c] for s in SPLITS) == 0]
    if thin:
        print("WARNING - classes missing from at least one split:", ", ".join(thin))
    print(f"Wrote {dst}. Next: python validate_final_dataset.py (point ROOT at it).")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["report", "build"])
    ap.add_argument("--src", default=r"D:\ewaste_final")
    ap.add_argument("--dst", default=r"D:\ewaste_v2")
    ap.add_argument("--min-groups", type=int, default=DEFAULT_MIN_GROUPS)
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,valid,test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.src)
    if a.command == "report":
        report(src, a.min_groups)
    else:
        r = [float(x) for x in a.ratios.split(",")]
        build(src, Path(a.dst), dict(zip(SPLITS, r)), a.seed)


if __name__ == "__main__":
    main()
