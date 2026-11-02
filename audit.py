from pathlib import Path
from collections import Counter
import csv
import yaml
import numpy as np
from PIL import Image

ROOT = Path(".")
data = yaml.safe_load(open("data.yaml", encoding="utf-8"))
names = data["names"]
splits = ["train", "valid", "test"]

counts = {s: Counter() for s in splits}
missing_labels = []
empty_labels = []
malformed = []
image_stems = {}
source_names = {}

# 1. Dataset and annotation audit
for split in splits:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = {
        p.stem: p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    }

    labels = {p.stem: p for p in label_dir.glob("*.txt")}

    image_stems[split] = images
    print(f"\n{split}: {len(images)} images, {len(labels)} label files")

    for stem in images.keys() - labels.keys():
        missing_labels.append((split, stem))

    for stem, label_path in labels.items():
        if not label_path.exists():
            missing_labels.append((split, stem))
            continue
        lines = [
        line.split()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()]

        if not lines:
            empty_labels.append((split, stem))
            continue

        formats = set()

        for row_num, row in enumerate(lines, start=1):
            try:
                cls = int(row[0])
                coords = list(map(float, row[1:]))

                if cls < 0 or cls >= len(names):
                    raise ValueError("class ID out of range")

                if len(coords) == 4:
                    fmt = "box"
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    fmt = "polygon"
                else:
                    raise ValueError("invalid coordinate count")

                if not np.isfinite(coords).all():
                    raise ValueError("non-finite coordinates")

                if any(v < 0 or v > 1 for v in coords):
                    raise ValueError("coordinates outside [0,1]")

                if fmt == "box":
                    x, y, w, h = coords
                    if w <= 0 or h <= 0:
                        raise ValueError("non-positive box dimensions")
                    if x - w/2 < -1e-6 or x + w/2 > 1 + 1e-6:
                        raise ValueError("box exceeds image horizontally")
                    if y - h/2 < -1e-6 or y + h/2 > 1 + 1e-6:
                        raise ValueError("box exceeds image vertically")

                counts[split][cls] += 1
                formats.add(fmt)

            except Exception as e:
                malformed.append((split, stem, row_num, str(e)))

        if len(formats) > 1:
            print(f"WARNING: mixed box/polygon formats: {split}/{stem}")

print("\n--- Annotation audit summary ---")
print("Images without label files:", len(missing_labels))
print("Empty label files:", len(empty_labels))
print("Malformed annotation rows:", len(malformed))

if missing_labels:
    print("Missing-label examples:", missing_labels[:10])
if empty_labels:
    print("Empty-label examples:", empty_labels[:10])
if malformed:
    print("Malformed-row examples:", malformed[:10])

# 2. Per-class instance counts
with open("class_counts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class_id", "name", "train", "valid", "test", "flag"])

    for i, name in enumerate(names):
        tr = counts["train"][i]
        va = counts["valid"][i]
        te = counts["test"][i]
        writer.writerow([
            i, name, tr, va, te,
            "THIN" if tr < 50 else ""
        ])

print("\nWrote class_counts.csv")

# 3. Filename overlap
def source_key(path):
    return path.name.split(".rf.")[0]

source_names = {
    s: {source_key(p) for p in image_stems[s].values()}
    for s in splits
}

for a, b in [("train", "test"), ("train", "valid"), ("valid", "test")]:
    overlap = source_names[a] & source_names[b]
    print(f"Same-source filename overlap {a}/{b}: {len(overlap)}")

# 4. Near-duplicate check using difference hash
def dhash(path):
    with Image.open(path) as im:
        px = np.asarray(
            im.convert("L").resize((9, 8)),
            dtype=np.int16
        )
    bits = px[:, :-1] > px[:, 1:]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value

try:
    train_hashes = [
        (p.name, dhash(p))
        for p in image_stems["train"].values()
    ]

    train_values = np.array(
        [h for _, h in train_hashes], dtype=np.uint64
    )

    for split in ["test", "valid"]:
        exact = 0
        near = []

        for name, path in image_stems[split].items():
            h = np.uint64(dhash(path))
            distances = [
                int((int(h) ^ int(th)).bit_count())
                for th in train_values
            ]

            best = min(distances) if distances else 64

            if best == 0:
                exact += 1
            if best <= 4:
                near.append((path.name, best))

        print(
            f"{split}: exact train duplicates={exact}; "
            f"near-duplicates (distance <= 4)={len(near)} "
            f"of {len(image_stems[split])}"
        )

        if split == "test":
            with open(
                "test_near_dups.csv", "w",
                newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["image", "distance"])
                writer.writerows(near)

    print("Wrote test_near_dups.csv")

except Exception as e:
    print("Near-duplicate check failed:", repr(e))
    print("The annotation audit and class counts were still generated.")