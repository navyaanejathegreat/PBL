from pathlib import Path
from collections import defaultdict
import hashlib
import sys

import yaml

# Usage: python validate_final_dataset.py [dataset_folder]   (default D:\ewaste_final)
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ewaste_final")
with open(ROOT / "data.yaml", encoding="utf-8") as _f:
    NC = int(yaml.safe_load(_f)["nc"])
SPLITS = ["train", "valid", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

errors = []
hash_splits = defaultdict(set)
total_images = 0
total_boxes = 0

for split in SPLITS:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}

    missing = image_stems - label_stems
    orphaned = label_stems - image_stems

    for stem in missing:
        errors.append(f"{split}: missing label for {stem}")

    for stem in orphaned:
        errors.append(f"{split}: label has no image: {stem}")

    split_boxes = 0

    for img in images:
        digest = hashlib.sha256(img.read_bytes()).hexdigest()
        hash_splits[digest].add(split)

        label = label_dir / f"{img.stem}.txt"

        if not label.exists():
            continue

        for line_no, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{split}/{label.name}:{line_no}: expected 5 values, "
                    f"found {len(parts)}"
                )
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{split}/{label.name}:{line_no}: non-numeric value"
                )
                continue

            if not 0 <= cls < NC:
                errors.append(
                    f"{split}/{label.name}:{line_no}: invalid class ID {cls}"
                )

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(
                    f"{split}/{label.name}:{line_no}: coordinate outside [0,1]"
                )

            if w <= 0 or h <= 0:
                errors.append(
                    f"{split}/{label.name}:{line_no}: non-positive box size"
                )

            split_boxes += 1

    total_images += len(images)
    total_boxes += split_boxes

    print(
        f"{split.upper()}: {len(images)} images, "
        f"{split_boxes} boxes, "
        f"{len(missing)} missing labels, "
        f"{len(orphaned)} orphan labels"
    )

cross_split = [
    digest for digest, splits in hash_splits.items()
    if len(splits) > 1
]

print("\n--- FINAL VALIDATION ---")
print("Total images:", total_images)
print("Total bounding boxes:", total_boxes)
print("Cross-split exact duplicates:", len(cross_split))
print("Annotation errors:", len(errors))

for error in errors[:30]:
    print("ERROR:", error)

if len(errors) > 30:
    print(f"... and {len(errors) - 30} more errors")

if errors or cross_split:
    print("\nVALIDATION FAILED. Do not train yet.")
else:
    print("\nVALIDATION PASSED.")