from pathlib import Path
from collections import defaultdict
import hashlib
import shutil
import csv

SOURCE = Path(r"D:\ewaste_clean")
DEST = Path(r"D:\ewaste_final")
SPLITS = ["test", "valid", "train"]  # Keep test copies first
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

if DEST.exists():
    raise FileExistsError(
        f"{DEST} already exists. Rename or remove it before running this script."
    )

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Index images by their exact content hash.
groups = defaultdict(list)

for split in SPLITS:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing images or labels for split: {split}")

    for img in image_dir.iterdir():
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            label = label_dir / f"{img.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {img}")
            groups[sha256(img)].append((split, img, label))

# Choose one copy of each identical image, preferring test, then valid, then train.
kept = []
removed = []

for digest, entries in groups.items():
    entries.sort(key=lambda x: SPLITS.index(x[0]))
    kept.append(entries[0])

    for entry in entries[1:]:
        removed.append((digest, entry[0], str(entry[1]), str(entries[0][1])))

# Copy retained images and matching labels.
for split, img, label in kept:
    out_img = DEST / split / "images" / img.name
    out_label = DEST / split / "labels" / label.name
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out_img)
    shutil.copy2(label, out_label)

# Copy the dataset YAML unchanged.
yaml_source = SOURCE / "data.yaml"
if not yaml_source.exists():
    raise FileNotFoundError(f"Missing dataset YAML: {yaml_source}")
shutil.copy2(yaml_source, DEST / "data.yaml")

# Save a removal log for reproducibility.
with (DEST / "removed_duplicates.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["sha256", "removed_split", "removed_image", "kept_image"])
    writer.writerows(removed)

print("\nDataset correction complete.")
print(f"Original unique image groups: {len(groups)}")
print(f"Images retained: {len(kept)}")
print(f"Duplicate copies removed: {len(removed)}")
print(f"Corrected dataset: {DEST}")
print(f"Removal log: {DEST / 'removed_duplicates.csv'}")