from pathlib import Path
from collections import Counter
import shutil
import yaml
import csv
import hashlib

ROOT = Path(r"D:\E-Waste Dataset.v1i.yolov8")
OUTPUT = Path(r"D:\ewaste_clean")

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

with open(ROOT / "data.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, list):
    names = dict(enumerate(names))

class_counts = {split: Counter() for split in SPLITS}
report = []

# Avoid accidentally overwriting an earlier prepared dataset.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename it or remove it "
        "after checking its contents before running again."
    )


def convert_label(source, destination, split, image_name):
    """Convert valid box/polygon rows to YOLO detection boxes."""

    if not source.exists():
        report.append([
            split, image_name, "missing_label", 0
        ])
        return False

    output_rows = []

    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1
    ):
        parts = line.split()

        if not parts:
            continue

        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]

            if class_id not in names:
                raise ValueError("Unknown class ID")

            if not all(0 <= v <= 1 for v in coords):
                raise ValueError("Coordinates outside [0, 1]")

            if len(coords) == 4:
                # Already a YOLO bounding box.
                cx, cy, width, height = coords

            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Polygon: x1,y1,x2,y2,...
                xs = coords[0::2]
                ys = coords[1::2]

                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

            else:
                raise ValueError(
                    f"Invalid coordinate count: {len(coords)}"
                )

            if width <= 0 or height <= 0:
                raise ValueError("Zero-area bounding box")

            # Clamp floating-point boundary noise.
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            width = min(width, 1.0)
            height = min(height, 1.0)

            output_rows.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{width:.6f} {height:.6f}"
            )

            class_counts[split][class_id] += 1

        except Exception as error:
            report.append([
                split,
                f"{image_name}:line{line_number}",
                f"invalid_row: {error}",
                0
            ])

    if not output_rows:
        report.append([
            split, image_name, "empty_or_invalid_label", 0
        ])
        return False

    destination.write_text(
        "\n".join(output_rows) + "\n",
        encoding="utf-8"
    )

    return True


for split in SPLITS:
    source_images = ROOT / split / "images"
    source_labels = ROOT / split / "labels"

    output_images = OUTPUT / split / "images"
    output_labels = OUTPUT / split / "labels"

    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)

    image_paths = sorted(
        p for p in source_images.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    kept = 0
    skipped = 0

    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"

        # Label and image must share the same stem, or YOLO cannot pair them.
        # (Earlier version saved labels under a SHA-1 name but kept the
        # original image name, so the two no longer matched.)
        destination_label = output_labels / f"{image_path.stem}.txt"

        if not convert_label(
            label_path,
            destination_label,
            split,
            image_path.name
        ):
            skipped += 1
            continue

        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        kept += 1

    print(
        f"{split}: retained {kept} images; "
        f"skipped {skipped} images"
    )


# Save class and annotation audit.
with open(
    OUTPUT / "preparation_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "split", "image_or_row", "status", "instances"
    ])
    writer.writerows(report)


# Build a detection-only dataset configuration.
clean_config = {
    "path": str(OUTPUT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": names,
    "nc": len(names),
}

with open(
    OUTPUT / "data.yaml",
    "w",
    encoding="utf-8"
) as f:
    yaml.safe_dump(
        clean_config,
        f,
        sort_keys=False,
        allow_unicode=True
    )


print("\n===== CLASSES WITH ZERO TRAINING INSTANCES =====")

for class_id, class_name in names.items():
    if class_counts["train"][class_id] == 0:
        print(
            f"ID {class_id}: {class_name} | "
            f"valid={class_counts['valid'][class_id]} | "
            f"test={class_counts['test'][class_id]}"
        )

print("\nPrepared dataset:", OUTPUT)
print("Dataset YAML:", OUTPUT / "data.yaml")
print("Preparation report:", OUTPUT / "preparation_report.csv")
print("\nOriginal dataset has not been modified.")