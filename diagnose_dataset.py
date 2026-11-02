from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(r"D:\E-Waste Dataset.v1i.yolov8")

with open(ROOT / "data.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, list):
    names = dict(enumerate(names))


def inspect_split(split):
    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"

    counts = Counter()
    mixed_files = []
    malformed = []
    missing_labels = []
    empty_labels = []

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".webp", ".tif", ".tiff"
    }

    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    image_stems = {p.stem for p in images}

    for image in images:
        label_file = labels_dir / f"{image.stem}.txt"

        if not label_file.exists():
            missing_labels.append(image.name)
            continue

        try:
            lines = label_file.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as error:
            malformed.append(f"{label_file.name}: {error}")
            continue

        row_types = set()
        valid_rows = 0

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if class_id not in names:
                    raise ValueError(f"Unknown class ID: {class_id}")

                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError("Coordinates outside [0, 1]")

                if len(coords) == 4:
                    row_types.add("box")

                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    row_types.add("polygon")

                else:
                    raise ValueError(
                        f"Unexpected coordinate count: {len(coords)}"
                    )

                counts[class_id] += 1
                valid_rows += 1

            except Exception as error:
                malformed.append(
                    f"{label_file.name}:{line_number}: {error}"
                )

        if valid_rows == 0:
            empty_labels.append(label_file.name)

        if len(row_types) > 1:
            mixed_files.append(label_file.name)

    label_stems = {
        p.stem for p in labels_dir.glob("*.txt")
    }

    orphan_labels = sorted(label_stems - image_stems)

    print(f"\n===== {split.upper()} =====")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(list(labels_dir.glob('*.txt')))}")
    print(f"Images missing labels: {len(missing_labels)}")
    print(f"Empty/invalid label files: {len(empty_labels)}")
    print(f"Orphan label files: {len(orphan_labels)}")
    print(f"Mixed box/polygon files: {len(mixed_files)}")
    print(f"Malformed rows: {len(malformed)}")
    print(f"Annotated instances: {sum(counts.values())}")
    print(f"Classes represented: {len(counts)}")

    if missing_labels:
        print("Images missing labels:")
        for item in missing_labels[:10]:
            print(" ", item)

    if orphan_labels:
        print("Orphan labels:")
        for item in orphan_labels[:10]:
            print(" ", item)

    if mixed_files:
        print("Mixed annotation files:")
        for item in mixed_files:
            print(" ", item)

    if malformed:
        print("Malformed annotation rows:")
        for item in malformed[:20]:
            print(" ", item)

    return counts


train = inspect_split("train")
valid = inspect_split("valid")
test = inspect_split("test")

print("\n===== CLASSES WITH ZERO TRAINING INSTANCES =====")

for class_id, class_name in names.items():
    if train[class_id] == 0:
        print(
            f"ID {class_id}: {class_name} | "
            f"valid={valid[class_id]} | "
            f"test={test[class_id]}"
        )

print("\n===== BATTERY AND HDD COUNTS =====")

for class_id, class_name in names.items():
    if class_name in {"Battery", "HDD"}:
        print(
            f"{class_name}: "
            f"train={train[class_id]}, "
            f"valid={valid[class_id]}, "
            f"test={test[class_id]}"
        )