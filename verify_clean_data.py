from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(r"D:\ewaste_clean")

with open(ROOT / "data.yaml", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
if isinstance(names, list):
    names = dict(enumerate(names))

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "valid", "test"]:
    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    counts = Counter()
    errors = []
    empty = 0

    for image in images:
        label = labels_dir / f"{image.stem}.txt"

        if not label.exists():
            errors.append(f"Missing label: {image.name}")
            continue

        rows = [
            line.split()
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not rows:
            empty += 1
            continue

        for row in rows:
            try:
                class_id = int(row[0])
                coords = [float(v) for v in row[1:]]

                if class_id not in names:
                    raise ValueError("Unknown class ID")

                if len(coords) != 4:
                    raise ValueError("Expected 4 box coordinates")

                if not all(0 <= v <= 1 for v in coords):
                    raise ValueError("Coordinates outside [0, 1]")

                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError("Zero-area box")

                counts[class_id] += 1

            except Exception as e:
                errors.append(f"{label.name}: {e}")

    print(f"\n===== {split.upper()} =====")
    print("Images:", len(images))
    print("Instances:", sum(counts.values()))
    print("Empty labels:", empty)
    print("Errors:", len(errors))

    for error in errors[:10]:
        print(" ", error)

    if split == "train":
        print("\nClasses with zero training instances:")
        for class_id, name in names.items():
            if counts[class_id] == 0:
                print(f"  {class_id}: {name}")