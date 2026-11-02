from pathlib import Path
from collections import Counter
import argparse
import csv

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def audit(root):
    root = Path(root)
    report = []

    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"Skipping {split}: images or labels folder missing.")
            continue

        images = [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        labels = {
            p.stem: p for p in label_dir.rglob("*.txt")
        }

        counts = Counter()
        missing_labels = []
        malformed_rows = 0
        instances = 0

        for image_path in images:
            label_path = labels.get(image_path.stem)

            if label_path is None:
                missing_labels.append(image_path.name)
                continue

            try:
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines():

                    parts = line.split()

                    if not parts:
                        continue

                    # Detection labels: class + 4 box coordinates.
                    # Segmentation labels: class + polygon coordinates.
                    if len(parts) < 5:
                        malformed_rows += 1
                        continue

                    class_id = int(parts[0])
                    coordinates = [float(x) for x in parts[1:]]

                    if (
                        len(coordinates) != 4
                        and (
                            len(coordinates) < 6
                            or len(coordinates) % 2 != 0
                        )
                    ):
                        malformed_rows += 1
                        continue

                    if not all(
                        0 <= value <= 1 for value in coordinates
                    ):
                        malformed_rows += 1
                        continue

                    counts[class_id] += 1
                    instances += 1

            except (OSError, ValueError):
                malformed_rows += 1

        split_report = {
            "split": split,
            "images": len(images),
            "label_files": len(labels),
            "missing_labels": len(missing_labels),
            "instances": instances,
            "classes_present": len(counts),
            "malformed_rows": malformed_rows,
        }

        report.append(split_report)

        print(f"\n===== {split.upper()} =====")
        for key, value in split_report.items():
            if key != "split":
                print(f"{key}: {value}")

        if missing_labels:
            print("First missing labels:", missing_labels[:10])

        print("Class counts:", dict(sorted(counts.items())))

    output = root / "pipeline_audit.csv"

    if report:
        with output.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=report[0].keys()
            )
            writer.writeheader()
            writer.writerows(report)

        print(f"\nAudit report saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["audit"]
    )
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    if args.command == "audit":
        audit(args.root)