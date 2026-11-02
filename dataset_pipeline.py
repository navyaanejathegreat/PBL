from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import sqlite3

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def images_in(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def labels_by_stem(folder):
    result, duplicates = {}, []
    folder = Path(folder)
    if not folder.exists():
        return result, duplicates
    for p in folder.rglob("*.txt"):
        key = p.relative_to(folder).with_suffix("").as_posix()
        if key in result:
            duplicates.append((key, str(result[key]), str(p)))
        else:
            result[key] = p
    return result, duplicates


def label_key_for_image(image_path, image_root):
    return image_path.relative_to(image_root).with_suffix("").as_posix()


def label_stats(path, num_classes=None):
    """Validate YOLO boxes and polygon rows; return class counts, bad-row count, mixed-format flag."""
    counts, formats, malformed = Counter(), set(), 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return counts, 1, False

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            cls_value = float(parts[0])
            if not cls_value.is_integer():
                raise ValueError
            cls = int(cls_value)
            coords = [float(x) for x in parts[1:]]
            if cls < 0 or (num_classes is not None and cls >= num_classes):
                raise ValueError
            if not coords or not all(0.0 <= x <= 1.0 for x in coords):
                raise ValueError
            if len(coords) == 4:
                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError
                formats.add("box")
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                formats.add("polygon")
            else:
                raise ValueError
            counts[cls] += 1
        except (ValueError, OverflowError):
            malformed += 1
    return counts, malformed, len(formats) > 1


def audit(root, num_classes=None):
    root = Path(root)
    rows = []
    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        imgs = images_in(image_dir)
        labels, duplicate_stems = labels_by_stem(label_dir)
        image_keys = {label_key_for_image(p, image_dir) for p in imgs}
        counts = Counter()
        missing = malformed = mixed = instances = empty = 0

        for img in imgs:
            lab = labels.get(label_key_for_image(img, image_dir))
            if lab is None:
                missing += 1
                continue
            c, bad, mix = label_stats(lab, num_classes)
            counts.update(c)
            instances += sum(c.values())
            malformed += bad
            mixed += int(mix)
            try:
                empty += int(not lab.read_text(encoding="utf-8").strip())
            except (OSError, UnicodeError):
                pass

        row = {
            "split": split, "images": len(imgs), "label_files": len(labels),
            "missing_labels": missing, "orphan_labels": len(set(labels) - image_keys),
            "duplicate_label_stems": len(duplicate_stems), "empty_labels": empty,
            "instances": instances, "classes_present": len(counts),
            "malformed_rows": malformed, "mixed_format_files": mixed,
        }
        rows.append(row)
        print(f"\n===== {split.upper()} =====")
        for key, value in row.items():
            if key != "split":
                print(f"{key}: {value}")
        print("Class counts:", dict(sorted(counts.items())))
        if duplicate_stems:
            print("Duplicate label paths (first 5):", duplicate_stems[:5])

    out = root / "pipeline_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAudit report: {out}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(root, cross_split_only=True):
    """Find byte-identical files using cached SHA-256 hashes. Never deletes files."""
    root = Path(root).resolve()
    db_path = root / "pipeline_hashes.sqlite"
    report = root / ("cross_split_duplicates.csv" if cross_split_only else "duplicate_candidates.csv")
    files = [(split, p) for split in SPLITS for p in images_in(root / split / "images")]
    print(f"Scanning {len(files)} images. Cached hashes are reused when size and modification time match.")

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hashes (
            path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime REAL NOT NULL, sha256 TEXT NOT NULL
        )
    """)
    conn.commit()
    by_hash = defaultdict(list)

    for i, (split, path) in enumerate(files, 1):
        key = str(path.resolve())
        stat = path.stat()
        cached = conn.execute(
            "SELECT size, mtime, sha256 FROM hashes WHERE path=?", (key,)
        ).fetchone()
        if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime:
            digest = cached[2]
        else:
            digest = sha256_file(path)
            conn.execute(
                "INSERT OR REPLACE INTO hashes(path, size, mtime, sha256) VALUES (?, ?, ?, ?)",
                (key, stat.st_size, stat.st_mtime, digest),
            )
        by_hash[digest].append((split, str(path)))
        if i % 1000 == 0:
            conn.commit()
            print(f"Processed {i}/{len(files)}")
    conn.commit()
    conn.close()

    pair_count = cross_count = 0
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "split", "path", "matching_split", "matching_path"])
        for digest, items in by_hash.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                split_a, path_a = items[i]
                for j in range(i + 1, len(items)):
                    split_b, path_b = items[j]
                    cross = split_a != split_b
                    if cross:
                        cross_count += 1
                    if cross_split_only and not cross:
                        continue
                    writer.writerow([digest, split_a, path_a, split_b, path_b])
                    pair_count += 1

    print(f"Duplicate pairs written to report: {pair_count}")
    print(f"Cross-split exact duplicate pairs found: {cross_count}")
    print(f"Report: {report}")
    print("No images were deleted. SHA-256 detects byte-identical files only.")


def verify(root, num_classes=None):
    root = Path(root)
    all_counts, had_issues = {}, False
    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        imgs = images_in(image_dir)
        labels, duplicate_stems = labels_by_stem(label_dir)
        image_keys = set()
        counts = Counter()
        missing = malformed = 0
        for img in imgs:
            key = label_key_for_image(img, image_dir)
            image_keys.add(key)
            lab = labels.get(key)
            if lab is None:
                missing += 1
                continue
            c, bad, _ = label_stats(lab, num_classes)
            counts.update(c)
            malformed += bad
        orphan = len(set(labels) - image_keys)
        issues = missing + malformed + orphan + len(duplicate_stems)
        had_issues |= issues > 0
        all_counts[split] = counts
        print(f"{split}: images={len(imgs)}, instances={sum(counts.values())}, "
              f"missing_labels={missing}, orphan_labels={orphan}, "
              f"duplicate_label_stems={len(duplicate_stems)}, malformed_rows={malformed}")

    train = all_counts["train"]
    if train:
        max_class = max((max(c.keys(), default=-1) for c in all_counts.values()), default=-1)
        missing_classes = [i for i in range(max_class + 1) if train[i] == 0]
        print("Class IDs with zero training instances:", missing_classes)
    else:
        print("No training instances found.")
        had_issues = True
    print("Verification completed with issues." if had_issues else
          "Verification passed: no missing/orphan labels or malformed rows detected.")


def main():
    parser = argparse.ArgumentParser(description="Reusable YOLO dataset pipeline")
    parser.add_argument("command", choices=["audit", "deduplicate", "verify"])
    parser.add_argument("--root", required=True, help="Dataset root containing train/valid/test")
    parser.add_argument("--classes", type=int, default=None,
                        help="Optional class count; validates class IDs against this range")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Report duplicates within the same split as well as cross-split duplicates")
    args = parser.parse_args()
    if args.classes is not None and args.classes <= 0:
        parser.error("--classes must be positive")
    if args.command == "audit":
        audit(args.root, args.classes)
    elif args.command == "deduplicate":
        deduplicate(args.root, cross_split_only=not args.all_pairs)
    else:
        verify(args.root, args.classes)


if __name__ == "__main__":
    main()
