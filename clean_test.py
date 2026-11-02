from pathlib import Path
from PIL import Image
import numpy as np
import hashlib
import shutil
import csv
import yaml

# =====================================================
# CONFIGURATION
# =====================================================

DATASET = Path(r"D:\E-Waste Dataset.v1i.yolov8")

TRAIN_IMAGES = DATASET / "train" / "images"
VALID_IMAGES = DATASET / "valid" / "images"
TEST_IMAGES = DATASET / "test" / "images"
TEST_LABELS = DATASET / "test" / "labels"

OUTPUT = DATASET / "clean_eval"
CLEAN_IMAGES = OUTPUT / "test" / "images"
CLEAN_LABELS = OUTPUT / "test" / "labels"

ORIGINAL_YAML = DATASET / "data.yaml"
CLEAN_YAML = OUTPUT / "data_clean.yaml"

# Candidate thresholds, not proof of duplicate identity.
MAX_HASH_DISTANCE = 4
MIN_CORRELATION = 0.95

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# =====================================================
# IMAGE UTILITIES
# =====================================================

def image_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_hash(path):
    """64-bit difference hash, robust to modest image resizing."""
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))
        pixels = np.asarray(image, dtype=np.int16)

    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0

    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return value


def image_thumbnail(path):
    """Normalized grayscale thumbnail for candidate verification."""
    with Image.open(path) as image:
        image = image.convert("L").resize((64, 64))

        array = np.asarray(image, dtype=np.float32)

    array -= array.mean()
    norm = np.linalg.norm(array)

    if norm == 0:
        return None

    return array / norm


def correlation(path_a, path_b):
    a = image_thumbnail(path_a)
    b = image_thumbnail(path_b)

    if a is None or b is None:
        return 0.0

    # Normalized correlation. Similarity is approximate, not proof.
    return float(np.sum(a * b))


def md5_file(path):
    digest = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# =====================================================
# STEP 1: VALIDATE PATHS
# =====================================================

for folder in [
    TRAIN_IMAGES,
    VALID_IMAGES,
    TEST_IMAGES,
    TEST_LABELS,
]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Required folder not found:\n{folder}"
        )

if not ORIGINAL_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{ORIGINAL_YAML}"
    )

print("=" * 60)
print("E-WASTE DATASET: AUTOMATED TEST CLEANUP")
print("=" * 60)


# =====================================================
# STEP 2: INDEX TRAIN + VALIDATION IMAGES
# =====================================================

reference_images = (
    image_files(TRAIN_IMAGES)
    + image_files(VALID_IMAGES)
)

test_images = image_files(TEST_IMAGES)

if not reference_images or not test_images:
    raise RuntimeError("Training, validation, or test images are missing.")

print(f"\nReference images: {len(reference_images)}")
print(f"Original test images: {len(test_images)}")

reference_hashes = []
reference_errors = []

for i, path in enumerate(reference_images, start=1):
    try:
        reference_hashes.append(image_hash(path))
    except Exception as error:
        reference_errors.append((str(path), str(error)))

    if i % 5000 == 0:
        print(f"Processed {i}/{len(reference_images)} reference images")

if reference_errors:
    with open(OUTPUT.parent / "reference_image_errors.csv", "w",
              newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "error"])
        writer.writerows(reference_errors)

# Keep each hash paired with its correct source path.
valid_references = []

for path in reference_images:
    try:
        valid_references.append((path, image_hash(path)))
    except Exception:
        pass

reference_images = [item[0] for item in valid_references]
reference_hashes = np.array(
    [item[1] for item in valid_references],
    dtype=np.uint64
)

print(f"Usable reference images: {len(reference_images)}")


# =====================================================
# STEP 3: PREPARE CLEAN OUTPUT
# =====================================================

# This output folder belongs to this script.
# Existing output is replaced, but the original dataset is untouched.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

CLEAN_IMAGES.mkdir(parents=True)
CLEAN_LABELS.mkdir(parents=True)


# =====================================================
# STEP 4: FIND LIKELY TRAIN/VALIDATION OVERLAP
# =====================================================

report_rows = []
kept = 0
removed = 0
missing_labels = 0
failed = 0

for index, test_path in enumerate(test_images, start=1):

    try:
        test_hash = image_hash(test_path)
        test_md5 = md5_file(test_path)

        # Vectorized Hamming distance between 64-bit hashes.
        xor_values = np.bitwise_xor(
            reference_hashes,
            np.uint64(test_hash)
        )

        byte_values = xor_values.view(np.uint8).reshape(-1, 8)

        distances = np.unpackbits(
            byte_values,
            axis=1
        ).sum(axis=1)

        candidates = np.flatnonzero(
            distances <= MAX_HASH_DISTANCE
        )

        duplicate_match = None
        best_distance = None
        best_correlation = None

        # Verify only perceptual-hash candidates.
        for candidate_index in candidates:
            reference_path = reference_images[candidate_index]

            distance = int(distances[candidate_index])

            # Exact file match is definitive.
            if md5_file(reference_path) == test_md5:
                duplicate_match = reference_path
                best_distance = distance
                best_correlation = 1.0
                break

            similarity = correlation(test_path, reference_path)

            if similarity >= MIN_CORRELATION:
                if (
                    best_correlation is None
                    or similarity > best_correlation
                ):
                    duplicate_match = reference_path
                    best_distance = distance
                    best_correlation = similarity

        label_path = TEST_LABELS / f"{test_path.stem}.txt"

        if not label_path.exists():
            report_rows.append([
                str(test_path), "missing_label", "",
                "", "", ""
            ])
            missing_labels += 1
            continue

        if duplicate_match is not None:
            report_rows.append([
                str(test_path),
                "likely_overlap",
                str(duplicate_match),
                best_distance,
                best_correlation,
                ""
            ])

            removed += 1

        else:
            # Preserve the original test labels unchanged.
            shutil.copy2(
                test_path,
                CLEAN_IMAGES / test_path.name
            )

            shutil.copy2(
                label_path,
                CLEAN_LABELS / label_path.name
            )

            report_rows.append([
                str(test_path), "kept", "",
                "", "", ""
            ])

            kept += 1

    except Exception as error:
        report_rows.append([
            str(test_path), "error", "",
            "", "", str(error)
        ])
        failed += 1

    if index % 100 == 0 or index == len(test_images):
        print(
            f"Test images processed: {index}/{len(test_images)} "
            f"| kept: {kept} | likely overlaps: {removed}"
        )


# =====================================================
# STEP 5: SAVE REPORT
# =====================================================

REPORT = OUTPUT / "test_dup_report.csv"

with open(REPORT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "test_image",
        "status",
        "matched_reference",
        "hash_distance",
        "correlation",
        "error"
    ])

    writer.writerows(report_rows)


# =====================================================
# STEP 6: CREATE A SEPARATE EVALUATION YAML
# =====================================================

with open(ORIGINAL_YAML, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Keep original train and validation paths.
# Only the test path points to the cleaned test set.
data["path"] = str(DATASET)
data["train"] = "train/images"
data["val"] = "valid/images"
data["test"] = "clean_eval/test/images"

with open(CLEAN_YAML, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        data,
        file,
        sort_keys=False,
        allow_unicode=True
    )


# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("CLEAN TEST SET CREATED")
print("=" * 60)

print(f"Original test images: {len(test_images)}")
print(f"Retained images: {kept}")
print(f"Likely overlapping images excluded: {removed}")
print(f"Images missing labels: {missing_labels}")
print(f"Images with processing errors: {failed}")

print(f"\nClean test images: {CLEAN_IMAGES}")
print(f"Clean test labels: {CLEAN_LABELS}")
print(f"Duplicate report: {REPORT}")
print(f"Evaluation YAML: {CLEAN_YAML}")

print("\nIMPORTANT:")
print("Excluded images are suspected overlaps, not confirmed duplicates.")
print("The original dataset has not been modified.")
print("Review the report before interpreting cleaned-test metrics.")