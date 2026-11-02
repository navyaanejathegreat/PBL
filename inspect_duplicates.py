from pathlib import Path
import csv
from PIL import Image, ImageDraw

ROOT = Path(".")
CSV_FILE = ROOT / "test_near_dups.csv"
OUTPUT = ROOT / "duplicate_review"

OUTPUT.mkdir(exist_ok=True)

with open(CSV_FILE, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Review exact hash matches first
exact = [r for r in rows if int(r["distance"]) == 0]

print(f"Exact-hash candidates: {len(exact)}")

test_dir = ROOT / "test" / "images"
train_dir = ROOT / "train" / "images"

# Recreate the training hashes
def dhash(path):
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8))
        pixels = list(im.getdata())

    bits = []
    for y in range(8):
        row = pixels[y * 9:(y + 1) * 9]
        bits.extend(row[x] > row[x + 1] for x in range(8))

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return value

train_hashes = {}

print("Hashing training images...")

for path in train_dir.iterdir():
    if path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        continue

    try:
        h = dhash(path)
        train_hashes.setdefault(h, []).append(path)
    except Exception:
        pass

print("Comparing exact-hash candidates...")

for i, row in enumerate(exact[:30], start=1):
    test_path = test_dir / row["image"]

    if not test_path.exists():
        print("Missing test image:", row["image"])
        continue

    try:
        h = dhash(test_path)
        matches = train_hashes.get(h, [])

        if not matches:
            continue

        train_path = matches[0]

        with Image.open(test_path) as im:
            test_im = im.convert("RGB")
            test_im.thumbnail((400, 300))

        with Image.open(train_path) as im:
            train_im = im.convert("RGB")
            train_im.thumbnail((400, 300))

        canvas = Image.new("RGB", (820, 360), "white")
        canvas.paste(train_im, (5, 35))
        canvas.paste(test_im, (415, 35))

        draw = ImageDraw.Draw(canvas)
        draw.text((5, 8), f"TRAIN: {train_path.name}", fill="black")
        draw.text((415, 8), f"TEST: {test_path.name}", fill="black")

        output_path = OUTPUT / f"pair_{i:02d}.jpg"
        canvas.save(output_path)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Could not inspect {row['image']}: {e}")

print("\nOpen the duplicate_review folder to inspect the image pairs.")