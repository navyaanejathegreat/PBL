"""
Download extra training images for your e-waste classes from Google Open Images V7
and save them in YOLO format with YOUR class IDs (from D:\\ewaste_final\\data.yaml).

Open Images photos are "in the wild" (cluttered rooms, odd angles, partial views),
which is exactly what a product-photo dataset like Roboflow's lacks.

Setup (use a SEPARATE environment so your training environment is not touched):
    python -m venv D:\\venvs\\data
    D:\\venvs\\data\\Scripts\\activate
    pip install fiftyone pyyaml

Run:
    python download_openimages.py --per-class 400
    python download_openimages.py --per-class 400 --only Hair-Dryer Camera Printer

Output (default D:\\extra_data\\openimages):
    train\\images\\*.jpg, train\\labels\\*.txt   YOLO boxes using your 77 class IDs
    data.yaml                                  same class list as ewaste_final
    download_report.csv                        images and boxes per class

Notes
- Each Open Images class is mapped to one of your class names below. Names Open
  Images does not have are reported and skipped, so the script never guesses.
- "Group" boxes (one box around several objects) are removed.
- Only boxes of the mapped classes are kept. Other e-waste objects in the same
  photo may be unlabelled; that is the price of this source, so review a sample.
- By default only the validation and test splits are used: their annotation
  files are small (about 100 MB). Add --splits validation test train for more
  images per class at the cost of about 2 GB more annotation files.
"""

from pathlib import Path
from collections import Counter
import argparse
import csv
import shutil
import sys

import yaml

# Open Images class name  ->  your class name (D:\ewaste_final\data.yaml)
MAPPING = {
    "Mobile phone": "Smartphone",
    "Laptop": "Laptop",
    "Computer keyboard": "Computer-Keyboard",
    "Computer mouse": "Computer-Mouse",
    "Computer monitor": "Flat-Panel-Monitor",
    "Television": "Flat-Panel-TV",          # some old CRT TVs too: spot-check
    "Tablet computer": "Tablet",
    "Headphones": "Headphone",
    "Remote control": "TV-Remote-Control",
    "Camera": "Camera",
    "Printer": "Printer",
    "Calculator": "Calculator",
    "Microwave oven": "Microwave",
    "Refrigerator": "Refrigerator",
    "Washing machine": "Washing-Machine",
    "Dishwasher": "Dishwasher",
    "Oven": "Oven",
    "Gas stove": "Stove",
    "Toaster": "Toaster",
    "Coffeemaker": "Coffee-Machine",
    "Hair dryer": "Hair-Dryer",
    "Ceiling fan": "Ceiling-Fan",
    "Mechanical fan": "Floor-Fan",           # includes desk fans: spot-check
    "Torch": "Flashlight",
    "Loudspeaker": "Speaker",
    "Musical keyboard": "Electronic-Keyboard",
    "Lawn mower": "Rotary-Mower",
    "Street light": "Street-Lamp",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-yaml", default=r"D:\ewaste_final\data.yaml")
    ap.add_argument("--out", default=r"D:\extra_data\openimages")
    ap.add_argument("--cache", default=r"D:\extra_data\fiftyone_cache")
    ap.add_argument("--per-class", type=int, default=400,
                    help="max images to download per Open Images class")
    ap.add_argument("--splits", nargs="*", default=["validation", "test"],
                    help="Open Images splits to draw from. validation+test need only small "
                         "annotation files (~100 MB); adding train needs ~2 GB more")
    ap.add_argument("--only", nargs="*", default=None,
                    help="limit to these of YOUR class names, e.g. Hair-Dryer Camera")
    a = ap.parse_args()

    import os
    # keep every fiftyone file on D: (C: is full and its user folder is not writable)
    os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", r"D:\extra_data\fiftyone_zoo")
    os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", r"D:\extra_data\fiftyone_data")
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", r"D:\fiftyone_db")
    os.environ.setdefault("FIFTYONE_MODEL_ZOO_DIR", r"D:\extra_data\fiftyone_models")
    import fiftyone as fo
    import fiftyone.zoo as foz
    import fiftyone.utils.openimages as fouo
    from fiftyone import ViewField as F

    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} already exists. Rename or remove it first.")

    with open(a.classes_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]

    bad_targets = [v for v in MAPPING.values() if v not in names]
    if bad_targets:
        sys.exit(f"These target names are not in your data.yaml: {bad_targets}")

    fo.config.dataset_zoo_dir = os.environ["FIFTYONE_DATASET_ZOO_DIR"]
    fo.config.default_dataset_dir = os.environ["FIFTYONE_DEFAULT_DATASET_DIR"]
    print(f"fiftyone files go to {fo.config.dataset_zoo_dir}")
    available = set(fouo.get_classes(version="v7"))
    wanted = {k: v for k, v in MAPPING.items()
              if a.only is None or v in a.only}
    missing = [k for k in wanted if k not in available]
    for k in missing:
        print(f"SKIP: Open Images has no class named '{k}'")
    wanted = {k: v for k, v in wanted.items() if k in available}
    if not wanted:
        sys.exit("Nothing to download.")

    combined = fo.Dataset()
    for oi_name, mine in wanted.items():
        left = a.per_class
        for split in a.splits:
            if left <= 0:
                break
            print(f"\n=== {oi_name} -> {mine}: up to {left} images from {split}")
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[oi_name],
                only_matching=True,
                max_samples=left,
                shuffle=True,
                seed=0,
                drop_existing_dataset=True,
                dataset_name=f"oi-{split}-{oi_name.replace(' ', '-').lower()}",
            )
            left -= len(ds)
            combined.merge_samples(ds)      # same image twice -> merged once

    field = "ground_truth"
    view = combined.map_labels(field, wanted)
    view = view.filter_labels(field, F("label").is_in(list(wanted.values())), only_matches=True)
    view = view.filter_labels(field, F("IsGroupOf") != True, only_matches=True)  # noqa: E712

    tmp = out.parent / (out.name + "_export_tmp")
    view.export(
        export_dir=str(tmp),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=field,
        split="train",
        classes=names,                      # keeps YOUR 77 class IDs
    )

    # YOLOv5 export layout is images/train + labels/train; switch to train/images
    for sub in ("images", "labels"):
        (out / "train" / sub).mkdir(parents=True, exist_ok=True)
        for p in (tmp / sub / "train").iterdir():
            shutil.move(str(p), out / "train" / sub / p.name)
    shutil.rmtree(tmp, ignore_errors=True)

    with (out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out), "train": "train/images", "val": "train/images",
                        "names": dict(enumerate(names)), "nc": len(names)},
                       f, sort_keys=False, allow_unicode=True)

    boxes, imgs = Counter(), Counter()
    for lab in (out / "train" / "labels").glob("*.txt"):
        cls = [int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()]
        boxes.update(cls)
        imgs.update(set(cls))
    with (out / "download_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "images", "boxes"])
        for i, n in enumerate(names):
            if imgs[i]:
                w.writerow([i, n, imgs[i], boxes[i]])
                print(f"{n:<28} images={imgs[i]:>5} boxes={boxes[i]:>6}")
    n_img = len(list((out / "train" / "images").iterdir()))
    print(f"\nSaved {n_img} images to {out}")
    print("Next: open a few in the train folder and check the boxes before merging.")


if __name__ == "__main__":
    main()
