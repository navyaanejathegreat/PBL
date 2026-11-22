"""Deployment wrapper: run a trained YOLO model on the test split and hand everything to the C++ app.

Writes (into --out):
  classes.csv      id,name,stream          (stream = WEEE bin for each class)
  gt.csv           image,cls,cx,cy,w,h     (normalised YOLO boxes)
  preds.csv        image,cls,conf,cx,cy,w,h
  timing.csv       images,total_ms,preprocess_ms,inference_ms,postprocess_ms

Then run:  bin\\qc_app --dir <out>

Usage (from the project folder):
  python cpp_core\\python\\export_predictions.py --weights D:\\ewaste_runs\\final_v2\\weights\\best.pt ^
         --data D:\\ewaste_v2\\data.yaml --out cpp_core\\qc_input\\yolo26n
"""
from pathlib import Path
import argparse
import csv
import time

import yaml

STREAM_OF = {
    "temp": "Air-Conditioner Refrigerator Freezer Cooling-Display Water-Dispenser",
    "screen": "CRT-Monitor CRT-TV Flat-Panel-Monitor Flat-Panel-TV Laptop Patient-Monitoring-System",
    "lamp": "Compact-Fluorescent-Lamps Straight-Tube-Fluorescent-Lamp LED-Bulb",
    "large": "Washing-Machine Dishwasher Oven Stove Range-Hood Boiler Photovoltaic-Panel Server Rotary-Mower "
             "Street-Lamp Ceiling-Fan Electric-Bicycle",
    "it": "Smartphone Bar-Phone Telephone-Set Router Network-Switch Computer-Keyboard Computer-Mouse Desktop-PC "
          "Printer Calculator HDD SSD USB-Flash-Drive PCB Power-Adapter TV-Remote-Control Headphone",
    "battery": "Battery",
}


def stream_of(name):
    for s, names in STREAM_OF.items():
        if name in names.split():
            return s
    return "small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=r"D:\ewaste_v2\data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001, help="keep low: the C++ app needs the full ranking for mAP")
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.data, encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    root = Path(cfg.get("path") or Path(a.data).parent)
    img_dir = root / cfg[a.split if a.split != "valid" else "val"]
    lab_dir = img_dir.parent / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    with open(out / "classes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "name", "stream"])
        for i, n in enumerate(names):
            w.writerow([i, n, stream_of(n)])

    with open(out / "gt.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "cls", "cx", "cy", "w", "h"])
        for i, p in enumerate(images):
            lab = lab_dir / (p.stem + ".txt")
            if lab.exists():
                for line in lab.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) == 5:
                        w.writerow([i] + parts)

    model = YOLO(a.weights)
    model.predict(str(images[0]), imgsz=a.imgsz, device=a.device, verbose=False)  # warm-up
    pre = inf = post = 0.0
    t0 = time.perf_counter()
    with open(out / "preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "cls", "conf", "cx", "cy", "w", "h"])
        for start in range(0, len(images), 32):
            batch = images[start:start + 32]
            for j, r in enumerate(model.predict([str(p) for p in batch], imgsz=a.imgsz, device=a.device,
                                                conf=a.conf, max_det=300, verbose=False)):
                pre += r.speed["preprocess"]; inf += r.speed["inference"]; post += r.speed["postprocess"]
                b = r.boxes
                for c, s, xywh in zip(b.cls.tolist(), b.conf.tolist(), b.xywhn.tolist()):
                    w.writerow([start + j, int(c), f"{s:.5f}"] + [f"{v:.6f}" for v in xywh])
    total = (time.perf_counter() - t0) * 1000
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["images", "total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"])
        w.writerow([len(images), f"{total:.1f}", f"{pre:.1f}", f"{inf:.1f}", f"{post:.1f}"])
    print(f"{len(images)} images -> {out}  ({len(images) / total * 1000:.1f} images/s end to end)")
    print(f"Now run:  cpp_core\\bin\\qc_app --dir {out}")


if __name__ == "__main__":
    main()
