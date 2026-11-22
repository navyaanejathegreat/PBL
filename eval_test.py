"""
Evaluate a trained model on the TEST split and save per-class results.

    python eval_test.py <weights.pt> <name> [--device cpu|0] [--data path\\to\\data.yaml]

Examples:
    python eval_test.py "runs\\detect\\runs\\ewaste\\yolo26n_baseline\\weights\\best.pt" old_baseline --device cpu
    python eval_test.py "runs\\detect\\yolo26n_deduplicated-4\\weights\\best.pt" clean_baseline --device 0

Writes per_class_<name>.csv next to this script and prints mAP50 / mAP50-95.
"""
import argparse
import csv

from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=r"D:\ewaste_final\data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    m = YOLO(a.weights).val(data=a.data, split="test", device=a.device, batch=8,
                            imgsz=a.imgsz, plots=True, name=f"test_{a.name}")

    with open(f"per_class_{a.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "ap50_95"])
        for c in m.box.ap_class_index:
            w.writerow([m.names[int(c)], round(float(m.box.maps[int(c)]), 4)])

    print(f"\n{a.name}: mAP50 = {m.box.map50:.3f}   mAP50-95 = {m.box.map:.3f}")
    print(f"Per-class results saved to per_class_{a.name}.csv")


# Required on Windows: data-loader workers re-import this file.
if __name__ == "__main__":
    main()
