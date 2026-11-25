"""
E-Waste Sorting Inspector: local web app.

  Overview tab   live training progress (reads results.csv while YOLO trains),
                 earlier runs, test scores, dataset numbers, assignment checklist
  Inspect tab    upload a photo or use the webcam; the trained model detects items
                 and sorts them into the six EU WEEE recycling streams

Run (from the webapp folder, in your normal Python, not the fiftyone env):
    python server.py
    python server.py --model "D:\\ewaste_runs\\model_3\\weights\\best.pt" --device cpu
Then open http://localhost:8000

Uses only the Python standard library plus ultralytics and Pillow, which you
already have. Inference runs on the CPU by default so it never competes with
training on the GPU.
"""
import argparse
import base64
import csv
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                                  # D:\E-Waste Dataset.v1i.yolov8

# Training runs shown on the Overview tab: (label, folder, what it is)
RUNS = [
    ("Baseline", PROJECT / "runs/detect/runs/ewaste/yolo26n_baseline",
     "Original Roboflow splits (had leakage)"),
    ("Leak-free", PROJECT / "runs/detect/yolo26n_deduplicated-4",
     "Exact duplicates removed (ewaste_final)"),
    ("Final", Path(r"D:\ewaste_runs\final_v2"),
     "Grouped re-split, 66 classes (ewaste_v2)"),
    ("+ Bangladesh", Path(r"D:\ewaste_runs\final_v4_ft"),
     "Fine-tuned with an independent dataset's photos (ewaste_v4)"),
]
STATUS_FILE = HERE / "project_status.json"
STALL_MINUTES = 30
MODELS_FILE = HERE / "models.json"

_models = {}          # key -> loaded YOLO model (loaded on first use, then kept)
_model_path = None    # model given with --model (shown as "Custom" in the list)
_device = "cpu"


def model_list():
    """Every model in models.json whose weights exist, in order, plus --model if given."""
    entries = []
    if MODELS_FILE.exists():
        entries = json.loads(MODELS_FILE.read_text(encoding="utf-8")).get("models", [])
    out = []
    for m in entries:
        path = Path(m["path"].replace("{PROJECT}", str(PROJECT)))
        out.append({**m, "path": str(path), "available": path.exists()})
    if _model_path:
        out.append({"key": "custom", "name": f"Custom: {_model_path.parent.parent.name}", "path": str(_model_path),
                    "available": _model_path.exists(), "default": True, "note": "Given with --model"})
    avail = [m for m in out if m["available"]]
    default = next((m["key"] for m in reversed(avail) if m.get("default")), avail[-1]["key"] if avail else None)
    return out, default


def get_model(key=None):
    models, default = model_list()
    key = key or default
    m = next((m for m in models if m["key"] == key), None)
    if m is None or not m["available"]:
        raise FileNotFoundError(f"Model '{key}' has no weights file. Check its path in webapp/models.json.")
    if key not in _models:
        from ultralytics import YOLO
        _models[key] = YOLO(m["path"])
    return _models[key], m


def read_epochs_from_args(run_dir):
    args = run_dir / "args.yaml"
    if not args.exists():
        return None
    for line in args.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("epochs:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def run_status(label, run_dir, note):
    out = {"label": label, "folder": str(run_dir), "note": note, "exists": run_dir.exists(),
           "epochs_total": read_epochs_from_args(run_dir), "rows": [], "state": "not started"}
    res = run_dir / "results.csv"
    if not res.exists():
        if run_dir.exists():
            out["state"] = "starting"
        return out
    with res.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {k.strip(): v for k, v in r.items() if k}
            try:
                out["rows"].append({
                    "epoch": int(float(r["epoch"])),
                    "map50": float(r["metrics/mAP50(B)"]),
                    "map": float(r["metrics/mAP50-95(B)"]),
                    "time": float(r.get("time", 0) or 0),
                })
            except (KeyError, ValueError):
                continue
    mtime = res.stat().st_mtime
    out["updated"] = mtime
    out["minutes_since_update"] = round((time.time() - mtime) / 60, 1)
    done = (run_dir / "weights" / "best.pt").exists() and (run_dir / "results.png").exists()
    n, total = len(out["rows"]), out["epochs_total"]
    if done or (total and n >= total):
        out["state"] = "finished"
    elif out["minutes_since_update"] > STALL_MINUTES:
        out["state"] = "stalled"
    else:
        out["state"] = "training"
    return out


def status_payload():
    project = {}
    if STATUS_FILE.exists():
        project = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    models, default = model_list()
    return {
        "now": time.time(),
        "runs": [run_status(*r) for r in RUNS],
        "project": project,
        "models": models,
        "default_model": default,
        "device": _device,
    }


def predict(payload):
    data_url = payload["image"]
    raw = base64.b64decode(data_url.split(",", 1)[1])
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    conf = float(payload.get("conf", 0.25))
    model, info = get_model(payload.get("model"))
    t0 = time.perf_counter()
    res = model.predict(img, conf=conf, device=_device, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    names = res.names
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append({"name": names[int(b.cls)], "conf": round(float(b.conf), 3),
                     "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    dets.sort(key=lambda d: -d["conf"])
    return {"width": img.width, "height": img.height, "ms": round(ms, 1),
            "detections": dets, "model": info["key"], "model_name": info["name"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/fonts/") and self.path.endswith(".woff2"):
            f = HERE / "static" / "fonts" / Path(self.path).name
            if f.exists():
                self._send(200, f.read_bytes(), "font/woff2")
            else:
                self._send(404, {"error": "font not found"})
        elif self.path == "/api/status":
            try:
                self._send(200, status_payload())
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 25 * 1024 * 1024:
                self._send(413, {"error": "Image is larger than 25 MB."})
                return
            payload = json.loads(self.rfile.read(n))
            self._send(200, predict(payload))
        except FileNotFoundError as e:
            self._send(503, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"Detection failed: {e}"})


def main():
    global _model_path, _device
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="extra best.pt to list as Custom (models.json lists the rest)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or 0 for the GPU")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    _model_path = Path(a.model) if a.model else None
    _device = a.device
    models, default = model_list()
    print("Models: " + ", ".join(m["name"] + ("" if m["available"] else " (no weights yet)") for m in models))
    print(f"Default: {default}")
    print(f"Device: {_device}")
    print(f"Open    http://localhost:{a.port}   (Ctrl+C to stop)")
    class Server(ThreadingHTTPServer):
        # On Windows, address reuse lets a second copy bind the same port while the old copy
        # keeps answering, so the browser silently talks to the old server. Refuse instead.
        allow_reuse_address = False
    try:
        httpd = Server(("127.0.0.1", a.port), Handler)
    except OSError:
        sys.exit(f"Port {a.port} is already in use: an older copy of this server is still running. "
                 f"Close it (Ctrl+C in its window) or use --port {a.port + 1}.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
