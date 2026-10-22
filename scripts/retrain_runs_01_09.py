"""Retrain DOE runs 01-09 (original weights were not archived) with a fixed seed.

Weights land in models/runs/<run_name>.pt. No evaluation happens here; every
run is scored later by scripts/eval_all_runs.py under a single protocol.
Run 01 uses 200px to match the archived row; Run 10/11 weights are kept as-is.
"""
import re
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO, settings

sys.path.insert(0, "src")
settings.update({"mlflow": False})

src = Path("src/train.py").read_text()
ns = {}
catalog_src = src[src.index("EXPERIMENT_CATALOG = ["):]
exec(catalog_src[: catalog_src.index("\n]\n") + 3], ns)

DEFAULTS = dict(batch=32, lr0=0.01, optimizer="auto", hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                scale=0.5, translate=0.1, degrees=0.0, fliplr=0.5, flipud=0.5, mosaic=1.0, erasing=0.0)
OUT = Path("models/runs")
OUT.mkdir(parents=True, exist_ok=True)

for cfg in ns["EXPERIMENT_CATALOG"]:
    name = cfg["run_name"]
    if name.startswith("10_"):
        continue
    if name.startswith("01_"):
        name, cfg["imgsz"] = "01_baseline_yolov8n_imgsz200", 200
    if (OUT / f"{name}.pt").exists():
        continue
    args = {**DEFAULTS, **{k: v for k, v in cfg.items() if k not in ("run_name", "base_model", "tags")}}
    model = YOLO(cfg["base_model"])
    model.train(data="data/processed/neu-det/dataset.yaml", device=0, project="runs/retrain", name=name,
                seed=0, deterministic=True, plots=False, workers=4, exist_ok=True, verbose=False, **args)
    shutil.copy2(Path(model.trainer.save_dir) / "weights" / "best.pt", OUT / f"{name}.pt")
    print("DONE", name, flush=True)
