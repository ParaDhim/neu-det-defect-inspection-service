"""Train Runs 01, 06, 07, 10, 11 with seeds 0/1/2 for mean +/- std reporting.

Weights: models/seeds/<run>_s<seed>.pt. Configs are read from src/train.py
(EXPERIMENT_CATALOG) and src/train_v2.py (EXPERIMENTS); Run 01 uses 200px.
"""
import shutil
from pathlib import Path

from ultralytics import YOLO, settings

settings.update({"mlflow": False})


def literal(path, name):
    t = Path(path).read_text()
    t = t[t.index(f"{name} = ["):]
    ns = {}
    exec(t[: t.index("\n]\n") + 3], ns)
    return ns[name]


DEFAULTS = dict(batch=32, lr0=0.01, optimizer="auto", hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                scale=0.5, translate=0.1, degrees=0.0, fliplr=0.5, flipud=0.5, mosaic=1.0, erasing=0.0)
cat = {c["run_name"][:2]: c for c in literal("src/train.py", "EXPERIMENT_CATALOG")}
v2 = {c["run_name"][:2]: c for c in literal("src/train_v2.py", "EXPERIMENTS")}
cfgs = {"01": cat["01"], "06": cat["06"], "07": cat["07"], "10": cat["10"], "11": v2["11"]}
cfgs["01"]["imgsz"] = 200
OUT = Path("models/seeds")
OUT.mkdir(parents=True, exist_ok=True)

for seed in (0, 1, 2):
    for key, cfg in cfgs.items():
        name = f"{key}_s{seed}"
        if (OUT / f"{name}.pt").exists():
            continue
        skip = ("run_name", "base_model", "model", "tags", "label")
        args = {**DEFAULTS, **{k: v for k, v in cfg.items() if k not in skip}}
        model = YOLO(cfg.get("base_model") or cfg["model"])
        model.train(data="data/processed/neu-det/dataset.yaml", device=0, project="runs/seeds", name=name,
                    seed=seed, deterministic=True, plots=False, workers=4, exist_ok=True, verbose=False, **args)
        shutil.copy2(Path(model.trainer.save_dir) / "weights" / "best.pt", OUT / f"{name}.pt")
        print("DONE", name, flush=True)
