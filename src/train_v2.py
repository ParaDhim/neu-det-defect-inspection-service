"""
Full-power retraining script for NEU-DET defect inspection.

Strategy:
  - YOLOv8s (11.1M params) at 640×640 — the main fix.
    The previous runs used 200-224px which drops crazing/scale features.
  - 100 epochs with cosine LR, AdamW, CopyPaste + heavy augmentation.
  - patience=50 (default) — safe for noisy heavy-aug training curves.
  - Apple M2 MPS GPU acceleration.
  - Logged to MLflow as a new experiment "v2_highres_training".
  - Final champion saved to models/champion_v2_best.pt

  NOTE: YOLOv8m (run 12) was removed — on MPS/small dataset (NEU-DET ~1800
  images) YOLOv8s at 640px matches YOLOv8m quality at 2-3x less compute.

Why this fixes the poor scores:
  - Crazing: fine fissures (< 2px at 200px) become detectable at 640px.
  - Rolled-in scale: diffuse edges become resolvable at higher res.
  - Overall mAP50 expected to jump from 50% → 65-80%.
"""

import os
import shutil
import time
import json
import mlflow
from pathlib import Path
from ultralytics import YOLO, settings

# ── disable Ultralytics' internal MLflow to avoid param conflicts ──
settings.update({"mlflow": False})

# ── paths ──
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "v2_highres_training"
DATA_YAML = "data/processed/neu-det/dataset.yaml"
SHIFTED_YAML = "data/processed/neu-det-shifted/dataset_shifted.yaml"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

EXPERIMENTS = [
    # ── Run 1: YOLOv8s @ 640, heavy aug — the core fix ──
    {
        "run_name": "11_yolov8s_640_heavy_aug",
        "model": "yolov8s.pt",
        "imgsz": 640,
        "epochs": 100,
        "patience": 50,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "erasing": 0.4,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "close_mosaic": 15,
        "label": "YOLOv8s 640px Heavy Aug",
    },
    # ── Run 2: YOLOv8s @ 640 with stronger HSV (pushes crazing/scale) ──
    {
        "run_name": "13_yolov8s_640_max_aug",
        "model": "yolov8s.pt",
        "imgsz": 640,
        "epochs": 100,
        "patience": 50,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.01,
        "cos_lr": True,
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.6,
        "degrees": 10.0,
        "translate": 0.15,
        "scale": 0.6,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "erasing": 0.5,
        "mixup": 0.2,
        "copy_paste": 0.4,
        "close_mosaic": 20,
        "label": "YOLOv8s 640px Max Aug",
    },
]


def train_and_log(cfg: dict) -> dict:
    """Train one experiment, log to MLflow, return metrics dict."""
    run_name = cfg["run_name"]
    print(f"\n{'='*60}")
    print(f"  Starting: {run_name} — {cfg['label']}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": cfg["model"],
                "imgsz": cfg["imgsz"],
                "epochs": cfg["epochs"],
                "patience": cfg.get("patience", 50),
                "batch": cfg["batch"],
                "optimizer": cfg["optimizer"],
                "lr0": cfg["lr0"],
                "lrf": cfg["lrf"],
                "cos_lr": cfg["cos_lr"],
                "hsv_v": cfg["hsv_v"],
                "mosaic": cfg["mosaic"],
                "copy_paste": cfg.get("copy_paste", 0.0),
                "mixup": cfg.get("mixup", 0.0),
                "erasing": cfg.get("erasing", 0.0),
            }
        )

        model = YOLO(cfg["model"])

        t0 = time.time()
        model.train(
            data=DATA_YAML,
            epochs=cfg["epochs"],
            patience=cfg.get("patience", 50),
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            optimizer=cfg["optimizer"],
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            cos_lr=cfg["cos_lr"],
            hsv_h=cfg.get("hsv_h", 0.015),
            hsv_s=cfg.get("hsv_s", 0.7),
            hsv_v=cfg["hsv_v"],
            degrees=cfg.get("degrees", 0.0),
            translate=cfg.get("translate", 0.1),
            scale=cfg.get("scale", 0.5),
            fliplr=cfg.get("fliplr", 0.5),
            mosaic=cfg["mosaic"],
            erasing=cfg.get("erasing", 0.0),
            mixup=cfg.get("mixup", 0.0),
            copy_paste=cfg.get("copy_paste", 0.0),
            close_mosaic=cfg.get("close_mosaic", 10),
            device="mps",
            project="runs/train",
            name=run_name,
            exist_ok=True,
            verbose=False,
            plots=True,
        )
        train_time = time.time() - t0

        # ── Evaluate on clean test set ──
        clean_m = model.val(data=DATA_YAML, split="test", imgsz=cfg["imgsz"], conf=0.25, verbose=False)
        c_map50 = float(clean_m.box.map50)
        c_map5095 = float(clean_m.box.map)
        c_p = float(clean_m.box.mp)
        c_r = float(clean_m.box.mr)
        c_f1 = 2 * c_p * c_r / max(c_p + c_r, 1e-6)

        # Per-class clean
        per_class_clean = {}
        for idx, cname in enumerate(CLASSES):
            pp = float(clean_m.box.p[idx]) if idx < len(clean_m.box.p) else 0.0
            rr = float(clean_m.box.r[idx]) if idx < len(clean_m.box.r) else 0.0
            ff = 2 * pp * rr / max(pp + rr, 1e-6)
            ap = float(clean_m.box.ap50[idx]) if idx < len(clean_m.box.ap50) else 0.0
            per_class_clean[cname] = {"precision": round(pp, 4), "recall": round(rr, 4), "f1": round(ff, 4), "ap50": round(ap, 4)}
            mlflow.log_metric(f"clean_{cname}_ap50", ap)
            mlflow.log_metric(f"clean_{cname}_f1", ff)

        # ── Evaluate on shifted test set ──
        s_map50 = s_map5095 = s_f1 = 0.0
        per_class_shifted = {}
        shifted_yaml = Path(SHIFTED_YAML)
        if shifted_yaml.exists():
            shifted_m = model.val(data=str(shifted_yaml), split="test", imgsz=cfg["imgsz"], conf=0.25, verbose=False)
            s_map50 = float(shifted_m.box.map50)
            s_map5095 = float(shifted_m.box.map)
            s_p = float(shifted_m.box.mp)
            s_r = float(shifted_m.box.mr)
            s_f1 = 2 * s_p * s_r / max(s_p + s_r, 1e-6)
            for idx, cname in enumerate(CLASSES):
                pp = float(shifted_m.box.p[idx]) if idx < len(shifted_m.box.p) else 0.0
                rr = float(shifted_m.box.r[idx]) if idx < len(shifted_m.box.r) else 0.0
                ff = 2 * pp * rr / max(pp + rr, 1e-6)
                ap = float(shifted_m.box.ap50[idx]) if idx < len(shifted_m.box.ap50) else 0.0
                per_class_shifted[cname] = {"precision": round(pp, 4), "recall": round(rr, 4), "f1": round(ff, 4), "ap50": round(ap, 4)}
                mlflow.log_metric(f"shifted_{cname}_ap50", ap)
                mlflow.log_metric(f"shifted_{cname}_f1", ff)

        drop_abs = c_map50 - s_map50
        drop_rel = (drop_abs / c_map50 * 100) if c_map50 > 0 else 0.0

        mlflow.log_metrics(
            {
                "clean_test_mAP50": c_map50,
                "clean_test_mAP50_95": c_map5095,
                "clean_test_precision": c_p,
                "clean_test_recall": c_r,
                "clean_test_f1": c_f1,
                "shifted_test_mAP50": s_map50,
                "shifted_test_mAP50_95": s_map5095,
                "shifted_test_f1": s_f1,
                "mAP50_drop_abs": drop_abs,
                "mAP50_drop_rel_pct": drop_rel,
                "train_time_sec": train_time,
            }
        )

        # Save best weights
        best_pt = Path(f"runs/train/{run_name}/weights/best.pt")
        if best_pt.exists():
            dest = Path(OUTPUT_DIR) / f"{run_name}_best.pt"
            shutil.copy(best_pt, dest)
            mlflow.log_artifact(str(dest), artifact_path="weights")

        # Copy plots to reports/
        for plot in ["confusion_matrix.png", "results.png", "PR_curve.png", "F1_curve.png"]:
            p = Path(f"runs/train/{run_name}/{plot}")
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="plots")
                dest_report = Path("reports") / f"v2_{run_name}_{plot}"
                shutil.copy(p, dest_report)

        result = {
            "run_name": run_name,
            "label": cfg["label"],
            "model": cfg["model"],
            "imgsz": cfg["imgsz"],
            "clean_mAP50": round(c_map50, 4),
            "clean_mAP50_95": round(c_map5095, 4),
            "clean_f1": round(c_f1, 4),
            "shifted_mAP50": round(s_map50, 4),
            "shifted_mAP50_95": round(s_map5095, 4),
            "shifted_f1": round(s_f1, 4),
            "drop_abs": round(drop_abs, 4),
            "drop_rel_pct": round(drop_rel, 2),
            "train_time_sec": round(train_time, 1),
            "per_class_clean": per_class_clean,
            "per_class_shifted": per_class_shifted,
        }
        print(
            f"\n  RESULT → clean_mAP50={c_map50:.3f} | shifted_mAP50={s_map50:.3f}"
            f" | drop={drop_rel:.1f}% | clean_F1={c_f1:.3f}"
        )
        return result


def main():
    all_results = []
    best_map = 0.0
    best_run = None

    for cfg in EXPERIMENTS:
        result = train_and_log(cfg)
        all_results.append(result)
        if result["clean_mAP50"] > best_map:
            best_map = result["clean_mAP50"]
            best_run = result["run_name"]

    # Save summary
    summary_path = Path("reports/v2_experiments_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Summary → {summary_path}")

    # Promote overall champion
    if best_run:
        src = Path(OUTPUT_DIR) / f"{best_run}_best.pt"
        if src.exists():
            shutil.copy(src, Path(OUTPUT_DIR) / "champion_v2_best.pt")
            print(f"✓ New champion: {best_run} → models/champion_v2_best.pt  (mAP50={best_map:.4f})")

    print("\n\n=== FINAL LEADERBOARD ===")
    print(f"{'Run':<45} {'Clean mAP50':>12} {'Shifted mAP50':>14} {'Drop%':>7} {'Clean F1':>9}")
    print("-" * 92)
    for r in all_results:
        star = " ★" if r["run_name"] == best_run else ""
        print(
            f"{r['run_name']:<45} {r['clean_mAP50']:>12.4f} {r['shifted_mAP50']:>14.4f}"
            f" {r['drop_rel_pct']:>6.1f}% {r['clean_f1']:>9.4f}{star}"
        )


if __name__ == "__main__":
    main()
