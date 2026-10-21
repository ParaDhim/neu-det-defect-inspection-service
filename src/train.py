"""
Training and Experiment Tracking Pipeline for Steel Surface Defect Detection using YOLOv8 & MLflow.
Executes systematic research experiments across model capacity, image resolution, augmentation regimes,
and loss weighting. Evaluates models on both clean and lighting-shifted test benchmarks.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import mlflow
from ultralytics import YOLO, settings

# Disable Ultralytics built-in mlflow integration so our explicit tracker has full control
settings.update({"mlflow": False})

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def setup_mlflow(experiment_name: str = "neu-det-defect-inspection", tracking_uri: str = "sqlite:///mlflow.db"):
    """Initializes MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "steel-surface-defect-inspection",
                "framework": "ultralytics-yolov8",
                "hardware": "cuda-gpu" if torch.cuda.is_available() else ("apple-silicon-mps" if torch.backends.mps.is_available() else "cpu"),
            },
        )
    else:
        exp_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def evaluate_on_benchmark(model: YOLO, data_yaml: str, split: str = "test", imgsz: int = 224, conf: float = 0.25):
    """
    Evaluates a trained model on a specific data YAML and split, extracting overall and per-class AP.
    """
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, conf=conf, verbose=False)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": {},
    }

    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None and len(metrics.box.ap50) > 0:
        ap50_vals = metrics.box.ap50
    else:
        ap50_vals = [0.0] * len(CLASSES)

    maps_vals = metrics.box.maps if hasattr(metrics.box, "maps") and metrics.box.maps is not None else [0.0] * len(CLASSES)

    for idx, cname in enumerate(CLASSES):
        p_c_50 = float(ap50_vals[idx]) if idx < len(ap50_vals) else 0.0
        p_c_map = float(maps_vals[idx]) if idx < len(maps_vals) else 0.0
        results["per_class"][cname] = {
            "AP50": p_c_50,
            "AP50_95": p_c_map,
        }

    return results, metrics


def plot_and_save_confusion_matrix(metrics, output_path: str = "reports/confusion_matrix.png"):
    """Extracts confusion matrix from Ultralytics metrics and plots a publication-ready heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        matrix = metrics.confusion_matrix.matrix
        class_labels = CLASSES + ["background"]
        df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

        plt.figure(figsize=(9, 7))
        sns.heatmap(df_cm, annot=True, fmt=".0f", cmap="Blues", cbar=True, linewidths=0.5)
        plt.title("NEU-DET Confusion Matrix (Test Split)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Warning: could not plot confusion matrix directly: {e}")
        return None


def run_experiment(
    run_name: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 10,
    imgsz: int = 224,
    batch: int = 32,
    lr0: float = 0.01,
    optimizer: str = "auto",
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    scale: float = 0.5,
    translate: float = 0.1,
    degrees: float = 0.0,
    fliplr: float = 0.5,
    flipud: float = 0.5,
    mosaic: float = 1.0,
    erasing: float = 0.0,
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    clean_data_yaml: str = "data/processed/neu-det/dataset.yaml",
    shifted_data_yaml: str = "data/processed/neu-det-shifted/dataset_shifted.yaml",
    tags: dict = None,
):
    """
    Executes a single end-to-end training experiment and logs all parameters, metrics,
    and artifacts to MLflow.
    """
    setup_mlflow()
    print(f"\n=======================================================")
    print(f"Starting Experiment: {run_name}")
    print(f"Model: {base_model} | Epochs: {epochs} | Imgsz: {imgsz} | Device: {device}")
    print(f"=======================================================")

    with mlflow.start_run(run_name=run_name):
        params = {
            "model_architecture": base_model,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "optimizer": optimizer,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "scale": scale,
            "translate": translate,
            "degrees": degrees,
            "fliplr": fliplr,
            "flipud": flipud,
            "mosaic": mosaic,
            "erasing": erasing,
            "device": device,
        }
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        model = YOLO(base_model)

        project_dir = "runs/train"
        results = model.train(
            data=clean_data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            optimizer=optimizer,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            scale=scale,
            translate=translate,
            degrees=degrees,
            fliplr=fliplr,
            flipud=flipud,
            mosaic=mosaic,
            erasing=erasing,
            device=device,
            project=project_dir,
            name=run_name,
            plots=True,
            save=True,
            verbose=False,
            workers=2,
            exist_ok=True,
        )

        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = save_dir / "weights" / "last.pt"

        best_model = YOLO(str(best_pt))

        # Evaluate on Clean Test Split
        print("\nEvaluating on Clean Test Set...")
        clean_res, clean_metrics = evaluate_on_benchmark(best_model, clean_data_yaml, split="test", imgsz=imgsz)

        # Evaluate on Shifted Lighting Test Split
        print("Evaluating on Lighting-Shifted Test Set...")
        shifted_res, shifted_metrics = evaluate_on_benchmark(best_model, shifted_data_yaml, split="val", imgsz=imgsz)

        mAP50_clean = clean_res["mAP50"]
        mAP50_shifted = shifted_res["mAP50"]
        mAP50_drop = mAP50_clean - mAP50_shifted
        rel_drop_pct = (mAP50_drop / max(mAP50_clean, 1e-4)) * 100.0

        print(f"-> Clean Test mAP50: {mAP50_clean:.4f} | Shifted Test mAP50: {mAP50_shifted:.4f}")
        print(f"-> Lighting Shift Drop: -{mAP50_drop:.4f} (-{rel_drop_pct:.1f}%)")

        # Log Primary Metrics to MLflow
        mlflow.log_metrics({
            "clean_test_mAP50": mAP50_clean,
            "clean_test_mAP50_95": clean_res["mAP50_95"],
            "clean_test_precision": clean_res["precision"],
            "clean_test_recall": clean_res["recall"],
            "shifted_test_mAP50": mAP50_shifted,
            "shifted_test_mAP50_95": shifted_res["mAP50_95"],
            "shifted_test_precision": shifted_res["precision"],
            "shifted_test_recall": shifted_res["recall"],
            "lighting_shift_mAP50_drop": mAP50_drop,
            "lighting_shift_rel_drop_pct": rel_drop_pct,
        })

        # Log Per-Class Clean AP
        for cname, vals in clean_res["per_class"].items():
            mlflow.log_metric(f"clean_AP50_{cname}", vals["AP50"])
            mlflow.log_metric(f"clean_AP50_95_{cname}", vals["AP50_95"])

        # Log Per-Class Shifted AP
        for cname, vals in shifted_res["per_class"].items():
            mlflow.log_metric(f"shifted_AP50_{cname}", vals["AP50"])
            mlflow.log_metric(f"shifted_AP50_95_{cname}", vals["AP50_95"])

        # Save and log confusion matrix
        cm_path = f"reports/cm_{run_name}.png"
        plot_and_save_confusion_matrix(clean_metrics, output_path=cm_path)
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path)

        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")

        for plot_file in save_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_file), artifact_path="ultralytics_plots")

        summary = {
            "run_name": run_name,
            "best_weights": str(best_pt),
            "clean_mAP50": mAP50_clean,
            "clean_mAP50_95": clean_res["mAP50_95"],
            "shifted_mAP50": mAP50_shifted,
            "shifted_mAP50_95": shifted_res["mAP50_95"],
            "mAP50_drop": mAP50_drop,
            "rel_drop_pct": rel_drop_pct,
            "clean_per_class": clean_res["per_class"],
            "shifted_per_class": shifted_res["per_class"],
        }

        # Append to master json report
        update_experiments_report(summary)
        return summary


EXPERIMENT_CATALOG = [
    {
        "run_name": "01_baseline_yolov8n_imgsz224",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "scale": 0.0,
        "mosaic": 0.0,
        "tags": {"tier": "baseline", "type": "clean_no_aug"},
    },
    {
        "run_name": "02_baseline_yolov8n_imgsz320",
        "base_model": "yolov8n.pt",
        "imgsz": 320,
        "epochs": 10,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "scale": 0.0,
        "mosaic": 0.0,
        "tags": {"tier": "baseline", "type": "clean_highres"},
    },
    {
        "run_name": "03_robust_yolov8n_light_aug",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 5.0,
        "scale": 0.2,
        "mosaic": 0.5,
        "tags": {"tier": "robust", "aug_strength": "light"},
    },
    {
        "run_name": "04_robust_yolov8n_heavy_aug",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "hsv_h": 0.03,
        "hsv_s": 0.7,
        "hsv_v": 0.6,
        "degrees": 15.0,
        "scale": 0.5,
        "mosaic": 1.0,
        "erasing": 0.2,
        "tags": {"tier": "robust", "aug_strength": "heavy"},
    },
    {
        "run_name": "05_robust_yolov8s_heavy_aug",
        "base_model": "yolov8s.pt",
        "imgsz": 224,
        "epochs": 10,
        "hsv_h": 0.03,
        "hsv_s": 0.7,
        "hsv_v": 0.6,
        "degrees": 15.0,
        "scale": 0.5,
        "mosaic": 1.0,
        "tags": {"tier": "architecture_scale", "size": "small"},
    },
    {
        "run_name": "06_robust_yolov8n_adamw_cosine",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.5,
        "mosaic": 0.8,
        "tags": {"tier": "optimizer", "opt": "AdamW"},
    },
    {
        "run_name": "07_robust_yolov8n_sgd_momentum",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "optimizer": "SGD",
        "lr0": 0.01,
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.5,
        "mosaic": 0.8,
        "tags": {"tier": "optimizer", "opt": "SGD"},
    },
    {
        "run_name": "08_robust_yolov8n_geom_heavy",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 10,
        "degrees": 30.0,
        "translate": 0.2,
        "scale": 0.6,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "tags": {"tier": "geometric_robust"},
    },
    {
        "run_name": "09_robust_yolov8n_imgsz320_heavy",
        "base_model": "yolov8n.pt",
        "imgsz": 320,
        "epochs": 10,
        "hsv_h": 0.03,
        "hsv_s": 0.7,
        "hsv_v": 0.6,
        "degrees": 15.0,
        "scale": 0.5,
        "mosaic": 1.0,
        "tags": {"tier": "robust", "resolution": "320"},
    },
    {
        "run_name": "10_champion_yolov8n_production",
        "base_model": "yolov8n.pt",
        "imgsz": 224,
        "epochs": 12,
        "optimizer": "AdamW",
        "lr0": 0.003,
        "hsv_h": 0.025,
        "hsv_s": 0.65,
        "hsv_v": 0.55,
        "degrees": 15.0,
        "scale": 0.5,
        "mosaic": 1.0,
        "erasing": 0.15,
        "tags": {"tier": "champion", "deployment_target": "edge_onnx"},
    },
]


def update_experiments_report(summary: dict, path: str = "reports/experiments_summary.json"):
    """Loads existing summaries, appends or updates the new summary, and re-writes JSON and MD."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summaries = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                summaries = json.load(f)
        except Exception:
            summaries = []

    # Update or append
    updated = False
    for i, s in enumerate(summaries):
        if s.get("run_name") == summary.get("run_name"):
            summaries[i] = summary
            updated = True
            break
    if not updated:
        summaries.append(summary)

    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Regenerate markdown
    md_lines = [
        "### NEU-DET MLflow Experiment Tracking Suite Results",
        "",
        "| Run ID | Model Architecture | Resolution | Clean Test mAP50 | Shifted Test mAP50 | mAP50 Drop | Rel Drop (%) | Shifted mAP50-95 |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in summaries:
        r_name = s["run_name"]
        cl_50 = f"{s['clean_mAP50'] * 100:.1f}%"
        sh_50 = f"{s['shifted_mAP50'] * 100:.1f}%"
        drop = f"{s['mAP50_drop'] * 100:.1f}%"
        rel_drop = f"{s['rel_drop_pct']:.1f}%"
        sh_map = f"{s['shifted_mAP50_95'] * 100:.1f}%"
        arch = "YOLOv8s" if "yolov8s" in r_name else "YOLOv8n"
        res = "320" if "320" in r_name else "224"
        md_lines.append(f"| `{r_name}` | {arch} | {res}x{res} | {cl_50} | {sh_50} | {drop} | {rel_drop} | {sh_map} |")

    md_lines.append("")
    with open("reports/experiments_summary.md", "w") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEU-DET models with MLflow")
    parser.add_argument("--run-index", type=int, default=None, help="Run specific experiment index (0-9)")
    parser.add_argument("--run-range", type=str, default=None, help="Run range of experiment indices, e.g. 3,6")
    args = parser.parse_args()

    if args.run_index is not None:
        cfg = EXPERIMENT_CATALOG[args.run_index]
        run_experiment(**cfg)
    elif args.run_range:
        start_idx, end_idx = map(int, args.run_range.split(","))
        for idx in range(start_idx, end_idx):
            cfg = EXPERIMENT_CATALOG[idx]
            run_experiment(**cfg)
    else:
        for cfg in EXPERIMENT_CATALOG:
            run_experiment(**cfg)
