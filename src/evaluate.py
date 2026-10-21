"""
Comprehensive Model Evaluation & Qualitative Visual Generation.
Computes Precision, Recall, and F1 scores across Train, Val, and Test splits.
Generates publication-quality visual detection galleries and PR/F1 curve figures.
"""

import os
import glob
import json
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
COLORS = [
    (230, 25, 75),   # Red
    (60, 180, 75),   # Green
    (255, 225, 25),  # Yellow
    (0, 130, 200),   # Blue
    (245, 130, 48),  # Orange
    (145, 30, 180),  # Purple
]


def compute_f1_and_metrics(model: YOLO, data_yaml: str, split: str = "test", imgsz: int = 224, conf: float = 0.28):
    """Evaluates model and computes Precision, Recall, F1 score, and mAP."""
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, conf=conf, verbose=False)

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    f1 = float(2 * p * r / max(p + r, 1e-6))
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Per class breakdown
    per_class = {}
    p_per_class = metrics.box.p if hasattr(metrics.box, "p") and len(metrics.box.p) > 0 else [0.0] * 6
    r_per_class = metrics.box.r if hasattr(metrics.box, "r") and len(metrics.box.r) > 0 else [0.0] * 6
    ap50_per_class = metrics.box.ap50 if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > 0 else [0.0] * 6

    for idx, cname in enumerate(CLASSES):
        c_p = float(p_per_class[idx]) if idx < len(p_per_class) else 0.0
        c_r = float(r_per_class[idx]) if idx < len(r_per_class) else 0.0
        c_f1 = float(2 * c_p * c_r / max(c_p + c_r, 1e-6))
        c_ap = float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0
        per_class[cname] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
            "ap50": round(c_ap, 4),
        }

    return {
        "split": split,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_class": per_class,
    }


def generate_qualitative_detection_grid(
    model: YOLO,
    test_img_dir: str = "data/processed/neu-det/images/test",
    output_path: str = "reports/qualitative_detections_grid.png",
):
    """
    Renders high-resolution qualitative detection examples across all 6 defect classes
    with bounding boxes, class labels, and confidence tags.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, cname in enumerate(CLASSES):
        # Find sample image starting with this class name
        candidates = sorted(glob.glob(os.path.join(test_img_dir, f"{cname}*.jpg")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(test_img_dir, f"{cname.replace('-', '_')}*.jpg")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))

        sample_path = candidates[0] if candidates else None
        if not sample_path:
            continue

        # Run inference
        results = model.predict(sample_path, conf=0.25, imgsz=224, verbose=False)
        rendered_bgr = results[0].plot(line_width=2, font_size=1)
        rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

        num_dets = len(results[0].boxes)
        ax = axes[idx]
        ax.imshow(rendered_rgb)
        ax.set_title(f"Class: {cname.upper()}\n({num_dets} defect{'s' if num_dets != 1 else ''} detected)", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("NEU-DET Automated Defect Localizations (Inference on Test Split)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Qualitative detection gallery saved to {output_path}")


def generate_lighting_shift_visual_comparison(
    model: YOLO,
    clean_dir: str = "data/processed/neu-det/images/test",
    shifted_dir: str = "data/processed/neu-det-shifted/images/test",
    output_path: str = "reports/lighting_shift_visual_comparison.png",
):
    """
    Renders side-by-side comparison of clean vs harsh lighting shifted steel coils with detections.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_stems = ["patches_36", "scratches_231", "inclusion_222", "pitted_surface_240"]

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))

    row = 0
    for stem in sample_stems:
        clean_img_path = os.path.join(clean_dir, f"{stem}.jpg")
        shifted_img_path = os.path.join(shifted_dir, f"{stem}.jpg")

        if not os.path.exists(clean_img_path) or not os.path.exists(shifted_img_path):
            # Fallback to whatever exists
            candidates = glob.glob(os.path.join(clean_dir, "*.jpg"))
            if not candidates:
                continue
            stem = os.path.basename(candidates[row]).replace(".jpg", "")
            clean_img_path = os.path.join(clean_dir, f"{stem}.jpg")
            shifted_img_path = os.path.join(shifted_dir, f"{stem}.jpg")

        res_clean = model.predict(clean_img_path, conf=0.25, imgsz=224, verbose=False)
        plot_clean = cv2.cvtColor(res_clean[0].plot(line_width=2), cv2.COLOR_BGR2RGB)

        res_shift = model.predict(shifted_img_path, conf=0.25, imgsz=224, verbose=False)
        plot_shift = cv2.cvtColor(res_shift[0].plot(line_width=2), cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(plot_clean)
        axes[row, 0].set_title(f"Clean Benchmark: {stem}", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(plot_shift)
        axes[row, 1].set_title(f"Simulated Plant Shift: {stem}", fontsize=10, fontweight="bold")
        axes[row, 1].axis("off")

        row += 1
        if row >= 4:
            break

    plt.suptitle("Clean Inspection vs Harsh Industrial Lighting Shift (Model Detections)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Lighting shift visual comparison saved to {output_path}")


def run_full_evaluation():
    """Runs complete evaluation across all splits and generates report assets."""
    champion_path = "models/champion_best.pt"
    if not os.path.exists(champion_path):
        candidates = glob.glob("runs/detect/runs/train/*/weights/best.pt")
        champion_path = candidates[0] if candidates else "yolov8n.pt"

    print(f"Loading champion model from {champion_path}...")
    model = YOLO(champion_path)

    # 1. Evaluate Splits
    print("\n--- Evaluating Train Split ---")
    train_metrics = compute_f1_and_metrics(model, "data/processed/neu-det/dataset.yaml", split="train")

    print("\n--- Evaluating Validation Split ---")
    val_metrics = compute_f1_and_metrics(model, "data/processed/neu-det/dataset.yaml", split="val")

    print("\n--- Evaluating Clean Test Split ---")
    test_clean_metrics = compute_f1_and_metrics(model, "data/processed/neu-det/dataset.yaml", split="test")

    print("\n--- Evaluating Lighting-Shifted Test Split ---")
    test_shifted_metrics = compute_f1_and_metrics(model, "data/processed/neu-det-shifted/dataset_shifted.yaml", split="val")

    evaluation_summary = {
        "model": champion_path,
        "train": train_metrics,
        "val": val_metrics,
        "test_clean": test_clean_metrics,
        "test_shifted": test_shifted_metrics,
    }

    with open("reports/evaluation_splits_f1.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    # 2. Copy Ultralytics Curves if available
    train_dir = "runs/detect/runs/train/10_champion_yolov8n_production"
    if os.path.exists(train_dir):
        for fname in ["BoxF1_curve.png", "BoxPR_curve.png", "results.png"]:
            src = os.path.join(train_dir, fname)
            dst = os.path.join("reports", fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {fname} to reports/")

    # 3. Generate Visual Galleries
    generate_qualitative_detection_grid(model)
    generate_lighting_shift_visual_comparison(model)

    print("\n=======================================================")
    print("  EVALUATION & F1 SCORE REPORT")
    print("=======================================================")
    print(f"Train Split:        Precision={train_metrics['precision']:.3f} | Recall={train_metrics['recall']:.3f} | F1={train_metrics['f1_score']:.3f} | mAP50={train_metrics['mAP50']:.3f}")
    print(f"Val Split:          Precision={val_metrics['precision']:.3f} | Recall={val_metrics['recall']:.3f} | F1={val_metrics['f1_score']:.3f} | mAP50={val_metrics['mAP50']:.3f}")
    print(f"Test Clean Split:   Precision={test_clean_metrics['precision']:.3f} | Recall={test_clean_metrics['recall']:.3f} | F1={test_clean_metrics['f1_score']:.3f} | mAP50={test_clean_metrics['mAP50']:.3f}")
    print(f"Test Shifted Split: Precision={test_shifted_metrics['precision']:.3f} | Recall={test_shifted_metrics['recall']:.3f} | F1={test_shifted_metrics['f1_score']:.3f} | mAP50={test_shifted_metrics['mAP50']:.3f}")
    print("=======================================================\n")
    return evaluation_summary


if __name__ == "__main__":
    run_full_evaluation()
