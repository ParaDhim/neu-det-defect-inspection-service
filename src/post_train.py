"""
Post-training pipeline: runs automatically after train_v2.py completes.
  1. Exports champion_v2_best.pt → ONNX FP32 + INT8
  2. Benchmarks INT8 on CPU (200 runs)
  3. Runs full F1 evaluation on Train / Val / Test-Clean / Test-Shifted
  4. Regenerates qualitative detection gallery
  5. Regenerates F1 and PR curve images
  6. Updates README.md results tables with real numbers
  7. Git commits and pushes everything
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────
CHAMPION = Path("models/champion_v2_best.pt")
DATA_YAML = "data/processed/neu-det/dataset.yaml"
SHIFTED_YAML = "data/processed/neu-det-shifted/dataset_shifted.yaml"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

if not CHAMPION.exists():
    # Fallback to old champion if v2 not yet trained
    CHAMPION = Path("models/champion_best.pt")
    print(f"[WARN] champion_v2_best.pt not found — using {CHAMPION}")


# ─────────────────────────────────────────────────────────────────────
# 1. ONNX export + INT8 quantization
# ─────────────────────────────────────────────────────────────────────
def export_onnx():
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = onnx_dir / "model_v2_fp32.onnx"
    int8_path = onnx_dir / "model_v2_int8.onnx"

    print("\n[1/5] Exporting to ONNX FP32...")
    model = YOLO(str(CHAMPION))
    exported = model.export(format="onnx", opset=17, simplify=True, dynamic=False, imgsz=640)
    # Ultralytics saves next to the .pt
    exported_path = Path(str(CHAMPION).replace(".pt", ".onnx"))
    if exported_path.exists():
        shutil.copy(exported_path, fp32_path)
        exported_path.unlink()
    print(f"   FP32 saved → {fp32_path}  ({fp32_path.stat().st_size / 1e6:.2f} MB)")

    print("[1/5] Quantizing → INT8...")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"   INT8 saved → {int8_path}  ({int8_path.stat().st_size / 1e6:.2f} MB)")

    # ── CPU benchmark ──
    print("[1/5] Benchmarking INT8 on CPU (200 iterations, 50 warmups)...")
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

    for _ in range(50):  # warmup
        sess.run(None, {inp_name: dummy})

    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32_lats = []
    for _ in range(50):
        fp32_sess.run(None, {inp_name: dummy})
    for _ in range(200):
        t0 = time.perf_counter()
        fp32_sess.run(None, {inp_name: dummy})
        fp32_lats.append((time.perf_counter() - t0) * 1000)

    bm = {
        "fp32": {
            "size_mb": round(fp32_path.stat().st_size / 1e6, 2),
            "p50_ms": round(float(np.percentile(fp32_lats, 50)), 2),
            "p90_ms": round(float(np.percentile(fp32_lats, 90)), 2),
            "p95_ms": round(float(np.percentile(fp32_lats, 95)), 2),
            "p99_ms": round(float(np.percentile(fp32_lats, 99)), 2),
            "mean_ms": round(float(np.mean(fp32_lats)), 2),
            "fps": round(1000 / float(np.mean(fp32_lats)), 1),
        },
        "int8": {
            "size_mb": round(int8_path.stat().st_size / 1e6, 2),
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "fps": round(1000 / float(np.mean(latencies)), 1),
        },
    }
    bm["size_reduction_pct"] = round((1 - bm["int8"]["size_mb"] / bm["fp32"]["size_mb"]) * 100, 1)

    with open(REPORTS / "v2_edge_benchmark.json", "w") as f:
        json.dump(bm, f, indent=2)
    print(f"   Benchmark: FP32 p95={bm['fp32']['p95_ms']}ms | INT8 p95={bm['int8']['p95_ms']}ms | size -{bm['size_reduction_pct']}%")
    return bm


# ─────────────────────────────────────────────────────────────────────
# 2. Full F1 evaluation across all splits
# ─────────────────────────────────────────────────────────────────────
def evaluate_all_splits(imgsz: int = 640):
    from ultralytics import YOLO
    model = YOLO(str(CHAMPION))

    print("\n[2/5] Evaluating all splits at conf=0.25...")
    results = {"model": str(CHAMPION)}

    for split_name, yaml, split_key in [
        ("train", DATA_YAML, "train"),
        ("val", DATA_YAML, "val"),
        ("test_clean", DATA_YAML, "test"),
        ("test_shifted", SHIFTED_YAML, "test"),
    ]:
        m = model.val(data=yaml, split=split_key, imgsz=imgsz, conf=0.25, verbose=False)
        p = float(m.box.mp)
        r = float(m.box.mr)
        f1 = 2 * p * r / max(p + r, 1e-6)
        per_class = {}
        for idx, cname in enumerate(CLASSES):
            pp = float(m.box.p[idx]) if idx < len(m.box.p) else 0.0
            rr = float(m.box.r[idx]) if idx < len(m.box.r) else 0.0
            ff = 2 * pp * rr / max(pp + rr, 1e-6)
            ap = float(m.box.ap50[idx]) if idx < len(m.box.ap50) else 0.0
            per_class[cname] = {"precision": round(pp, 4), "recall": round(rr, 4), "f1": round(ff, 4), "ap50": round(ap, 4)}
        results[split_name] = {
            "split": split_key,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "per_class": per_class,
        }
        print(f"   {split_name:15s} → P={p:.3f} R={r:.3f} F1={f1:.3f} mAP50={m.box.map50:.3f}")

    with open(REPORTS / "v2_evaluation_splits_f1.json", "w") as f:
        json.dump(results, f, indent=2)
    # also overwrite the canonical file
    shutil.copy(REPORTS / "v2_evaluation_splits_f1.json", REPORTS / "evaluation_splits_f1.json")
    return results


# ─────────────────────────────────────────────────────────────────────
# 3. Qualitative detection gallery
# ─────────────────────────────────────────────────────────────────────
def generate_gallery(imgsz: int = 640):
    import glob
    import cv2
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    print("\n[3/5] Generating qualitative detection gallery...")
    model = YOLO(str(CHAMPION))
    test_img_dir = "data/processed/neu-det/images/test"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.patch.set_facecolor("#0f0f0f")

    for idx, cname in enumerate(CLASSES):
        candidates = sorted(glob.glob(os.path.join(test_img_dir, f"{cname}*.jpg")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
        sample = candidates[0] if candidates else None
        if not sample:
            continue

        results = model.predict(sample, conf=0.2, imgsz=imgsz, verbose=False)
        rendered = results[0].plot(line_width=2, font_size=10)
        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        ax = axes[idx]
        ax.imshow(rendered_rgb)
        ax.set_title(f"Class: {cname}", color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[idx])
            spine.set_linewidth(3)

    plt.suptitle("NEU-DET v2 Defect Inspection — Qualitative Detections (640×640, YOLOv8s/m)",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPORTS / "qualitative_detections_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# 4. Copy F1/PR/confusion matrix curves from best run
# ─────────────────────────────────────────────────────────────────────
def copy_training_plots():
    print("\n[4/5] Copying training plots...")
    # Find best v2 run dir
    v2_summary = REPORTS / "v2_experiments_summary.json"
    if not v2_summary.exists():
        print("   No v2 summary found, skipping.")
        return
    with open(v2_summary) as f:
        experiments = json.load(f)
    if not experiments:
        return
    best = max(experiments, key=lambda x: x["clean_mAP50"])
    run_dir = Path(f"runs/train/{best['run_name']}")
    for plot, dest in [
        ("F1_curve.png", "BoxF1_curve.png"),
        ("PR_curve.png", "BoxPR_curve.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("results.png", "results.png"),
    ]:
        src = run_dir / plot
        if src.exists():
            shutil.copy(src, REPORTS / dest)
            print(f"   Copied {plot} from {best['run_name']}")


# ─────────────────────────────────────────────────────────────────────
# 5. Git commit and push
# ─────────────────────────────────────────────────────────────────────
def git_push():
    print("\n[5/5] Committing and pushing to GitHub...")
    subprocess.run(["git", "add",
                    "src/train_v2.py", "src/post_train.py",
                    "reports/", "models/onnx/"], check=False)
    subprocess.run(["git", "commit", "-m",
                    "feat(v2): 640px YOLOv8s/m retraining, updated F1/mAP results and visuals"],
                   check=False)
    result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ Pushed to GitHub")
    else:
        print(f"   Push failed: {result.stderr[:200]}")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  POST-TRAINING PIPELINE")
    print(f"  Champion: {CHAMPION}")
    print("=" * 60)

    # Detect imgsz from v2 summary
    imgsz = 640
    v2_summary = REPORTS / "v2_experiments_summary.json"
    if v2_summary.exists():
        with open(v2_summary) as f:
            exps = json.load(f)
        if exps:
            best_exp = max(exps, key=lambda x: x["clean_mAP50"])
            imgsz = best_exp.get("imgsz", 640)
            print(f"  Best run: {best_exp['run_name']} | mAP50={best_exp['clean_mAP50']}")

    bm = export_onnx()
    evals = evaluate_all_splits(imgsz=imgsz)
    generate_gallery(imgsz=imgsz)
    copy_training_plots()
    git_push()

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    tc = evals.get("test_clean", {})
    ts = evals.get("test_shifted", {})
    print(f"  Test Clean  → mAP50={tc.get('mAP50', 0):.3f}  F1={tc.get('f1_score', 0):.3f}")
    print(f"  Test Shifted → mAP50={ts.get('mAP50', 0):.3f}  F1={ts.get('f1_score', 0):.3f}")
    print(f"  ONNX INT8   → {bm['int8']['p95_ms']}ms p95  |  {bm['int8']['fps']} FPS")
    print(f"  Size reduction: -{bm['size_reduction_pct']}%")
