"""Operating-point metrics (per-model val-F1-optimal conf) under the same protocol as eval_all_runs.py.

Reports, per model and split: macro P/R (unweighted mean of per-class TP/(TP+FP)
and TP/(TP+FN) from the confusion-matrix counts), micro P/R (pooled counts), F1
from each, and per-class TP/FP/FN. Ultralytics' own `mp`/`mr` are saved as
`ultralytics_macro_*`; they come from interpolated P/R curves at the best-F1
confidence with a different matching rule, so they do not reconcile with counts. Also saves a confusion
matrix figure on the clean test split.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

CLEAN = "data/processed/neu-det/dataset.yaml"
SHIFTED = "data/processed/neu-det-shifted/dataset_shifted.yaml"
MODELS = {
    "Run 01 ONNX FP32 (224px, edge)": ("models/onnx/model_edge_fp32.onnx", 224, "confusion_matrix_edge_fp32.png", 0.22),
    "Run 11 ONNX FP32 (640px)": ("models/onnx/model_fp32.onnx", 640, "confusion_matrix_run11_fp32.png", 0.19),
}
SPLITS = [("train", CLEAN), ("val", CLEAN), ("test", CLEAN), ("test_shifted", SHIFTED)]
NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


out = {"protocol": "conf=CONF, iou=0.45, rect=False, imgsz=native"}
for label, (path, imgsz, cm_name, CONF) in MODELS.items():
    out[label] = {}
    for split, yaml in SPLITS:
        real_split = "test" if split == "test_shifted" else split
        r = YOLO(path).val(data=yaml, split=real_split, conf=CONF, iou=0.45, imgsz=imgsz, rect=False,
                           plots=True, verbose=False,
                           project="runs/opeval", name=f"{label}-{split}".replace(" ", "_"), exist_ok=True)
        m = r.confusion_matrix.matrix  # (nc+1, nc+1): rows = predicted, cols = ground truth, last = background
        tp = np.diag(m)[:6]
        fp = m[:6].sum(1) - tp  # predictions of class c not matched to a GT of class c
        fn = m[:, :6].sum(0) - tp  # GT of class c not predicted as c
        micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
        micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
        mp = float(np.mean(tp / np.maximum(tp + fp, 1)))
        mr = float(np.mean(tp / np.maximum(tp + fn, 1)))
        out[label][split] = {
            "macro_precision": round(mp, 3), "macro_recall": round(mr, 3), "macro_f1": round(f1(mp, mr), 3),
            "micro_precision": round(float(micro_p), 3), "micro_recall": round(float(micro_r), 3),
            "micro_f1": round(f1(micro_p, micro_r), 3),
            "ultralytics_macro_precision": round(float(r.box.mp), 3),
            "ultralytics_macro_recall": round(float(r.box.mr), 3),
            "mAP50": round(float(r.box.map50), 3), "mAP50_95": round(float(r.box.map), 3),
            "per_class": {n: {"gt": int(tp[i] + fn[i]), "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i])}
                          for i, n in enumerate(NAMES)},
        }
        if split == "test":
            shutil.copy2(Path(r.save_dir) / "confusion_matrix.png", f"reports/{cm_name}")
            shutil.copy2(Path(r.save_dir) / "confusion_matrix_normalized.png",
                         f"reports/{cm_name.replace('.png', '_normalized.png')}")
        print("RESULT", label, split, {k: v for k, v in out[label][split].items() if k != "per_class"}, flush=True)
Path("reports/operating_point_opt.json").write_text(json.dumps(out, indent=2))
