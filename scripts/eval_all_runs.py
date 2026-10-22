"""Single evaluation protocol for every model in the README.

Protocol: NEU-DET test split (179 images), conf=0.001, iou=0.7 (Ultralytics
default), rect=False, imgsz = the model's training/export resolution.

rect=False matters: with the Ultralytics default (rect=True) a .pt model on
200x200 images is padded to ceil(imgsz/32 + 0.5)*32 (672 for imgsz=640, 256 for
224), i.e. evaluated at a higher effective resolution than its fixed-shape ONNX
export, which inflated .pt scores by ~7 mAP50 points. Every model, .pt or ONNX,
now sees exactly imgsz x imgsz inputs.

Model selection must use the `clean_val` entries only; `clean` (test) and
`shifted` (test) are reported afterwards.

Output: reports/unified_eval_conf001.json and reports/unified_eval_conf001.md
"""
import json
from pathlib import Path

from ultralytics import YOLO

CLEAN = "data/processed/neu-det/dataset.yaml"
SHIFTED = "data/processed/neu-det-shifted/dataset_shifted.yaml"
R = "models/runs"

RUNS = [
    ("01_baseline_yolov8n_imgsz200", f"{R}/01_baseline_yolov8n_imgsz200.pt", 200),
    ("02_baseline_yolov8n_imgsz320", f"{R}/02_baseline_yolov8n_imgsz320.pt", 320),
    ("03_robust_yolov8n_light_aug", f"{R}/03_robust_yolov8n_light_aug.pt", 224),
    ("04_robust_yolov8n_heavy_aug", f"{R}/04_robust_yolov8n_heavy_aug.pt", 224),
    ("05_robust_yolov8s_heavy_aug", f"{R}/05_robust_yolov8s_heavy_aug.pt", 224),
    ("06_robust_yolov8n_adamw_cosine", f"{R}/06_robust_yolov8n_adamw_cosine.pt", 224),
    ("07_robust_yolov8n_sgd_momentum", f"{R}/07_robust_yolov8n_sgd_momentum.pt", 224),
    ("08_robust_yolov8n_geom_heavy", f"{R}/08_robust_yolov8n_geom_heavy.pt", 224),
    ("09_robust_yolov8n_imgsz320_heavy", f"{R}/09_robust_yolov8n_imgsz320_heavy.pt", 320),
    ("10_champion_yolov8n_production", "models/champion_baseline_v1.pt", 224),
    ("11_yolov8s_640_heavy_aug", "models/champion_best.pt", 640),
]
ONNX = [
    ("Run 10 ONNX FP32 (224px)", "models/onnx/model_v1_fp32.onnx", 224),
    ("Run 10 ONNX Dynamic INT8 (224px)", "models/onnx/model_v1_int8.onnx", 224),
    ("Run 11 ONNX FP32 (640px)", "models/onnx/model_fp32.onnx", 640),
    ("Run 11 ONNX Dynamic INT8 (640px)", "models/onnx/model_int8.onnx", 640),
    ("Run 11 ONNX Static QDQ INT8 (640px)", "models/onnx/model_qdq_int8.onnx", 640),
]


def evaluate(path, yaml, imgsz, split="test"):
    r = YOLO(path).val(data=yaml, split=split, conf=0.001, imgsz=imgsz, rect=False,
                       plots=False, verbose=False, workers=0)
    return {
        "mAP50": round(float(r.box.map50) * 100, 2),
        "mAP50_95": round(float(r.box.map) * 100, 2),
        "per_class_ap50": [round(float(x) * 100, 2) for x in r.box.ap50],
    }


if __name__ == "__main__":
    out = {"protocol": "test split, conf=0.001, iou=0.7, rect=False, imgsz=native", "models": {}}
    for label, path, imgsz in RUNS + ONNX:
        if not Path(path).exists():
            print("MISSING", path)
            continue
        out["models"][label] = {
            "weights": path, "imgsz": imgsz,
            "clean_val": evaluate(path, CLEAN, imgsz, "val"),
            "clean": evaluate(path, CLEAN, imgsz),
            "shifted": evaluate(path, SHIFTED, imgsz),
        }
        m = out["models"][label]
        print(f"RESULT {label}: val {m['clean_val']['mAP50']} clean {m['clean']['mAP50']} / {m['clean']['mAP50_95']}  "
              f"shifted {m['shifted']['mAP50']}", flush=True)
    Path("reports/unified_eval_conf001.json").write_text(json.dumps(out, indent=2))
