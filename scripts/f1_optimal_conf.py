"""Validation-split F1-optimal confidence threshold per model (macro F1 over classes)."""
import numpy as np
from ultralytics import YOLO

for label, path, imgsz in [("Run 01 FP32 224px (edge)", "models/onnx/model_edge_fp32.onnx", 224),
                           ("Run 11 FP32 640px", "models/onnx/model_fp32.onnx", 640)]:
    r = YOLO(path).val(data="data/processed/neu-det/dataset.yaml", split="val", conf=0.001, imgsz=imgsz,
                       rect=False, plots=False, verbose=False)
    f1 = np.asarray(r.box.f1_curve).mean(0)
    px = np.linspace(0, 1, f1.shape[0])
    i = int(f1.argmax())
    print(f"RESULT {label}: best val macro-F1 {f1[i]:.3f} at conf {px[i]:.3f}; F1 at 0.28 = {f1[int(0.28*(len(px)-1))]:.3f}")
