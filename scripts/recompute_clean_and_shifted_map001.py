import json
import os
from pathlib import Path
from ultralytics import YOLO

MODELS = [
    ("Run 11 PyTorch .pt (640px)", "models/champion_best.pt", 640),
    ("Run 11 ONNX FP32 (640px)", "models/onnx/model_fp32.onnx", 640),
    ("Run 11 ONNX Static QDQ INT8 (640px)", "models/onnx/model_qdq_int8.onnx", 640),
    ("Run 11 ONNX Dynamic INT8 (640px)", "models/onnx/model_int8.onnx", 640),
    ("Run 10 Edge PyTorch .pt (224px)", "models/champion_baseline_v1.pt", 224),
    ("Run 10 Edge ONNX FP32 (224px)", "models/onnx/model_v1_fp32.onnx", 224),
    ("Run 10 Edge ONNX INT8 (224px)", "models/onnx/model_v1_int8.onnx", 224),
]

CLEAN_YAML = "data/processed/neu-det/dataset.yaml"
SHIFTED_YAML = "data/processed/neu-det-shifted/dataset_shifted.yaml"

def run_eval(data_yaml, split="test"):
    results = {}
    for label, path, imgsz in MODELS:
        print(f"\nEvaluating {label} on {data_yaml} (split={split}, imgsz={imgsz}, conf=0.001)...")
        model = YOLO(path)
        val_res = model.val(data=data_yaml, split=split, conf=0.001, imgsz=imgsz, verbose=False)
        
        map50 = round(float(val_res.box.map50) * 100, 2)
        map50_95 = round(float(val_res.box.map) * 100, 2)
        crazing_ap = round(float(val_res.box.ap50[0]) * 100, 2) if len(val_res.box.ap50) > 0 else 0.0
        
        per_class_ap = [round(float(x) * 100, 2) for x in val_res.box.ap50]
        
        results[label] = {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "crazing_AP50": crazing_ap,
            "per_class_ap50": per_class_ap,
            "class_names": list(val_res.names.values()),
        }
        print(f" -> {label}: mAP50={map50}% mAP50-95={map50_95}% Crazing AP50={crazing_ap}%")
    return results

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    
    print("=== EVALUATING CLEAN TEST SET (conf=0.001) ===")
    clean_results = run_eval(CLEAN_YAML, "test")
    with open("reports/clean_shared_protocol_conf001.json", "w") as f:
        json.dump(clean_results, f, indent=2)
        
    print("\n=== EVALUATING SHIFTED TEST SET (conf=0.001) ===")
    shifted_results = run_eval(SHIFTED_YAML, "test")
    with open("reports/shifted_shared_protocol_conf001.json", "w") as f:
        json.dump(shifted_results, f, indent=2)
        
    print("\nEvaluation complete! Saved to reports/*_conf001.json")
