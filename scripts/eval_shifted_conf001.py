import json
from ultralytics import YOLO

SHIFTED_YAML = "data/processed/neu-det-shifted/dataset_shifted.yaml"

MODELS = [
    ("Run 11 PyTorch .pt (640px)", "models/champion_best.pt", 640),
    ("Run 11 ONNX FP32 (640px)", "models/onnx/model_fp32.onnx", 640),
    ("Run 11 ONNX Static QDQ INT8 (640px)", "models/onnx/model_qdq_int8.onnx", 640),
    ("Run 11 ONNX Dynamic INT8 (640px)", "models/onnx/model_int8.onnx", 640),
    ("Run 10 Edge PyTorch .pt (224px)", "models/champion_baseline_v1.pt", 224),
    ("Run 10 Edge ONNX FP32 (224px)", "models/onnx/model_v1_fp32.onnx", 224),
    ("Run 10 Edge ONNX INT8 (224px)", "models/onnx/model_v1_int8.onnx", 224),
]

results = {}
for label, path, imgsz in MODELS:
    print(f"\n--- Evaluating {label} on SHIFTED test set (imgsz={imgsz}, conf=0.001) ---")
    model = YOLO(path)
    val_res = model.val(data=SHIFTED_YAML, split="test", conf=0.001, imgsz=imgsz, verbose=False)
    
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
    print(f"RESULT -> {label}: mAP50={map50}% mAP50-95={map50_95}% Crazing AP50={crazing_ap}%")

with open("reports/shifted_shared_protocol_conf001.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll shifted evaluations complete! Saved to reports/shifted_shared_protocol_conf001.json")
