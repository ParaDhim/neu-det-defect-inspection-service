"""Export the validation-selected 224px candidate (Run 01, seed 2) to ONNX FP32 + dynamic INT8,
benchmark CPU latency (same settings as the other models), and score val/test/shifted."""
import json
import sys

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_all_runs import CLEAN, SHIFTED, evaluate
from export import benchmark_onnx_cpu, export_to_onnx, quantize_to_int8

OUT = "models/onnx"
fp32 = export_to_onnx(weights_path="models/champion_edge_v3.pt", output_dir=OUT, imgsz=224)
int8 = quantize_to_int8(fp32_onnx_path=fp32, output_int8_path=f"{OUT}/model_edge_int8.onnx")
res = {}
for tag, p in [("fp32", fp32), ("int8", int8)]:
    res[tag] = {"bench": benchmark_onnx_cpu(p, imgsz=224),
                "val": evaluate(p, CLEAN, 224, "val"), "test": evaluate(p, CLEAN, 224),
                "shifted": evaluate(p, SHIFTED, 224)}
res["run10_int8_bench_same_session"] = benchmark_onnx_cpu("models/onnx/model_v1_int8.onnx", imgsz=224)
json.dump(res, open("reports/edge_candidate_run01.json", "w"), indent=2)
for k, v in res.items():
    print("RESULT", k, {a: (b if a == "bench" else b["mAP50"]) for a, b in v.items()} if k != "run10_int8_bench_same_session" else v)
