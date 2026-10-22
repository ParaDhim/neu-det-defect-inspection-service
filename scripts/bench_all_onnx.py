"""Benchmark every ONNX model back-to-back with identical settings (4 threads, batch 1,
15 warmup, 100 timed runs), so latency comparisons come from one session."""
import json
import sys
sys.path.insert(0, "src")
from export import benchmark_onnx_cpu

M = [("Run 10 FP32 224", "models/onnx/model_v1_fp32.onnx", 224), ("Run 10 dyn INT8 224", "models/onnx/model_v1_int8.onnx", 224),
     ("Run 01 FP32 224", "models/onnx/model_edge_fp32.onnx", 224), ("Run 01 dyn INT8 224", "models/onnx/model_edge_int8.onnx", 224),
     ("Run 11 FP32 640", "models/onnx/model_fp32.onnx", 640), ("Run 11 dyn INT8 640", "models/onnx/model_int8.onnx", 640),
     ("Run 11 QDQ INT8 640", "models/onnx/model_qdq_int8.onnx", 640)]
out = {}
for _ in range(2):  # two passes to expose drift
    for label, p, sz in M:
        b = benchmark_onnx_cpu(p, imgsz=sz, warmup_runs=15, benchmark_runs=100)
        out.setdefault(label, []).append({k: b[k] for k in ("size_mb", "p50_ms", "p95_ms", "mean_latency_ms", "throughput_fps")})
        print("RESULT", label, b["size_mb"], b["p50_ms"], b["p95_ms"], b["throughput_fps"], flush=True)
json.dump(out, open("reports/latency_same_session.json", "w"), indent=2)
