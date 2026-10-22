"""
Edge Export, Quantization, and Benchmarking Pipeline for Steel Defect Inspection.
Exports trained PyTorch YOLOv8 models to ONNX FP32, performs INT8 dynamic quantization
via ONNX Runtime, and benchmarks CPU latency (p50, p90, p95, p99), throughput (FPS),
and model footprint.
"""

import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)
import cv2
from ultralytics import YOLO


class SteelDefectCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed calibration images to ONNX Runtime static quantizer."""

    def __init__(self, image_dir: str = "data/processed/neu-det/images/train", imgsz: int = 640, max_samples: int = 50):
        self.image_paths = sorted(list(Path(image_dir).glob("*.jpg")))[:max_samples]
        self.imgsz = imgsz
        self.enum_data = iter(self.image_paths)
        print(f"Calibration data reader initialized with {len(self.image_paths)} images at {imgsz}x{imgsz}")

    def get_next(self):
        try:
            img_path = next(self.enum_data)
        except StopIteration:
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return {"images": img}


def export_to_onnx(
    weights_path: str = "models/champion_best.pt",
    output_dir: str = "models/onnx",
    imgsz: int = 200,
    opset: int = 17,
) -> str:
    """Exports trained YOLOv8 model to ONNX FP32 format."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading weights from {weights_path}...")
    model = YOLO(weights_path)

    # Export to ONNX
    print(f"Exporting to ONNX FP32 (imgsz={imgsz}, opset={opset})...")
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=opset,
        simplify=True,
    )

    dest_fp32 = os.path.join(output_dir, "model_fp32.onnx")
    if os.path.abspath(exported_path) != os.path.abspath(dest_fp32):
        import shutil
        shutil.copy2(exported_path, dest_fp32)

    # Verify ONNX model graph
    onnx_model = onnx.load(dest_fp32)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(dest_fp32) / (1024 * 1024)
    print(f"Successfully exported valid ONNX FP32 model: {dest_fp32} ({file_size_mb:.2f} MB)")
    return dest_fp32


def quantize_to_int8(
    fp32_onnx_path: str = "models/onnx/model_fp32.onnx",
    output_int8_path: str = "models/onnx/model_int8.onnx",
) -> str:
    """Performs dynamic INT8 quantization on the FP32 ONNX model."""
    print(f"Quantizing {fp32_onnx_path} to INT8 dynamic quantization...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=output_int8_path,
        weight_type=QuantType.QInt8,
    )

    size_fp32 = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_int8_path) / (1024 * 1024)
    reduction = ((size_fp32 - size_int8) / size_fp32) * 100.0

    print(f"Quantization complete: {output_int8_path}")
    print(f"Size reduction: {size_fp32:.2f} MB -> {size_int8:.2f} MB (-{reduction:.1f}%)")
    return output_int8_path


def quantize_static_qdq(
    fp32_onnx_path: str = "models/onnx/model_fp32.onnx",
    output_qdq_path: str = "models/onnx/model_qdq_int8.onnx",
    calibration_image_dir: str = "data/processed/neu-det/images/train",
    imgsz: int = 640,
    max_samples: int = 50,
) -> str:
    """Performs static (QDQ) quantization with calibration data for CPU acceleration."""
    print(f"Quantizing {fp32_onnx_path} to static QDQ INT8 with calibration...")
    reader = SteelDefectCalibrationDataReader(
        image_dir=calibration_image_dir,
        imgsz=imgsz,
        max_samples=max_samples,
    )
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=output_qdq_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    size_fp32 = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    size_qdq = os.path.getsize(output_qdq_path) / (1024 * 1024)
    reduction = ((size_fp32 - size_qdq) / size_fp32) * 100.0
    print(f"Static QDQ complete: {output_qdq_path}")
    print(f"Size reduction: {size_fp32:.2f} MB -> {size_qdq:.2f} MB (-{reduction:.1f}%)")
    return output_qdq_path


def benchmark_onnx_cpu(
    onnx_path: str,
    imgsz: int = 200,
    warmup_runs: int = 50,
    benchmark_runs: int = 200,
    intra_op_threads: int = 4,
) -> dict:
    """
    Benchmarks single-stream CPU inference latency (p50, p90, p95, p99) and FPS.
    """
    print(f"\nBenchmarking {onnx_path} on CPU ({intra_op_threads} threads, {benchmark_runs} runs)...")

    # Session options
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = intra_op_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = [1, 3, imgsz, imgsz]

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})

    # Timed runs
    latencies_ms = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    p50 = float(np.percentile(latencies_ms, 50))
    p90 = float(np.percentile(latencies_ms, 90))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    fps = float(1000.0 / mean_lat)
    size_mb = float(os.path.getsize(onnx_path) / (1024 * 1024))

    stats = {
        "model_path": onnx_path,
        "size_mb": round(size_mb, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "throughput_fps": round(fps, 1),
        "benchmark_runs": benchmark_runs,
    }
    print(f"Results: mean={mean_lat:.2f}ms | p50={p50:.2f}ms | p95={p95:.2f}ms | FPS={fps:.1f}")
    return stats


def run_edge_export_and_benchmark(
    weights_path: str = "models/champion_best.pt",
    output_dir: str = "models/onnx",
    reports_path: str = "reports/edge_benchmark.json",
    imgsz: int = 200,
):
    """Executes full export, quantization, and benchmark comparison."""
    os.makedirs(os.path.dirname(reports_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    fp32_path = export_to_onnx(weights_path=weights_path, output_dir=output_dir, imgsz=imgsz)
    int8_path = os.path.join(output_dir, "model_int8.onnx")
    quantize_to_int8(fp32_onnx_path=fp32_path, output_int8_path=int8_path)

    fp32_bench = benchmark_onnx_cpu(fp32_path, imgsz=imgsz)
    int8_bench = benchmark_onnx_cpu(int8_path, imgsz=imgsz)

    # Compute comparative improvements
    speedup = fp32_bench["mean_latency_ms"] / max(int8_bench["mean_latency_ms"], 1e-4)
    p95_improvement_pct = ((fp32_bench["p95_ms"] - int8_bench["p95_ms"]) / fp32_bench["p95_ms"]) * 100.0
    size_reduction_pct = ((fp32_bench["size_mb"] - int8_bench["size_mb"]) / fp32_bench["size_mb"]) * 100.0

    comparison = {
        "hardware": "CPU (Intel/Apple)",
        "input_resolution": f"{imgsz}x{imgsz}",
        "fp32": fp32_bench,
        "int8": int8_bench,
        "comparison": {
            "speedup_factor": round(speedup, 2),
            "p95_latency_reduction_pct": round(p95_improvement_pct, 1),
            "size_reduction_pct": round(size_reduction_pct, 1),
        },
    }

    with open(reports_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Format Markdown comparison table
    md_content = f"""### Edge Model Export & CPU Benchmarking (ONNX FP32 vs INT8)

| Precision | Model Size | Latency p50 | Latency p95 | Mean Latency | Throughput | Size Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP32 (Baseline)** | {fp32_bench['size_mb']} MB | {fp32_bench['p50_ms']} ms | {fp32_bench['p95_ms']} ms | {fp32_bench['mean_latency_ms']} ms | {fp32_bench['throughput_fps']} FPS | - |
| **INT8 (Quantized)** | {int8_bench['size_mb']} MB | {int8_bench['p50_ms']} ms | {int8_bench['p95_ms']} ms | {int8_bench['mean_latency_ms']} ms | {int8_bench['throughput_fps']} FPS | **-{size_reduction_pct:.1f}%** |

**Summary:** INT8 quantization achieves a **{speedup:.2f}x speedup** on CPU with a **{p95_improvement_pct:.1f}% reduction** in p95 tail latency and **{size_reduction_pct:.1f}% memory compression**.
"""
    with open("reports/edge_benchmark.md", "w") as f:
        f.write(md_content)

    print("\n" + md_content)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and benchmark YOLOv8 to ONNX FP32 and INT8")
    parser.add_argument("--weights", type=str, default="models/champion_best.pt", help="Path to best PyTorch model")
    parser.add_argument("--imgsz", type=int, default=200, help="Input image dimension")
    args = parser.parse_args()

    run_edge_export_and_benchmark(weights_path=args.weights, imgsz=args.imgsz)
