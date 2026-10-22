> **STALE:** latencies here (INT8 7.4 ms) did not reproduce. See `reports/latency_same_session.json` (224px dynamic INT8 is ~5x slower than FP32).

### Edge Model Export & CPU Benchmarking (ONNX FP32 vs INT8)

| Precision | Model Size | Latency p50 | Latency p95 | Mean Latency | Throughput | Size Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP32 (Baseline)** | 12.02 MB | 6.67 ms | 7.34 ms | 6.72 ms | 148.7 FPS | - |
| **INT8 (Quantized)** | 3.17 MB | 6.36 ms | 7.42 ms | 6.49 ms | 154.0 FPS | **-73.6%** |

**Summary:** INT8 quantization achieves a **1.04x speedup** on CPU with a **-1.1% reduction** in p95 tail latency and **73.6% memory compression**.
