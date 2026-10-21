# Industrial Steel Surface Defect Inspection Service

[![CI/CD Pipeline](https://github.com/ParaDhim/neu-det-defect-inspection-service/actions/workflows/ci.yml/badge.svg)](https://github.com/ParaDhim/neu-det-defect-inspection-service/actions)
[![uv Powered](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-green.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI_0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **One-line Pitch:** An end-to-end metallurgical defect-detection and quality assurance platform—from robust YOLOv8 training and statistical drift monitoring to containerized deployment, delivering near-the-line inference at **7.4ms p95** on the lightweight edge baseline (Run 10, YOLOv8n at 224px, 154 FPS) and **75.0ms p95** on the selected model (Run 11, YOLOv8s at 640px via static QDQ INT8, 13.6 FPS) on CPU. Managed natively with **Astral `uv`** via `pyproject.toml` and `uv.lock`.

---

## Visual Defect Inspection Gallery

![Qualitative Defect Detections Grid](reports/qualitative_detections_grid.png)

*Figure 1: Automated defect localizations on test steel surfaces across all 6 target defect classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) with predicted bounding boxes and calibrated class confidence scores.*

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Dataset & Stratified Distribution](#dataset--stratified-distribution)
3. [Experiment Tracking & Research Results](#experiment-tracking--research-results)
4. [Precision, Recall & F1 Evaluation Across Splits](#precision-recall--f1-evaluation-across-splits)
5. [Harsh Lighting Shift & Robustness Analysis](#harsh-lighting-shift--robustness-analysis)
6. [Error Analysis & Confusion Matrix](#error-analysis--confusion-matrix)
7. [Edge Export & CPU Benchmarking (FP32 vs Dynamic INT8 vs Static QDQ)](#edge-export--cpu-benchmarking-fp32-vs-dynamic-int8-vs-static-qdq)
8. [Covariate Drift Monitoring & Automated Retraining Trigger](#covariate-drift-monitoring--automated-retraining-trigger)
9. [Unsupervised Anomaly Baseline (PatchCore vs Supervised YOLO)](#unsupervised-anomaly-baseline-patchcore-vs-supervised-yolo)
10. [Automated Metallurgical & LLM Reporting](#automated-metallurgical--llm-reporting)
11. [Serving API Specification](#serving-api-specification)
12. [Containerization & Kubernetes Deployment](#containerization--kubernetes-deployment)
13. [When ML is NOT the Right Answer](#when-ml-is-not-the-right-answer)
14. [Quickstart & Reproduction Guide (`uv` Native)](#quickstart--reproduction-guide-uv-native)

---

## System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Preprocessing"]
        A["NEU-DET Raw Images & Pascal VOC XML"] --> B["data/convert_and_split.py\nSeeded Stratified Split (80/10/10)"]
        B --> C["Clean Dataset\n(YOLO Format)"]
        C --> D["src/augment.py\nPhotometric Distortion Engine"]
        D --> E["Shifted Evaluation Benchmark\n(Darkening, Glare, Blur, Noise, Shadows)"]
    end

    subgraph Training_Pipeline["2. Experiment Tracking & Edge Export"]
        C --> F["src/train.py\nYOLOv8 Training + MLflow"]
        F --> G["MLflow Tracking & Registry\n(11 Multi-tier Experiment Runs)"]
        F --> H["models/champion_best.pt\nSelected Weights (Run 11: YOLOv8s 640px)"]
        H --> I["src/export.py\nStatic QDQ & Dynamic INT8 Quantization"]
        I --> J1["Edge Baseline: models/onnx/model_v1_int8.onnx\n(Run 10: YOLOv8n 224px, 3.17 MB, 7.4ms p95 CPU, 154 FPS)"]
        I --> J2["Selected Model: models/onnx/model_qdq_int8.onnx\n(Run 11: YOLOv8s 640px, 17.0 MB, 75.0ms p95 CPU, 13.6 FPS)"]
    end

    subgraph Serving_Cluster["3. Kubernetes Container Cluster"]
        K["Steel Strip Camera Stream"] --> L["FastAPI Gateway (/predict)"]
        L --> M["ONNX Runtime Engine\n(CPU Execution Provider)"]
        J2 --> M
        M --> N["NMS Postprocessor\n(Calibrated Validation F1 Conf Threshold 0.28)"]
        N --> O["Detections: Class, BBox, Confidence"]
        O --> P["src/serve/report.py\nMetallurgical QA & Disposition Report"]
    end

    subgraph Monitoring_Engine["4. Statistical Drift & Anomaly Inspection"]
        K --> Q["src/monitor/drift.py\nPhotometric Feature Extractor"]
        Q --> R["Two-Sample Kolmogorov-Smirnov Test\n(Live Window vs Baseline)"]
        R --> S{"p-value < 0.05 & Effect > 0.5?"}
        S -- Yes --> T["Raise Drift Alert Flag & Trigger Retraining Pipeline Rule"]
        S -- No --> U["Normal Steady-State Inspection"]
        K -.-> V["src/anomaly/patchcore.py\nPatchCore Memory Bank Baseline"]
        V -.-> W["Unsupervised Anomaly Heatmap (Preliminary Prototype)"]
    end
```

---

## Dataset & Stratified Distribution

The primary benchmark is **NEU-DET (Northeastern University Steel Surface Defect Database)** consisting of hot-rolled steel strip surface defects across six metallurgical defect classes.

### Dataset Verification & Clean VOC-to-YOLO Conversion
- **Upstream License:** Academic research release (Song Kechen & Yan Yunhui, IEEE TIM 2020).
- **Images:** 1,800 total unique images ($200 \times 200$).
- **Annotations:** Pascal VOC XML normalized to YOLO bounding box format $[x_{\text{center}}, y_{\text{center}}, w, h] \in [0.0, 1.0]$.
- **Partition:** Seeded, class-stratified split (**80% Train, 10% Val, 10% Test**) using `seed=42`: **1,441 Train, 180 Val, 179 Test**. Zero leakage verified across split partitions.
- **Drift Baseline Extracted:** 1,438 training images with valid verified bounding boxes had baseline photometric statistics extracted.

### Class Distribution Summary

| Class ID | Defect Class | Train Images | Val Images | Test Images | Total Images with Class | Train Boxes | Val Boxes | Test Boxes | Total Boxes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `crazing` | 240 | 30 | 30 | 300 | 552 | 58 | 79 | 689 |
| 1 | `inclusion` | 307 | 38 | 37 | 382 | 829 | 93 | 89 | 1011 |
| 2 | `patches` | 278 | 33 | 31 | 342 | 715 | 82 | 84 | 881 |
| 3 | `pitted_surface` | 241 | 30 | 30 | 301 | 348 | 43 | 41 | 432 |
| 4 | `rolled-in_scale` | 240 | 30 | 30 | 300 | 511 | 60 | 57 | 628 |
| 5 | `scratches` | 240 | 30 | 30 | 300 | 442 | 53 | 53 | 548 |
| **Total** | **All Unique Images** | **1,441** | **180** | **179** | **1,800** | **3,397** | **389** | **403** | **4,189** |

*Note on Multi-Label Co-occurrence: In the table above, the sum of image presences across individual classes equals 1,925 (greater than the 1,800 unique images) because individual steel strip images frequently contain multiple distinct defect categories co-occurring on the same strip surface (for example, non-metallic inclusions appearing in proximity to rolled-in scale).*

---

## Experiment Tracking & Research Results

We executed a systematic Design of Experiments (DOE) logged into **MLflow** (`sqlite:///mlflow.db`). Models were evaluated on both the canonical **Clean Test Set** and the physically perturbed **Shifted-Lighting Benchmark**.

### Master Experiment Comparison Table (7 Selected Runs of 11 Total DOE Runs)

| Run ID | Model Architecture | Resolution | Epochs | Augmentation Strategy | Clean Test mAP50 | Shifted Test mAP50 | mAP50 Drop | Rel Drop (%) | Clean mAP50-95 | Status |
|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `01_baseline_yolov8n_imgsz200` | YOLOv8n (3.0M) | 200x200 | 10 | Ultralytics default (standard HSV/mosaic; no lighting shift) | **66.2%** | 17.9% | -48.3% | -73.0% | 34.1% | *Clean Baseline Archive* |
| `02_baseline_yolov8n_imgsz320` | YOLOv8n (3.0M) | 320x320 | 10 | High-Res Clean Baseline | 62.8% | 14.8% | -48.0% | -76.5% | 31.9% | *Baseline Archive* |
| `03_robust_yolov8n_light_aug` | YOLOv8n (3.0M) | 224x224 | 10 | Light Aug (HSV $v=0.2$, mosaic) | 62.9% | 14.6% | -48.4% | -76.9% | 32.0% | *Baseline Archive* |
| `04_robust_yolov8n_heavy_aug` | YOLOv8n (3.0M) | 224x224 | 10 | Heavy Aug (HSV $v=0.6$, scale, erasing) | 59.8% | 15.1% | -44.7% | -74.8% | 30.5% | *Baseline Archive* |
| `05_robust_yolov8s_heavy_aug` | YOLOv8s (11.1M) | 224x224 | 10 | Capacity Scaling + Heavy Aug | 56.3% | 20.8% | -35.5% | -63.1% | 27.0% | *Baseline Archive* |
| `10_champion_yolov8n_production` | YOLOv8n (3.0M) | 224x224 | 12 | AdamW + Tuned Photometric Aug | 51.7% | 18.8% | -32.9% | -63.6% | 23.8% | *Robust Edge Baseline* |
| **`11_yolov8s_640_heavy_aug`** | **YOLOv8s (11.1M)** | **640x640** | **80** | **AdamW + Cosine LR + CopyPaste + Scale** | **63.8%** | **23.2%** | **-40.6%** | **-63.7%** | **35.1%** | **Selected Model (Run 11)** |

> **Comparative Baseline Analysis (Run 01 vs. Run 10 vs. Selected Run 11):**
> - **Run 01 Clean Baseline (66.2% clean mAP50):** Trained with standard Ultralytics defaults on clean data, Run 01 achieves the highest score on the clean test set (66.2%). However, under simulated lighting shifts (glare, shadows, noise, darkening), its mAP collapses from **66.2% to 17.9% (a -73.0% relative drop)**.
> - **Selected Model Run 11 (63.8% clean mAP50, 23.2% shifted mAP50):** The selected model trades 2.4% clean mAP50 compared to Run 01 to achieve:
>   1. **Higher mAP50-95:** Higher overall localization precision (**35.1%** vs 34.1%).
>   2. **Higher Shifted Robustness:** Lifts shifted-lighting mAP50 from **17.9% to 23.2% (+29.5% relative gain)**, reducing the relative performance drop from -73.0% down to -63.7%.
>   3. **Multi-Factorial Attribution Note:** On a test split of 179 images, this 5.3-point shifted mAP gain is a composite result of scaling capacity (YOLOv8s vs YOLOv8n), an extended training budget (80 epochs vs 10–12), cosine learning rate scheduling, and heavy augmentations; isolating the exact contribution of each factor would require multi-seed controlled ablations.
>   4. **Defect Recall Gains:** Compared to the low-resolution baseline (Run 10), `crazing` AP50 increased from **2.5% to 20.4%** and `rolled-in_scale` increased from **14.9% to 38.7%**, elevating test F1 from **0.540** to **0.696** (at validation-calibrated threshold 0.28).

### Per-Class AP50 Progression: Baseline (Run 10) vs Selected Model (Run 11)

| Defect Class | Previous Baseline AP50 | Selected Model AP50 | Relative Gain | Analysis & Hypotheses |
|:---|:---:|:---:|:---:|:---|
| `crazing` | 2.50% | **20.44%** | **+717.6% (8.2x)** | *Hypothesis:* Upsampling 200px inputs to 640px expands P3 feature maps from 25x25 to 80x80, which we hypothesize helps retain fine crack representations that were previously lost during early pooling. Note that longer training and model capacity are confounding factors. |
| `rolled-in_scale` | 14.94% | **38.73%** | **+159.2% (2.6x)** | Larger model capacity (11.1M vs 3.0M params) helps separate diffuse iron oxide gradient boundaries from baseline steel surface shading. |
| `inclusion` | 74.27% | **80.06%** | **+7.8%** | Higher feature resolution sharpens particulate boundary localization. |
| `patches` | 88.25% | **83.73%** | -5.1% | Macroscopic boundary localization remains stable across runs. |
| `pitted_surface` | 71.50% | **72.62%** | **+1.6%** | Consistent crater depression identification across resolutions. |
| `scratches` | 58.82% | **87.14%** | **+48.1%** | Longitudinal linear transit features benefit significantly from expanded training epochs and cosine decay. |
| **Overall mAP50** | **51.71%** | **63.79%** | **+23.4%** | **Substantial gain over Run 10 across all classes.** |
| **Test F1 Score** | **0.540** | **0.696** | **+28.9%** | **Evaluated at validation-calibrated confidence threshold 0.28.** |

---

## Precision, Recall & F1 Evaluation Across Splits

To benchmark generalization, Precision, Recall, F1 score, and mAP were evaluated across all splits using the confidence threshold (**0.28**), which was selected based on peak F1 performance on the **validation split**.

### Train vs Val vs Test Empirical Metrics (Selected Model)

| Dataset Split | Precision | Recall | **F1 Score** | mAP@50 | mAP@50-95 | Evaluated Images |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Split (In-Domain)** | 0.751 | 0.744 | **0.748** | 0.700 | 0.416 | 1,441 |
| **Validation Split (Tuning Split)** | 0.710 | 0.675 | **0.692** | 0.614 | 0.354 | 180 |
| **Test Split (Clean Benchmark)** | **0.698** | **0.693** | **0.696** | **0.638** | **0.351** | 179 |
| **Test Split (Harsh Shifted)** | 0.460 | 0.292 | **0.358** | 0.232 | 0.128 | 179 |

<p align="center">
  <img src="reports/BoxF1_curve.png" width="48%" alt="F1-Confidence Curve" />
  <img src="reports/BoxPR_curve.png" width="48%" alt="Precision-Recall Curve" />
</p>
<p align="center"><em>Figure 2: (Left) F1-Confidence Curve evaluated on the validation split showing peak F1 score of 0.692 at confidence threshold 0.28 (evaluated at 0.696 on test); (Right) Precision-Recall Curve across all 6 defect classes on validation.</em></p>

---

## Harsh Lighting Shift & Robustness Analysis

In real-world steel manufacturing, visual inspection cameras suffer from harsh factory conditions:
1. **Severe Under-exposure & Over-exposure:** Conveyor lighting fluctuation and variable coil reflectivity.
2. **Specular Reflections & Hotspots:** High-intensity halogen lamps producing localized glare rings.
3. **Occlusion & Shadow Casts:** Overhead gantry arms, tension levelers, and roller frames casting directional shadows.
4. **Thermal Grain & Sensor Noise:** High ambient temperature in hot rolling mills.
5. **Defocus & Vibration Blur:** High-speed line velocities ($15-25 \text{ m/s}$).

![Lighting Shift Visual Comparison](reports/lighting_shift_visual_comparison.png)

*Figure 3: Side-by-side visual comparison of test images under canonical clean inspection lighting (Left) versus harsh simulated industrial plant perturbations with shadows, noise, and glare (Right).*

### Per-Class Performance Breakdown: Clean vs Shifted (Selected Model)

| Defect Class | Clean AP50 | Shifted AP50 | Absolute Change | Vulnerability Diagnosis |
|:---|:---:|:---:|:---:|:---|
| `scratches` | **87.1%** | **59.0%** | -28.1% | Linear longitudinal profile remains distinguishable under directional shadows. |
| `patches` | **83.7%** | **41.6%** | -42.1% | Large irregular surface area enables high recognition despite factory contrast drops. |
| `inclusion` | **80.1%** | 10.0% | -70.1% | Slag inclusions remain sharp under clean light; prone to shadow noise confusion. |
| `pitted_surface` | **72.6%** | 24.2% | -48.4% | Crater depth contrast blurs under heavy optical defocus. |
| `rolled-in_scale` | **38.7%** | 0.4% | -38.3% | Highly sensitive to severe underexposure; requires contrast enhancement. |
| `crazing` | **20.4%** | 3.7% | -16.7% | Drops from 20.4% to 3.7% under extreme optical defocus and glare; remains fragile under lighting shifts and requires stabilized industrial illumination. |

---

## Error Analysis & Confusion Matrix

<p align="center">
  <img src="reports/confusion_matrix.png" width="70%" alt="Confusion Matrix" />
</p>
<p align="center"><em>Figure 4: NEU-DET Test Split Confusion Matrix heatmap with row-normalized class predictions for Selected Model.</em></p>

### Metallurgical Error Analysis with Matrix Metrics
*Evaluated on the NEU-DET Test Partition (179 images, 403 ground-truth bounding boxes) at operational confidence threshold 0.28 (IoU matching threshold = 0.45):*

In the test-set confusion matrix, **columns represent ground-truth instances** (totaling 403 defect instances across 179 images) and **rows represent model predictions**:
1. **Crazing Background Ambiguity & Fine-Crack Recall:** Of **79** ground-truth crazing bounding boxes, **20** were localized correctly (True Positive), while **59** were missed as background (False Negative). Subtle steel grain striations produced **9** background false alarms flagged as crazing (**29** total crazing predictions: 20 TP + 9 FP).
2. **Inclusion Recall & Background Misses:** **76** of **89** ground-truth inclusion instances were correctly identified (85.4% recall), with **13** missed as background due to low particle-to-matrix contrast. Low-contrast surface artifacts resulted in **37** background false positives (**113** total inclusion predictions: 76 TP + 37 FP).
3. **Cross-Class Defect Confusion:** Direct cross-defect misclassification remained zero across all classes (e.g., 0 confusion between inclusion and pitted surface, 0 between crazing and patches). All errors consist entirely of defect-versus-background boundary ambiguity under varying surface illumination.
4. **Patches Superior Detectability:** Patches exhibit large macroscopic surface boundaries with distinct texture roughness: **73** of **84** ground-truth patches were detected (86.9% recall, 11 missed), with **14** background false positives (**87** total patches predictions: 73 TP + 14 FP).
5. **Scratches Longitudinal Linearity:** **48** of **53** ground-truth scratches were correctly identified (90.6% recall, 5 missed), benefiting from linear transit alignment along the rolling axis, with **11** background false positives (**59** total scratches predictions: 48 TP + 11 FP).
6. **Pitted Surface & Rolled-in Scale:** **31** of **41** pitted surface boxes were detected (10 missed, 6 background false alarms; **37** total predictions), and **28** of **57** rolled-in scale boxes were detected (29 missed, 17 background false alarms; **45** total predictions).

---

## Edge Export & CPU Benchmarking (FP32 vs Dynamic INT8 vs Static QDQ)

We exported models to ONNX opset 17 and systematically compared **FP32**, **Dynamic INT8**, and **Static QDQ INT8 (with Calibration Data)** on CPU.

### Benchmark Environment
- **Processor:** Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz
- **Execution Provider:** ONNX Runtime 1.24.3 (Python) / 1.31.0 Engine (`CPUExecutionProvider`)
- **Execution Config:** 4 intra-op threads, single-stream batch size = 1
- **Iterations:** 60 timed benchmark runs following 15 warmup passes
- **Accuracy Split:** NEU-DET test split (179 images) evaluated under the standard detection protocol across the full precision-recall curve at confidence threshold `conf=0.001` (operational decision metrics reported at `conf=0.28`).

### CPU Latency, Memory Footprint & Quantization Accuracy Comparison

| Model Architecture & Quantization Tier | Model Footprint | Latency p50 | Latency p95 | Mean Latency | Throughput | Size Reduction | Clean Test mAP50 | Clean Test mAP50-95 | Shifted Test mAP50 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Edge Baseline YOLOv8n FP32 (224px, Run 10)** | 12.02 MB | 6.67 ms | **7.34 ms** | 6.72 ms | 148.7 FPS | Baseline | **61.7%** | 22.7% | **20.0%** |
| **Edge Baseline YOLOv8n Dynamic INT8 (224px, Run 10)** | **3.17 MB** | 6.36 ms | **7.42 ms** | 6.49 ms | **154.0 FPS** | **-73.6%** | **61.8%** | 23.5% | **21.1%** |
| **Selected YOLOv8s FP32 (640px, Run 11)** | 42.68 MB | 69.85 ms | **71.78 ms** | 70.01 ms | 14.3 FPS | Baseline v2 | **68.3%** | **30.3%** | **24.1%** |
| **Selected YOLOv8s Dynamic INT8 (640px, Run 11)** | **10.96 MB** | 408.55 ms | **416.34 ms** | 409.79 ms | 2.4 FPS | **-74.3%** | **59.1%** (-9.2%) | 25.5% | **20.5%** |
| **Selected YOLOv8s Static QDQ INT8 (640px, Run 11)** | **17.01 MB** | 73.12 ms | **75.03 ms** | 73.38 ms | **13.6 FPS** | **-60.1%** | **65.7%** (-2.7%) | **28.5%** | **20.6%** |

*Note on Latency Benchmarking: Latencies were standardized and re-benchmarked under identical conditions on an Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz using 4 intra-op threads, ONNX Runtime 1.24.3, batch=1, across 60 timed iterations following 15 warmup passes (FP32 p95: 71.8ms; dynamic INT8 p95: 416.3ms; static QDQ INT8 p95: 75.0ms). Prior unstandardized runs on differing thread counts and uncalibrated session configurations yielded variable latencies.*

### Technical Post-Mortem: Why Dynamic INT8 Slowed Down and What Static QDQ Recovered
1. **Why Dynamic INT8 Slowed Down:** ONNX Runtime's dynamic quantizer (`quantize_dynamic`) quantizes weights offline, but dynamically inspects activation tensors at runtime to compute min/max clipping and scale factors for each layer. For deep 2D convolutional networks at 640x640, the per-layer activation quantization overhead and fallback to FP32 for unsupported operators drastically exceeded INT8 GEMM gains, causing a ~5.8x slowdown (416.3 ms vs 71.8 ms p95) and a 9.2-point drop in mAP50.
2. **What Static QDQ Recovered:** Static QDQ quantization (`quantize_static`) with 60 calibration images removed the runtime activation-scaling overhead. Latency fell from 416.3 ms (dynamic INT8) to 75.0 ms p95, and the model shrank 60.1% (42.68 MB to 17.01 MB). It is not faster than FP32 (71.8 ms p95) on this CPU and costs 2.7 points of mAP50 (65.7% vs 68.3%) under the shared evaluation protocol, so its benefit here is model size, not speed. Next step: test on a CPU with INT8 acceleration (VNNI/AVX-512) and exclude the detection head from quantization.

### Model Selection: 224px Edge Baseline vs 640px Models

**Model selection:** On the shared evaluation protocol (`conf=0.001`), the 224px INT8 model matches the 640px QDQ model on shifted mAP50 (**21.1% vs 20.6%**) and trails by only 3.9 points on clean mAP50 (**61.8% vs 65.7%**) at roughly one-tenth of the latency (**7.4 ms vs 75.0 ms p95**, 154.0 FPS vs 13.6 FPS), making it the superior deployment choice for high-throughput factory lines where compute and latency dominate.

Furthermore, if model file size is not a constraint, **640px FP32 strictly dominates 640px Static QDQ INT8** on this CPU: it delivers higher clean accuracy (**68.3% vs 65.7%**), higher shifted robustness (**24.1% vs 20.6%**), and slightly lower latency (**71.8 ms vs 75.0 ms p95**). The 640px QDQ model is retained specifically for storage- or memory-constrained edge appliances where footprint must remain under 20 MB (17.01 MB, -60.1% vs FP32) while preserving fine-crack crazing recall (37.5% AP50 vs 29.9% for 224px INT8) without suffering dynamic INT8's 5.8x runtime slowdown.

| Deployed Architecture & Quantization Tier | Input Resolution | CPU Latency p95 | Throughput | Clean Test mAP50 | Shifted Test mAP50 | Crazing AP50 | Model Footprint | Primary Deployment Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Edge Baseline YOLOv8n INT8 (Run 10)** | 224x224 | **7.4 ms** | **154.0 FPS** | 61.8% | **21.1%** | 29.9% | **3.17 MB** | **Production Conveyor Line Choice:** Maximum throughput and minimal footprint. Matches 640px shifted mAP. |
| **Selected YOLOv8s FP32 (Run 11)** | 640x640 | **71.8 ms** | 14.3 FPS | **68.3%** | **24.1%** | 35.8% | 42.68 MB | **High-Accuracy Line Choice:** Dominates QDQ in accuracy and latency when RAM/disk size is unconstrained. |
| **Selected YOLOv8s Static QDQ INT8 (Run 11)** | 640x640 | **75.0 ms** | 13.6 FPS | **65.7%** | 20.6% | **37.5%** | **17.01 MB** | **Memory-Constrained Choice:** 60.1% smaller than FP32; highest fine-crack (`crazing`) recall without dynamic INT8 slowdown. |

---

## Covariate Drift Monitoring & Automated Retraining Trigger

Visual inspection systems in metallurgy inevitably experience distribution drift due to lamp degradation, lens contamination, and seasonal temperature variance.

### Statistical Framework
- **Photometric Feature Tracking:** Every inference image is evaluated for **Brightness** (mean intensity $\mu$), **Contrast** (standard deviation $\sigma$), **Sharpness** (Laplacian variance $\text{Var}(\nabla^2 I)$), and **Shannon Entropy**.
- **Two-Sample Kolmogorov-Smirnov Test:** A sliding window ($W=40-50$ requests) is evaluated against empirical baseline distributions ($N=1,438$ training images).
- **Automated Retraining Rule:** If $p\text{-value} < 0.05$ on Brightness/Contrast AND normalized effect size $\Delta = |\bar{X}_{\text{win}} - \mu_{\text{base}}| / \sigma_{\text{base}} > 0.5$, an automated warning alert is flagged and logged to Prometheus, raising a trigger flag for the offline retraining pipeline.

### Live Demonstration Verification (Simulated Darkening Stream)

```
[Phase 1: In-Distribution Clean Stream]
- Window Sample Size: 40
- Brightness Window Mean: 127.4 vs Baseline: 127.4 (KS p-value: 0.4591)
- Contrast Window Mean:   24.4 vs Baseline: 26.8 (KS p-value: 0.1286)
-> Result: DRIFT DETECTED: FALSE | TRIGGER RETRAINING: FALSE

[Phase 2: Simulated Plant Lighting Degradation Stream]
- Window Sample Size: 40 (Simulated lamp failure & cast shadows: -30 to -40 intensity shift)
- Brightness Window Mean: 95.7 vs Baseline: 127.4 (KS p-value: 1.63e-4, Effect Size: 0.72)
- Alarm Reason: "Brightness drift: window mean=95.7 vs baseline=127.4 (p=1.63e-4, effect=0.72)"
-> Result: DRIFT DETECTED: TRUE | TRIGGER RETRAINING: TRUE (Alert Flag Raised)
```

*Note on Augmentation vs Evaluation: `src/augment.py` was executed once offline to construct the static shifted-lighting test benchmark (`data/processed/neu-det-shifted/`). Training-time data augmentation operates dynamically within PyTorch/Ultralytics transforms on clean images to prevent data leakage.*

---

## Unsupervised Anomaly Baseline (PatchCore vs Supervised YOLO)

To explore scenarios with scarce or zero defect labels (e.g., commissioning a new alloy line), we implemented a prototype **PatchCore-style memory bank baseline** (`src/anomaly/patchcore.py`).

### Data Provenance & Prototype Evaluation
- **Training Set:** Trained on 50 defect-free steel surface patches harvested from non-defect background regions of NEU-DET (zero spatial overlap with ground-truth defect annotations).
- **Prototype AUROC:** Evaluated on 20 defect-free background patches vs 30 multi-class test defect images, yielding an image-level anomaly detection AUROC of 99.7% [95% CI: 98.5% - 100.0%].
- **Methodological Limitation:** Because normal patches were harvested as background crops from the same collection of images, this score represents an optimistic upper bound. A definitive evaluation requires un-cropped normal coils from cold-rolling mill commissioning to eliminate crop-size bias.
- **Measured CPU Latency:** **8.3 ms mean / 8.4 ms p95** on Intel Xeon Gold 6242R CPU ($200 \times 200$, batch=1).

### Comparative Architectural Analysis

| Attribute | Supervised YOLOv8 (Selected Model) | Unsupervised PatchCore (Baseline Prototype) |
|:---|:---|:---|
| **Supervision Level** | Full supervision (requires labeled bounding boxes for all 6 defect types). | One-class / Unsupervised (trained solely on normal "golden" steel). |
| **Defect Taxonomy** | Explicitly classifies 6 classes (`crazing`, `patches`, `scratches`, etc.). | Binary out-of-distribution anomaly score; no taxonomy labeling. |
| **Zero-Shot Novelty** | Fails on novel, unseen metallurgical defect classes. | **High zero-shot capability**; flags unexpected defect morphologies. |
| **Accuracy Metric** | **68.3% mAP50** (640px FP32) / **65.7% mAP50** (640px QDQ INT8) [75.4% .pt model; 63.8% at conf=0.25]. | Preliminary **99.7% AUROC** on prototype normal vs defect patches. |
| **Localization Output** | Tight bounding box coordinates $[x_1, y_1, x_2, y_2]$. | Continuous pixel-level distance heatmap. |
| **Edge CPU Latency** | **75.0 ms** (Selected 640px QDQ) / **7.4 ms** (Edge 224px INT8). | **8.4 ms** (ResNet18 feature extraction + nearest neighbor). |
| **Industrial Decision** | **Use on mature production lines** with established defect catalogs. | **Use on new alloy commissioning** or scarce failure scenarios. |

---

## Automated Metallurgical & LLM Reporting

### Role of the LLM in This Architecture
The LLM is **not used for computer vision or pixel-level defect detection**—YOLOv8 and ONNX Runtime handle 100% of the visual feature extraction, localization, and classification at edge latency (7.4ms – 75.0ms on CPU). 

Instead, the LLM functions strictly as an **optional downstream post-detection translation engine** at the `POST /report` endpoint:
1. **Bridging Machine Detections to Plant Operations:** Plant operators and floor supervisors do not consume raw coordinate arrays (`[{"class": "patches", "conf": 0.84, "box": [...]}]`). The LLM translates structured detections into example disposition rules (`PASS`, `HOLD / DOWNGRADE`, `REJECT`) defined in this project (not a plant standard) and readable shift handoff memos.
2. **Domain-Specific Root-Cause Hypotheses:** Example hypotheses an operator would need to verify, such as crazing linked to roll cooling issues and rolled-in scale linked to descaling pressure. These mappings are illustrative and were not validated by a domain expert.
3. **Air-Gapped Offline Fallback:** Because hot rolling mills are frequently air-gapped without external cloud connectivity, the service does not hard-depend on external LLMs. A fast, local **deterministic metallurgical expert rule system** (`generate_deterministic_report()`) executes with a measured mean latency of **0.004 ms (3.7 microseconds)** over 1,000 calls (0.52 ms total postprocess in end-to-end inference) completely offline with zero API dependencies, while external LLM synthesis (`LLM_MODEL_NAME`, defaulting to `claude-haiku-5-5`) is available as a configurable enhancement.

### Sample A: LLM-Generated Inspection Report Schema (`LLM_MODEL_NAME=claude-haiku-5-5`)

*Note on LLM Integration: The payload below represents the JSON inspection schema produced by the cloud LLM synthesis layer (`POST /report` when `ANTHROPIC_API_KEY` is configured in the environment). In air-gapped hot strip mill deployments without external cloud access, the service automatically falls back to the local deterministic rule engine shown in Sample B with zero runtime overhead.*

```json
{
  "timestamp": "2026-10-22 02:35:10 UTC",
  "overall_status": "HOLD / DOWNGRADE",
  "risk_level": "ELEVATED",
  "defect_count": 2,
  "summary": "Automated visual inspection identified two contiguous patches defects on the middle strip section. Surface tears indicate mechanical galling from transfer table roller friction during finish rolling passes.",
  "root_cause_hypothesis": "Hypothesis: Work roll surface contamination or guide plate cold-welding adhesion in finishing mill stands. (Requires physical verification by mill operators).",
  "disposition_recommendation": "Quarantine coil segment. Divert downstream processing from Class A automotive surface panels to Grade B structural framing applications. Clean conveyor apron plates.",
  "source": "llm_engine (claude-haiku-5-5)"
}
```

### Sample B: Deterministic Metallurgical Expert System Report (Regenerated with Selected 640px QDQ Model)

```json
{
  "timestamp": "2026-10-21 17:32:16 UTC",
  "overall_status": "HOLD / DOWNGRADE",
  "risk_level": "ELEVATED",
  "defect_count": 1,
  "summary": "Inspection completed with 1 defect(s) detected across 1 defect category(ies). Overall Coil Disposition: [HOLD / DOWNGRADE]. Dominant defects identified: patches (x1).",
  "disposition": "HOLD / DOWNGRADE FOR SECONDARY PROCESSING",
  "defects_breakdown": [
    {
      "class": "patches",
      "count": 1,
      "severity": "MODERATE",
      "description": "Irregular localized surface indentations or partial thickness tears.",
      "consequence": "Dimensional gauge inconsistency and poor visual aesthetic finish."
    }
  ],
  "probable_root_causes": [
    "Patches: Foreign particulate matter on intermediate transfer tables or localized cold-welding adhesion on guide plates."
  ],
  "action_items": [
    "DOWNGRADE to Grade B/Structural framing. Clean conveyor apron plates."
  ],
  "source": "deterministic_metallurgical_rules",
  "timing_ms": {
    "total": 97.42,
    "preprocess": 8.22,
    "inference": 88.68,
    "postprocess": 0.52
  },
  "model_used": "models/onnx/model_qdq_int8.onnx (640px Static QDQ INT8)"
}
```

*Note on Sample B Timings: The latency breakdown recorded above (97.42 ms total, 88.68 ms inference, 0.52 ms postprocess) represents a single un-cached, end-to-end HTTP API request pass on an unbatched test image through the FastAPI microservice, including HTTP multipart decoding, image tensor normalization, and JSON serialization. It is distinct from the isolated, warmed-up ONNX Runtime engine benchmark reported in the CPU table (75.0 ms p95).*

---

## Serving API Specification

The FastAPI microservice provides production endpoints with OpenAPI documentation at `http://localhost:8000/docs`:

- `GET /health`: Liveness and readiness probe for Kubernetes. Returns model status, memory, and uptime.
- `POST /predict`: Upload image multipart form. Returns bounding box coordinates, confidences, execution timings, and streaming drift status.
- `POST /report`: Upload image. Returns comprehensive metallurgical QA inspection report.
- `GET /drift`: Returns live Kolmogorov-Smirnov statistics, p-values, and automated retraining flag.
- `GET /metrics`: Prometheus plain-text operational metrics (`inspection_requests_total`, `inspection_defects_detected_total`, `inspection_drift_retrain_trigger`).

---

## Containerization & Kubernetes Deployment

The service is containerized with Docker and includes Kubernetes manifests under `k8s/` for deployment to local clusters (`kind` or `minikube`, requiring local Docker daemon privileges).

### Multi-Stage Dockerfile (Powered by uv)
- Minimal `python:3.11-slim` runtime image.
- Astral `uv` binary copied directly for ultra-fast deterministic installation.
- Non-root user execution (`appuser:appgroup`, UID 10001).
- Built-in container healthcheck querying `/health`.

### Kubernetes Cluster Manifests
- `k8s/deployment.yaml`: 2-pod deployment with rolling updates, resource limits ($1500\text{m CPU}, 1.5\text{GB RAM}$), and HTTP liveness/readiness probes against `/health`.
- `k8s/service.yaml`: ClusterIP service exposing port 80.
- `k8s/configmap.yaml`: Environmental configurations and confidence threshold (CONF_THRESHOLD=0.28).
- `k8s/hpa.yaml`: Horizontal Pod Autoscaler scaling from 2 to 8 replicas at 70% CPU utilization.

```bash
# Deploy to local kind or minikube cluster
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify pod status
kubectl get pods -l app=defect-inspection
```

---

## When ML is NOT the Right Answer

In industrial quality engineering, deploying deep learning when simpler methods suffice introduces unnecessary latency, maintenance debt, and failure modes.

| Operational Scenario | Why ML / Deep Learning is the WRONG Choice | Recommended Deterministic / Non-ML Alternative |
|:---|:---|:---|
| **Gross Geometric Dimensional Inspection** | Deep learning bounding boxes have pixel variance ($\pm 2-5\text{ px}$). High inference latency ($5-20\text{ ms}$). | **Calibrated Optical Telecentric Profilometry:** Laser triangulation or sub-pixel edge detection ($\pm 0.01\text{ mm}$ at $1,000\text{ FPS}$). |
| **Gross Strip Hole / Edge Tear Detection** | Deep neural network is severe over-engineering for high-contrast voids. | **Direct Backlit Photodiode Sensor Bar:** High-speed binary thresholding ($<0.1\text{ ms}$ reaction time). |
| **Strict Zero-Defect Guarantees** | Deep learning models produce probabilistic outputs with non-zero residual false negative rates. | **Deterministic Eddy-Current / Ultrasonic Non-Destructive Testing (NDT):** Physical volumetric testing. |
| **Sub-Millisecond Sorting Gates** | Neural network forward pass ($>5\text{ ms}$) exceeds mechanical solenoid air-ejector response windows. | **Hardware FPGA Fixed-Kernel 2D Convolutions:** Hardwired morphological filters executing in $<50\mu\text{s}$. |

---

## Quickstart & Reproduction Guide (`uv` Native)

### 1. Environment Setup via `uv`
```bash
# Sync dependencies and development tools directly from pyproject.toml & uv.lock
uv sync --all-groups
```

### 2. Data Preparation & Stratified Split
```bash
uv run python data/convert_and_split.py
uv run python src/augment.py
```

### 3. Edge Export & Benchmarking
```bash
uv run python src/export.py --weights models/champion_best.pt --imgsz 640
```

### 4. Drift Monitoring Demo
```bash
uv run python src/monitor/drift.py --demo
```

### 5. Anomaly Baseline Evaluation
```bash
uv run python src/anomaly/patchcore.py --eval
```

### 6. Run Test Suite
```bash
uv run pytest -v tests/
```

### 7. Launch Serving API
```bash
uv run uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
```
