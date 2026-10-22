# Unified Experiment Leaderboard

Single protocol: conf=0.001, rect=False, native resolution (scripts/eval_all_runs.py). Select on Val mAP50 only. Runs 01-09 retrained (seed 0); Runs 10/11 original weights. Multi-seed results: reports/seed_eval_conf001.json.

| Run ID | Arch | Res | Augmentation / optimizer | **Val mAP50** | Test mAP50 | Shifted test mAP50 | Test mAP50-95 | Weights |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| `01_baseline_yolov8n_imgsz200` | YOLOv8n | 200 | Ultralytics default (HSV/mosaic/scale off) | **70.3%** | 71.7% | 25.9% | 40.0% | retrained |
| `02_baseline_yolov8n_imgsz320` | YOLOv8n | 320 | High-res clean baseline | **69.5%** | 71.5% | 23.4% | 39.8% | retrained |
| `03_robust_yolov8n_light_aug` | YOLOv8n | 224 | Light aug (HSV v=0.2, mosaic 0.5) | **68.9%** | 70.9% | 24.1% | 34.9% | retrained |
| `04_robust_yolov8n_heavy_aug` | YOLOv8n | 224 | Heavy aug (HSV v=0.6, scale, erasing) | **66.0%** | 63.9% | 18.1% | 25.2% | retrained |
| `05_robust_yolov8s_heavy_aug` | YOLOv8s | 224 | Capacity scaling + heavy aug | **66.4%** | 65.3% | 23.4% | 27.0% | retrained |
| `06_robust_yolov8n_adamw_cosine` | YOLOv8n | 224 | AdamW lr 0.002, medium aug | **67.5%** | 67.1% | 23.9% | 34.9% | retrained |
| `07_robust_yolov8n_sgd_momentum` | YOLOv8n | 224 | SGD lr 0.01, medium aug | **66.6%** | 66.4% | 34.3% | 33.5% | retrained |
| `08_robust_yolov8n_geom_heavy` | YOLOv8n | 224 | Heavy geometric aug (rot 30) | **56.4%** | 57.6% | 15.3% | 22.3% | retrained |
| `09_robust_yolov8n_imgsz320_heavy` | YOLOv8n | 320 | 320px + heavy aug | **63.5%** | 62.7% | 16.8% | 26.4% | retrained |
| `10_champion_yolov8n_production` | YOLOv8n | 224 | AdamW + tuned photometric aug (12 ep) | **58.0%** | 61.7% | 19.9% | 22.7% | original |
| `11_yolov8s_640_heavy_aug` | YOLOv8s | 640 | AdamW + cosine LR + CopyPaste + scale (80 ep) | **70.4%** | 68.3% | 24.1% | 30.3% | original |
