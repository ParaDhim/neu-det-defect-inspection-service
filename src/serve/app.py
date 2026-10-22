"""
FastAPI Production Serving Application for Steel Surface Defect Inspection.
Provides:
- GET /health: Kubernetes liveness & readiness probe endpoint
- POST /predict: High-throughput ONNX Runtime inference & confidence logging
- POST /report: Metallurgical domain & LLM-assisted quality report generation
- GET /drift: Real-time Kolmogorov-Smirnov distribution drift monitoring status
- GET /metrics: Prometheus-compatible operational metrics
"""

import os
import io
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, List
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .predictor import ONNXPredictor
from .report import generate_llm_inspection_report
from src.monitor.drift import DriftMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("defect-inspection-service")

MODEL_PATH = os.getenv("MODEL_PATH", "models/onnx/model_edge_fp32.onnx")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.22"))
# Per-class overrides tuned on the validation split (crazing needs a lower threshold), JSON object.
CLASS_CONF_THRESHOLDS = json.loads(os.getenv("CLASS_CONF_THRESHOLDS", '{"crazing": 0.112}'))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
BASELINE_PATH = os.getenv("BASELINE_PATH", "src/monitor/baseline_stats.json")

# Operational state
start_time = time.time()
REQUEST_COUNT = 0
TOTAL_DETECTIONS = 0

# Lazy / eager initializers
_predictor: Optional[ONNXPredictor] = None
_drift_monitor: Optional[DriftMonitor] = None


def get_predictor() -> ONNXPredictor:
    """Returns singleton ONNXPredictor instance, initializing if necessary."""
    global _predictor
    if _predictor is None:
        logger.info(f"Initializing ONNXPredictor with model: {MODEL_PATH}")
        _predictor = ONNXPredictor(
            model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            class_conf_thresholds=CLASS_CONF_THRESHOLDS,
            iou_threshold=IOU_THRESHOLD,
        )
    return _predictor


def get_drift_monitor() -> DriftMonitor:
    """Returns singleton DriftMonitor instance, initializing if necessary."""
    global _drift_monitor
    if _drift_monitor is None:
        logger.info(f"Initializing DriftMonitor with baseline: {BASELINE_PATH}")
        _drift_monitor = DriftMonitor(baseline_path=BASELINE_PATH, window_size=50)
    return _drift_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup and shutdown resource lifecycle."""
    get_predictor()
    get_drift_monitor()
    logger.info("Inspection service initialized and ready to serve requests.")
    yield
    logger.info("Inspection service shutting down.")


app = FastAPI(
    title="NEU-DET Steel Surface Defect Inspection Service",
    description="End-to-end industrial computer vision service with ONNX Runtime, drift monitoring, and metallurgical reporting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class PredictResponse(BaseModel):
    status: str
    count: int
    detections: List[DetectionItem]
    timing_ms: dict
    drift_status: dict


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Kubernetes Liveness and Readiness Probe Endpoint.
    Verifies service health, model loading status, and runtime uptime.
    """
    uptime_sec = round(time.time() - start_time, 2)
    pred = get_predictor()
    drift = get_drift_monitor()
    model_loaded = pred.session is not None
    drift_ready = drift.baseline is not None

    status = "healthy" if model_loaded or os.path.exists("models") else "degraded"

    return {
        "status": status,
        "uptime_seconds": uptime_sec,
        "model": {
            "loaded": model_loaded,
            "path": MODEL_PATH,
            "confidence_threshold": CONF_THRESHOLD,
            "class_confidence_thresholds": CLASS_CONF_THRESHOLDS,
            "iou_threshold": IOU_THRESHOLD,
        },
        "drift_monitor": {
            "initialized": drift_ready,
            "window_size": len(drift.window),
            "total_observed": drift.total_observed,
        },
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_defect(
    file: UploadFile = File(..., description="Steel surface image file (JPEG/PNG)"),
    confidence_override: Optional[float] = Form(None),
):
    """
    Executes defect detection inference using ONNX Runtime.
    Logs request confidences and updates streaming drift monitoring.
    """
    global REQUEST_COUNT, TOTAL_DETECTIONS
    REQUEST_COUNT += 1

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image encoding.")

    pred = get_predictor()
    drift = get_drift_monitor()

    orig_conf = pred.conf_threshold
    if confidence_override is not None and 0.0 < confidence_override < 1.0:
        pred.conf_threshold = confidence_override

    result = pred.predict(image)
    pred.conf_threshold = orig_conf

    drift.record_image(image)
    drift_report = drift.check_drift()

    detections = result.get("detections", [])
    TOTAL_DETECTIONS += len(detections)

    for d in detections:
        logger.info(
            f"Prediction | Class: {d['class_name']} | Confidence: {d['confidence']:.4f} | BBox: {d['bbox']}"
        )

    return {
        "status": "success",
        "count": len(detections),
        "detections": detections,
        "timing_ms": result.get("timing_ms", {}),
        "drift_status": {
            "drift_detected": drift_report.get("drift_detected", False),
            "trigger_retraining": drift_report.get("trigger_retraining", False),
            "samples_in_window": len(drift.window),
        },
    }


@app.post("/report", tags=["Reporting"])
async def generate_report(
    file: UploadFile = File(..., description="Steel surface image file for full QA report"),
):
    """
    Runs defect detection and synthesizes an actionable metallurgical inspection report
    with severity rating, root cause hypothesis, and coil disposition instructions.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pred = get_predictor()
    drift = get_drift_monitor()

    pred_res = pred.predict(image)
    detections = pred_res.get("detections", [])

    drift.record_image(image)
    report = generate_llm_inspection_report(detections)
    report["timing_ms"] = pred_res.get("timing_ms", {})
    return report


@app.get("/drift", tags=["Monitoring"])
def get_drift_status():
    """Returns detailed Kolmogorov-Smirnov statistical drift test status and retraining trigger flag."""
    return get_drift_monitor().check_drift()


@app.get("/metrics", tags=["Monitoring"])
def prometheus_metrics():
    """Prometheus-compatible plain text operational metrics."""
    uptime = time.time() - start_time
    d_status = get_drift_monitor().check_drift()
    trigger_val = 1 if d_status.get("trigger_retraining", False) else 0

    lines = [
        "# HELP inspection_requests_total Total number of defect inspection requests served",
        "# TYPE inspection_requests_total counter",
        f"inspection_requests_total {REQUEST_COUNT}",
        "# HELP inspection_defects_detected_total Total bounding boxes detected",
        "# TYPE inspection_defects_detected_total counter",
        f"inspection_defects_detected_total {TOTAL_DETECTIONS}",
        "# HELP inspection_uptime_seconds Process uptime in seconds",
        "# TYPE inspection_uptime_seconds gauge",
        f"inspection_uptime_seconds {uptime:.1f}",
        "# HELP inspection_drift_retrain_trigger Retraining flag (1 if drift requires retraining, else 0)",
        "# TYPE inspection_drift_retrain_trigger gauge",
        f"inspection_drift_retrain_trigger {trigger_val}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
