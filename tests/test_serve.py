"""
Tests for FastAPI service endpoints (/health, /predict, /report, /drift, /metrics).
"""

import io
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient
from src.serve.app import app


@pytest.fixture
def client():
    """Initializes TestClient context for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def encoded_image_bytes():
    """Creates in-memory JPEG bytes of a test image."""
    img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (120, 100), (30, 30, 30), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_endpoint(client):
    """Verify /health returns 200 and expected schema."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "model" in data
    assert "drift_monitor" in data


def test_predict_endpoint_valid_image(client, encoded_image_bytes):
    """Verify /predict processes image and returns predictions."""
    files = {"file": ("test.jpg", encoded_image_bytes, "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "count" in data
    assert "detections" in data
    assert "timing_ms" in data
    assert "drift_status" in data


def test_predict_endpoint_invalid_file(client):
    """Verify /predict rejects non-image payload with HTTP 400."""
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_report_endpoint(client, encoded_image_bytes):
    """Verify /report generates structured metallurgical engineering summary."""
    files = {"file": ("test.jpg", encoded_image_bytes, "image/jpeg")}
    resp = client.post("/report", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_status" in data
    assert "summary" in data
    assert "disposition" in data
    assert data["overall_status"] in ["PASS", "PASS WITH ADVISORY", "HOLD / DOWNGRADE", "REJECT"]


def test_drift_endpoint(client):
    """Verify /drift returns current drift monitor report."""
    resp = client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_metrics_endpoint(client):
    """Verify /metrics returns Prometheus formatted plain text."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "inspection_requests_total" in resp.text
    assert "inspection_uptime_seconds" in resp.text
