"""
Tests for drift monitoring, photometric metric extraction, and Kolmogorov-Smirnov test logic.
"""

import os
import numpy as np
import pytest
from src.monitor.drift import compute_image_metrics, DriftMonitor


@pytest.fixture
def dark_image():
    """Synthetic severely underexposed image."""
    return np.full((200, 200, 3), 30, dtype=np.uint8)


@pytest.fixture
def normal_image():
    """Synthetic normal exposure image with natural variance."""
    return np.random.normal(128, 25, (200, 200, 3)).clip(0, 255).astype(np.uint8)


def test_compute_image_metrics(dark_image, normal_image):
    m_dark = compute_image_metrics(dark_image)
    m_norm = compute_image_metrics(normal_image)

    assert "brightness" in m_dark
    assert "contrast" in m_dark
    assert "sharpness" in m_dark
    assert "entropy" in m_dark

    assert m_dark["brightness"] < m_norm["brightness"]
    assert m_dark["contrast"] < m_norm["contrast"]


def test_drift_monitor_warmup():
    monitor = DriftMonitor(baseline_path="src/monitor/baseline_stats.json", window_size=30)
    # 5 samples < 15 warmup threshold
    for _ in range(5):
        img = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)
        monitor.record_image(img)

    report = monitor.check_drift()
    assert report["status"] == "warming_up"
    assert report["drift_detected"] is False
    assert report["trigger_retraining"] is False


def test_drift_monitor_detects_darkening(dark_image):
    monitor = DriftMonitor(baseline_path="src/monitor/baseline_stats.json", window_size=25)
    # Stream 25 severely dark images
    for _ in range(25):
        monitor.record_image(dark_image)

    report = monitor.check_drift()
    assert report["status"] == "active"
    assert report["drift_detected"] is True
    assert report["trigger_retraining"] is True
    assert report["details"]["brightness"]["drift_flag"] is True
    assert report["details"]["brightness"]["p_value"] < 0.05
