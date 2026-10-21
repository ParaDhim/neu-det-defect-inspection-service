"""
Tests for PatchCore unsupervised anomaly detector and patch feature extractor.
"""

import numpy as np
import pytest
import torch
from src.anomaly.patchcore import PatchExtractor, PatchCoreAnomalyDetector


def test_patch_extractor_output():
    """Verify ResNet feature extractor produces normalized L2 embeddings."""
    extractor = PatchExtractor(device="cpu")
    dummy = torch.randn(1, 3, 200, 200)
    features = extractor(dummy)

    # Features shape should be (1, 384, H', W')
    assert features.shape[0] == 1
    assert features.shape[1] == 384  # 128 (layer2) + 256 (layer3)
    # Check L2 norm along channel dimension is approximately 1.0
    norms = torch.norm(features, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_anomaly_detector_synthetic():
    """Verify PatchCore anomaly detector fits memory bank and scores an image."""
    detector = PatchCoreAnomalyDetector(device="cpu")

    # Synthetic normal images
    normal_patches = torch.randn(50, 384)
    # Normalize
    normal_patches = normal_patches / torch.norm(normal_patches, p=2, dim=1, keepdim=True)
    detector.memory_bank = normal_patches

    test_img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    score, heatmap = detector.score_image(test_img)

    assert isinstance(score, float)
    assert score >= 0.0
    assert heatmap.shape == (200, 200)
    assert 0.0 <= heatmap.min() <= 1.0
    assert 0.0 <= heatmap.max() <= 1.0
