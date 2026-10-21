"""
Tests for industrial lighting shift and physical perturbation transformations.
"""

import os
import glob
import numpy as np
import pytest
from src.augment import (
    apply_brightness_contrast,
    apply_gamma,
    apply_gaussian_noise,
    apply_blur,
    apply_glare_hotspot,
    apply_cast_shadow,
    apply_composite_lighting_shift,
)


@pytest.fixture
def sample_image():
    """Generates a standard 200x200 3-channel test image."""
    return np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)


def test_brightness_contrast_shape_and_dtype(sample_image):
    out = apply_brightness_contrast(sample_image, alpha=1.2, beta=20)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    assert 0 <= out.min() <= 255
    assert 0 <= out.max() <= 255


def test_gamma_correction(sample_image):
    out = apply_gamma(sample_image, gamma=0.7)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_gaussian_noise(sample_image):
    out = apply_gaussian_noise(sample_image, std=15)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    # Noise should alter values
    assert not np.array_equal(out, sample_image)


def test_blur(sample_image):
    out = apply_blur(sample_image, ksize=5)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_glare_hotspot(sample_image):
    out = apply_glare_hotspot(sample_image, radius=60, intensity=0.5)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    assert out.mean() >= sample_image.mean()


def test_cast_shadow(sample_image):
    out = apply_cast_shadow(sample_image, intensity=0.4)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    assert out.mean() <= sample_image.mean()


def test_composite_lighting_shift(sample_image):
    out = apply_composite_lighting_shift(sample_image, seed=42)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, sample_image)


def test_shifted_dataset_exists():
    """Verifies shifted benchmark dataset files exist and match test split."""
    shifted_dir = "data/processed/neu-det-shifted"
    assert os.path.exists(shifted_dir)
    assert os.path.exists(os.path.join(shifted_dir, "dataset_shifted.yaml"))
    test_imgs = glob.glob(os.path.join(shifted_dir, "images", "test", "*.jpg"))
    test_lbls = glob.glob(os.path.join(shifted_dir, "labels", "test", "*.txt"))
    assert len(test_imgs) > 0
    assert len(test_imgs) == len(test_lbls)
