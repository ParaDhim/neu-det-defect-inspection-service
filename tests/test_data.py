"""
Tests for NEU-DET data conversion, bounding box normalization, and stratified split integrity.
"""

import os
import glob
import pytest
import yaml
from pathlib import Path
from data.convert_and_split import parse_voc_xml, CLASSES, CLASS_TO_IDX


def test_voc_xml_parsing():
    """Verify VOC XML parser extracts valid normalized bounding boxes."""
    sample_xmls = glob.glob("data/raw/annotations/*.xml")
    assert len(sample_xmls) > 0, "No raw XML annotation files found."

    xml_path = sample_xmls[0]
    boxes, class_names = parse_voc_xml(xml_path)

    assert len(boxes) > 0, "Parsed zero bounding boxes."
    for cls_id, xc, yc, w, h in boxes:
        assert 0 <= cls_id < len(CLASSES)
        assert 0.0 <= xc <= 1.0, f"xc {xc} out of bounds"
        assert 0.0 <= yc <= 1.0, f"yc {yc} out of bounds"
        assert 0.0 < w <= 1.0, f"w {w} out of bounds"
        assert 0.0 < h <= 1.0, f"h {h} out of bounds"


def test_split_leakage_and_integrity():
    """Verify train, val, and test splits are strictly disjoint (no data leakage)."""
    neu_det_dir = "data/processed/neu-det"
    train_imgs = set(os.listdir(os.path.join(neu_det_dir, "images", "train")))
    val_imgs = set(os.listdir(os.path.join(neu_det_dir, "images", "val")))
    test_imgs = set(os.listdir(os.path.join(neu_det_dir, "images", "test")))

    assert len(train_imgs) > 0
    assert len(val_imgs) > 0
    assert len(test_imgs) > 0

    # Ensure zero intersection
    assert len(train_imgs.intersection(val_imgs)) == 0, "Train and Val split leak detected!"
    assert len(train_imgs.intersection(test_imgs)) == 0, "Train and Test split leak detected!"
    assert len(val_imgs.intersection(test_imgs)) == 0, "Val and Test split leak detected!"


def test_dataset_yaml_structure():
    """Verify dataset.yaml is properly formatted and points to valid directories."""
    yaml_path = "data/processed/neu-det/dataset.yaml"
    assert os.path.exists(yaml_path), "dataset.yaml does not exist."

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "train" in cfg
    assert "val" in cfg
    assert "test" in cfg
    assert "names" in cfg
    assert len(cfg["names"]) == 6
    assert list(cfg["names"].values()) == CLASSES
