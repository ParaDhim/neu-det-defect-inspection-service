"""
Data conversion and stratified splitting for NEU-DET defect inspection dataset.
Converts Pascal VOC XML bounding box annotations to YOLO format normalized coordinates,
performs a seeded class-stratified split (80% train, 10% val, 10% test),
generates YOLO dataset configuration YAML, and calculates class distribution statistics.
"""

import os
import glob
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# Canonical NEU-DET defect classes
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def parse_voc_xml(xml_path: str):
    """Parse Pascal VOC XML annotation file and extract normalized bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        width = float(size_elem.find("width").text)
        height = float(size_elem.find("height").text)
    else:
        width = 200.0
        height = 200.0

    boxes = []
    class_names = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_TO_IDX:
            # Handle possible name variants
            name_fixed = name.replace(" ", "_").replace("-", "_")
            if name_fixed in CLASS_TO_IDX:
                name = name_fixed
            elif "crazing" in name:
                name = "crazing"
            elif "inclusion" in name:
                name = "inclusion"
            elif "patch" in name:
                name = "patches"
            elif "pitted" in name:
                name = "pitted_surface"
            elif "rolled" in name or "scale" in name:
                name = "rolled-in_scale"
            elif "scratch" in name:
                name = "scratches"
            else:
                continue

        cls_id = CLASS_TO_IDX[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clamp to image boundaries
        xmin = max(0.0, min(xmin, width))
        ymin = max(0.0, min(ymin, height))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue

        # Convert to YOLO format (center_x, center_y, width, height) normalized to [0, 1]
        x_center = (xmin + bw / 2.0) / width
        y_center = (ymin + bh / 2.0) / height
        norm_w = bw / width
        norm_h = bh / height

        # Ensure normalized coordinates are valid [0, 1]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        boxes.append((cls_id, x_center, y_center, norm_w, norm_h))
        class_names.append(name)

    return boxes, class_names


def perform_stratified_split(
    raw_img_dir: str = "data/raw/images",
    raw_ann_dir: str = "data/raw/annotations",
    output_dir: str = "data/processed/neu-det",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Performs seeded stratified train/val/test splitting across the 6 defect categories.
    Copies images and writes YOLO txt label files.
    """
    np.random.seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    xml_files = sorted(glob.glob(os.path.join(raw_ann_dir, "*.xml")))
    print(f"Found {len(xml_files)} annotation files in {raw_ann_dir}")

    # Group images by their dominant class (for stratification)
    class_to_items = defaultdict(list)
    item_metadata = {}

    for xml_path in xml_files:
        stem = Path(xml_path).stem
        img_path = os.path.join(raw_img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            # Check without extension or case-insensitive
            candidates = glob.glob(os.path.join(raw_img_dir, f"{stem}.*"))
            if candidates:
                img_path = candidates[0]
            else:
                continue

        boxes, classes_in_img = parse_voc_xml(xml_path)
        if not classes_in_img:
            # Fallback to class prefix from filename, e.g. crazing_1.jpg
            for c in CLASSES:
                if stem.startswith(c) or stem.startswith(c.replace("-", "_")):
                    primary_class = c
                    break
            else:
                primary_class = "inclusion"
        else:
            primary_class = Counter(classes_in_img).most_common(1)[0][0]

        class_to_items[primary_class].append(stem)
        item_metadata[stem] = {
            "img_path": img_path,
            "xml_path": xml_path,
            "boxes": boxes,
            "classes": classes_in_img,
        }

    splits = {"train": [], "val": [], "test": []}

    # Stratify by primary class
    for cls_name, stems in class_to_items.items():
        stems_shuffled = np.random.permutation(sorted(stems)).tolist()
        n = len(stems_shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure test gets the rest
        train_stems = stems_shuffled[:n_train]
        val_stems = stems_shuffled[n_train : n_train + n_val]
        test_stems = stems_shuffled[n_train + n_val :]

        splits["train"].extend(train_stems)
        splits["val"].extend(val_stems)
        splits["test"].extend(test_stems)

    # Create destination directories
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split_name), exist_ok=True)

    split_stats = {
        "seed": seed,
        "classes": CLASSES,
        "split_counts": {},
        "class_counts_per_split": {
            "images": {c: {} for c in CLASSES},
            "boxes": {c: {} for c in CLASSES},
        },
        "total_boxes_per_split": {},
    }

    for split_name, stems in splits.items():
        split_stats["split_counts"][split_name] = len(stems)
        boxes_per_class_split = Counter()
        images_per_class_split = Counter()

        for stem in stems:
            meta = item_metadata[stem]
            dest_img = os.path.join(output_dir, "images", split_name, f"{stem}.jpg")
            dest_txt = os.path.join(output_dir, "labels", split_name, f"{stem}.txt")

            # Copy image
            shutil.copy2(meta["img_path"], dest_img)

            # Write YOLO labels
            with open(dest_txt, "w") as f:
                for cls_id, xc, yc, w, h in meta["boxes"]:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    boxes_per_class_split[CLASSES[cls_id]] += 1

            for c in set(meta["classes"]):
                images_per_class_split[c] += 1

        split_stats["total_boxes_per_split"][split_name] = sum(boxes_per_class_split.values())
        for c in CLASSES:
            split_stats["class_counts_per_split"]["images"][c][split_name] = images_per_class_split[c]
            split_stats["class_counts_per_split"]["boxes"][c][split_name] = boxes_per_class_split[c]

    # Write YOLO dataset.yaml
    dataset_yaml_path = os.path.join(output_dir, "dataset.yaml")
    abs_output_dir = os.path.abspath(output_dir)
    with open(dataset_yaml_path, "w") as f:
        f.write(f"# NEU-DET Surface Defect Inspection Dataset\n")
        f.write(f"path: {abs_output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"\nnames:\n")
        for idx, name in enumerate(CLASSES):
            f.write(f"  {idx}: {name}\n")

    # Write report json
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "class_distribution.json"), "w") as f:
        json.dump(split_stats, f, indent=2)

    with open("data/class_counts.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    # Write markdown table for README
    md_table = generate_markdown_report(split_stats)
    with open(os.path.join(reports_dir, "class_distribution.md"), "w") as f:
        f.write(md_table)

    print("Data conversion, split, and statistics completed successfully.")
    print(md_table)
    return split_stats


def generate_markdown_report(stats: dict) -> str:
    """Formats class distribution into a clean GitHub-flavored markdown table."""
    lines = [
        "### NEU-DET Dataset Class Distribution & Stratified Split",
        "",
        f"**Random Seed:** {stats['seed']} | **Total Images:** {sum(stats['split_counts'].values())} | **Total Boxes:** {sum(stats['total_boxes_per_split'].values())}",
        "",
        "| Class ID | Defect Class | Train Images | Val Images | Test Images | Total Images | Train Boxes | Val Boxes | Test Boxes | Total Boxes |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    tot_train_img = stats["split_counts"]["train"]
    tot_val_img = stats["split_counts"]["val"]
    tot_test_img = stats["split_counts"]["test"]
    tot_all_img = tot_train_img + tot_val_img + tot_test_img

    tot_train_box = stats["total_boxes_per_split"]["train"]
    tot_val_box = stats["total_boxes_per_split"]["val"]
    tot_test_box = stats["total_boxes_per_split"]["test"]
    tot_all_box = tot_train_box + tot_val_box + tot_test_box

    for idx, c in enumerate(stats["classes"]):
        tr_i = stats["class_counts_per_split"]["images"][c].get("train", 0)
        va_i = stats["class_counts_per_split"]["images"][c].get("val", 0)
        te_i = stats["class_counts_per_split"]["images"][c].get("test", 0)
        tot_i = tr_i + va_i + te_i

        tr_b = stats["class_counts_per_split"]["boxes"][c].get("train", 0)
        va_b = stats["class_counts_per_split"]["boxes"][c].get("val", 0)
        te_b = stats["class_counts_per_split"]["boxes"][c].get("test", 0)
        tot_b = tr_b + va_b + te_b

        lines.append(
            f"| {idx} | `{c}` | {tr_i} | {va_i} | {te_i} | {tot_i} | {tr_b} | {va_b} | {te_b} | {tot_b} |"
        )

    lines.append(
        f"| **Total** | **All Classes** | **{tot_train_img}** | **{tot_val_img}** | **{tot_test_img}** | **{tot_all_img}** | **{tot_train_box}** | **{tot_val_box}** | **{tot_test_box}** | **{tot_all_box}** |"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    perform_stratified_split()
