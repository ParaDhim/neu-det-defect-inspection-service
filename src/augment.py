"""
Lighting-shift and factory-environment perturbation generator for steel defect inspection.
Simulates real-world industrial metallurgical plant variations:
- Non-uniform illumination (shadows, hotspots/glare)
- Sensor noise & thermal grain
- Low-contrast strip lighting and overexposure
- Conveyor vibration blur
"""

import os
import glob
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path


def apply_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """Adjusts contrast (alpha) and brightness (beta)."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Non-linear gamma correction to simulate sensor sensitivity changes."""
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    """Simulates camera sensor thermal noise."""
    gauss = np.random.normal(mean, std, image.shape).astype("float32")
    noisy = np.clip(image.astype("float32") + gauss, 0, 255).astype("uint8")
    return noisy


def apply_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Simulates production line optical defocus or motion blur."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_glare_hotspot(image: np.ndarray, center=None, radius: int = 80, intensity: float = 0.5) -> np.ndarray:
    """Simulates specular reflection / hot spots from high-intensity halogen factory inspection lamps."""
    h, w = image.shape[:2]
    if center is None:
        center = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = np.clip(1.0 - (dist_from_center / float(radius)), 0, 1)
    mask = mask ** 2  # quadratic decay

    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=2)

    glare = (255 * mask * intensity).astype("float32")
    output = np.clip(image.astype("float32") + glare, 0, 255).astype("uint8")
    return output


def apply_cast_shadow(image: np.ndarray, intensity: float = 0.45) -> np.ndarray:
    """Simulates machinery, gantry, or roller occlusion shadows."""
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype="float32")

    # Generate random linear or polygonal shadow zone
    pt1 = (np.random.randint(0, w), 0)
    pt2 = (np.random.randint(0, w), h)
    poly = np.array([[0, 0], [pt1[0], pt1[1]], [pt2[0], pt2[1]], [0, h]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 1.0 - intensity)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=2)

    output = np.clip(image.astype("float32") * mask, 0, 255).astype("uint8")
    return output


def apply_composite_lighting_shift(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Applies a combined physical perturbation representing harsh factory environment shifts.
    Combines:
    1. Severe brightness & contrast drop/boost
    2. Optical blur
    3. Sensor noise
    4. Glare or Cast shadow
    """
    if seed is not None:
        np.random.seed(seed)

    out = image.copy()

    # 1. Random photometric shift: severe under-exposure or over-exposure
    mode = np.random.choice(["dark", "bright", "low_contrast", "glare_heavy"])
    if mode == "dark":
        # -40% brightness, reduced contrast
        out = apply_brightness_contrast(out, alpha=np.random.uniform(0.55, 0.75), beta=int(np.random.uniform(-45, -20)))
        out = apply_gamma(out, gamma=np.random.uniform(0.6, 0.8))
    elif mode == "bright":
        # Over-exposure
        out = apply_brightness_contrast(out, alpha=np.random.uniform(1.15, 1.35), beta=int(np.random.uniform(25, 45)))
        out = apply_gamma(out, gamma=np.random.uniform(1.2, 1.5))
    elif mode == "low_contrast":
        out = apply_brightness_contrast(out, alpha=np.random.uniform(0.45, 0.65), beta=int(np.random.uniform(-10, 10)))
    elif mode == "glare_heavy":
        out = apply_glare_hotspot(out, radius=np.random.randint(60, 100), intensity=np.random.uniform(0.4, 0.7))

    # 2. Add realistic shadow on 50% of samples
    if np.random.rand() > 0.4:
        out = apply_cast_shadow(out, intensity=np.random.uniform(0.35, 0.55))

    # 3. Add sensor noise
    if np.random.rand() > 0.3:
        out = apply_gaussian_noise(out, std=np.random.uniform(10, 22))

    # 4. Add slight vibration blur
    if np.random.rand() > 0.4:
        k = np.random.choice([3, 5])
        out = apply_blur(out, ksize=k)

    return out


def generate_shifted_test_dataset(
    source_dir: str = "data/processed/neu-det",
    output_dir: str = "data/processed/neu-det-shifted",
    seed: int = 100,
):
    """
    Creates an official shifted evaluation benchmark from the test split.
    Keeps bounding box annotations identical while transforming image pixels.
    """
    np.random.seed(seed)
    src_img_dir = os.path.join(source_dir, "images", "test")
    src_lbl_dir = os.path.join(source_dir, "labels", "test")

    dst_img_dir = os.path.join(output_dir, "images", "test")
    dst_lbl_dir = os.path.join(output_dir, "labels", "test")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    test_imgs = sorted(glob.glob(os.path.join(src_img_dir, "*.jpg")))
    print(f"Generating shifted benchmark for {len(test_imgs)} test images...")

    for idx, img_path in enumerate(test_imgs):
        stem = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Perturb image
        shifted_img = apply_composite_lighting_shift(img, seed=seed + idx)

        # Save perturbed image
        out_img_path = os.path.join(dst_img_dir, f"{stem}.jpg")
        cv2.imwrite(out_img_path, shifted_img)

        # Copy matching label file
        src_lbl = os.path.join(src_lbl_dir, f"{stem}.txt")
        dst_lbl = os.path.join(dst_lbl_dir, f"{stem}.txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    # Copy val split so YOLO evaluation can run test split as val
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    for f in glob.glob(os.path.join(dst_img_dir, "*.jpg")):
        shutil.copy2(f, os.path.join(output_dir, "images", "val", os.path.basename(f)))
    for f in glob.glob(os.path.join(dst_lbl_dir, "*.txt")):
        shutil.copy2(f, os.path.join(output_dir, "labels", "val", os.path.basename(f)))

    # Write dataset YAML
    abs_out = os.path.abspath(output_dir)
    yaml_content = f"""# NEU-DET Shifted Evaluation Benchmark (Lighting, Noise, Blur, Glare, Shadow)
path: {abs_out}
train: images/test
val: images/test
test: images/test

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    with open(os.path.join(output_dir, "dataset_shifted.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"Successfully generated shifted test benchmark at {output_dir}")
    print(f"Total shifted images: {len(test_imgs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lighting-shifted test set")
    parser.add_argument("--source", type=str, default="data/processed/neu-det", help="Source clean dataset dir")
    parser.add_argument("--output", type=str, default="data/processed/neu-det-shifted", help="Output shifted dataset dir")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    args = parser.parse_args()

    generate_shifted_test_dataset(args.source, args.output, args.seed)
