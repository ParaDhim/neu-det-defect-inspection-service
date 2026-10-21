"""
Production Input Distribution & Concept Drift Monitor for Industrial Vision.
Tracks per-request image photometric statistics (brightness, contrast, sharpness, entropy),
compares against training baseline distributions via the two-sample Kolmogorov-Smirnov (KS) test,
and enforces an automated retraining trigger when significant covariate shift occurs.
"""

import os
import glob
import json
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import cv2
from scipy import stats


def compute_image_metrics(image: np.ndarray) -> dict:
    """
    Calculates essential photometric and structural image statistics.
    Input image can be BGR or Grayscale uint8 array.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Brightness: mean intensity [0, 255]
    brightness = float(np.mean(gray))

    # 2. Contrast: standard deviation of pixel intensities
    contrast = float(np.std(gray))

    # 3. Sharpness: Laplacian variance (focus/blur indicator)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # 4. Image Entropy: information content
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "entropy": round(entropy, 2),
    }


def build_baseline_distribution(
    train_images_dir: str = "data/processed/neu-det/images/train",
    save_path: str = "src/monitor/baseline_stats.json",
) -> dict:
    """
    Computes reference baseline empirical distribution from training images.
    """
    img_paths = sorted(glob.glob(os.path.join(train_images_dir, "*.jpg")))
    print(f"Building baseline reference stats from {len(img_paths)} training images...")

    brightness_vals = []
    contrast_vals = []
    sharpness_vals = []
    entropy_vals = []

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        m = compute_image_metrics(img)
        brightness_vals.append(m["brightness"])
        contrast_vals.append(m["contrast"])
        sharpness_vals.append(m["sharpness"])
        entropy_vals.append(m["entropy"])

    baseline = {
        "sample_size": len(brightness_vals),
        "summary": {
            "brightness": {
                "mean": float(np.mean(brightness_vals)),
                "std": float(np.std(brightness_vals)),
                "p05": float(np.percentile(brightness_vals, 5)),
                "p50": float(np.percentile(brightness_vals, 50)),
                "p95": float(np.percentile(brightness_vals, 95)),
            },
            "contrast": {
                "mean": float(np.mean(contrast_vals)),
                "std": float(np.std(contrast_vals)),
                "p05": float(np.percentile(contrast_vals, 5)),
                "p50": float(np.percentile(contrast_vals, 50)),
                "p95": float(np.percentile(contrast_vals, 95)),
            },
            "sharpness": {
                "mean": float(np.mean(sharpness_vals)),
                "std": float(np.std(sharpness_vals)),
                "p05": float(np.percentile(sharpness_vals, 5)),
                "p50": float(np.percentile(sharpness_vals, 50)),
                "p95": float(np.percentile(sharpness_vals, 95)),
            },
            "entropy": {
                "mean": float(np.mean(entropy_vals)),
                "std": float(np.std(entropy_vals)),
            },
        },
        "raw_distributions": {
            "brightness": brightness_vals,
            "contrast": contrast_vals,
            "sharpness": sharpness_vals,
            "entropy": entropy_vals,
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to {save_path}:")
    print(f"  Brightness: {baseline['summary']['brightness']['mean']:.1f} ± {baseline['summary']['brightness']['std']:.1f}")
    print(f"  Contrast:   {baseline['summary']['contrast']['mean']:.1f} ± {baseline['summary']['contrast']['std']:.1f}")
    print(f"  Sharpness:  {baseline['summary']['sharpness']['mean']:.1f} ± {baseline['summary']['sharpness']['std']:.1f}")
    return baseline


class DriftMonitor:
    """
    Tracks streaming request metrics and tests for distribution shift.
    """

    def __init__(
        self,
        baseline_path: str = "src/monitor/baseline_stats.json",
        window_size: int = 50,
        alpha_threshold: float = 0.05,
        effect_size_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.alpha = alpha_threshold
        self.effect_size_threshold = effect_size_threshold

        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                self.baseline = json.load(f)
        else:
            self.baseline = None

        self.window = deque(maxlen=window_size)
        self.total_observed = 0

    def record_image(self, image: np.ndarray) -> dict:
        """Records an incoming image, updates the buffer, and returns its metrics."""
        metrics = compute_image_metrics(image)
        self.window.append(metrics)
        self.total_observed += 1
        return metrics

    def check_drift(self) -> dict:
        """
        Runs two-sample KS test comparing the current streaming window against the baseline.
        Returns drift flags and retraining recommendation.
        """
        if self.baseline is None:
            return {"status": "insufficient_data", "reason": "No baseline loaded"}

        if len(self.window) < 15:
            return {
                "status": "warming_up",
                "samples_collected": len(self.window),
                "samples_needed": 15,
                "drift_detected": False,
                "trigger_retraining": False,
            }

        win_brightness = [m["brightness"] for m in self.window]
        win_contrast = [m["contrast"] for m in self.window]
        win_sharpness = [m["sharpness"] for m in self.window]

        base_dist = self.baseline["raw_distributions"]
        base_summary = self.baseline["summary"]

        # 1. KS-Test for Brightness
        ks_b = stats.ks_2samp(win_brightness, base_dist["brightness"])
        mean_b_win = np.mean(win_brightness)
        mean_b_base = base_summary["brightness"]["mean"]
        std_b_base = base_summary["brightness"]["std"]
        effect_b = abs(mean_b_win - mean_b_base) / max(std_b_base, 1e-4)

        # 2. KS-Test for Contrast
        ks_c = stats.ks_2samp(win_contrast, base_dist["contrast"])
        mean_c_win = np.mean(win_contrast)
        mean_c_base = base_summary["contrast"]["mean"]
        std_c_base = base_summary["contrast"]["std"]
        effect_c = abs(mean_c_win - mean_c_base) / max(std_c_base, 1e-4)

        # 3. KS-Test for Sharpness
        ks_s = stats.ks_2samp(win_sharpness, base_dist["sharpness"])

        # Retraining Decision Logic
        brightness_drift = bool(ks_b.pvalue < self.alpha and effect_b > self.effect_size_threshold)
        contrast_drift = bool(ks_c.pvalue < self.alpha and effect_c > self.effect_size_threshold)
        sharpness_drift = bool(ks_s.pvalue < self.alpha)

        drift_detected = brightness_drift or contrast_drift or sharpness_drift
        trigger_retraining = brightness_drift or contrast_drift

        reasons = []
        if brightness_drift:
            reasons.append(
                f"Brightness drift: window mean={mean_b_win:.1f} vs baseline={mean_b_base:.1f} (p={ks_b.pvalue:.4e}, effect={effect_b:.2f})"
            )
        if contrast_drift:
            reasons.append(
                f"Contrast drift: window mean={mean_c_win:.1f} vs baseline={mean_c_base:.1f} (p={ks_c.pvalue:.4e}, effect={effect_c:.2f})"
            )
        if sharpness_drift:
            reasons.append(f"Sharpness distribution shift (p={ks_s.pvalue:.4e})")

        return {
            "status": "active",
            "window_size": len(self.window),
            "total_observed": self.total_observed,
            "drift_detected": drift_detected,
            "trigger_retraining": trigger_retraining,
            "details": {
                "brightness": {
                    "ks_statistic": round(float(ks_b.statistic), 4),
                    "p_value": float(ks_b.pvalue),
                    "window_mean": round(float(mean_b_win), 2),
                    "baseline_mean": round(float(mean_b_base), 2),
                    "effect_size": round(float(effect_b), 2),
                    "drift_flag": brightness_drift,
                },
                "contrast": {
                    "ks_statistic": round(float(ks_c.statistic), 4),
                    "p_value": float(ks_c.pvalue),
                    "window_mean": round(float(mean_c_win), 2),
                    "baseline_mean": round(float(mean_c_base), 2),
                    "effect_size": round(float(effect_c), 2),
                    "drift_flag": contrast_drift,
                },
                "sharpness": {
                    "ks_statistic": round(float(ks_s.statistic), 4),
                    "p_value": float(ks_s.pvalue),
                    "drift_flag": sharpness_drift,
                },
            },
            "reasons": reasons,
        }


def run_drift_demo(
    clean_test_dir: str = "data/processed/neu-det/images/test",
    shifted_test_dir: str = "data/processed/neu-det-shifted/images/test",
    report_output: str = "reports/drift_demo_report.json",
):
    """
    Demonstrates drift monitor catching severe factory lighting degradation.
    Phase 1: Streams clean in-distribution test images -> verify NO drift.
    Phase 2: Streams darkened / perturbed factory images -> verify DRIFT DETECTED and RETRAIN TRIGGERED.
    """
    print("\n=======================================================")
    print("  RUNNING DRIFT MONITORING DEMONSTRATION & KS-TEST")
    print("=======================================================")

    baseline_path = "src/monitor/baseline_stats.json"
    if not os.path.exists(baseline_path):
        build_baseline_distribution(save_path=baseline_path)

    monitor = DriftMonitor(baseline_path=baseline_path, window_size=40)

    # Phase 1: Stream 40 Clean Test Images
    clean_imgs = sorted(glob.glob(os.path.join(clean_test_dir, "*.jpg")))[:40]
    print(f"\n[Phase 1] Streaming {len(clean_imgs)} Clean Test Images...")
    for p in clean_imgs:
        img = cv2.imread(p)
        monitor.record_image(img)

    phase1_report = monitor.check_drift()
    print("Phase 1 Result:")
    print(f"  Drift Detected:     {phase1_report['drift_detected']}")
    print(f"  Trigger Retraining: {phase1_report['trigger_retraining']}")
    print(f"  Brightness KS p-val: {phase1_report['details']['brightness']['p_value']:.4f}")

    # Phase 2: Stream 40 Shifted/Darkened Test Images
    shifted_imgs = sorted(glob.glob(os.path.join(shifted_test_dir, "*.jpg")))[:40]
    print(f"\n[Phase 2] Simulating factory lighting shift with {len(shifted_imgs)} darkened/perturbed images...")
    for p in shifted_imgs:
        img = cv2.imread(p)
        monitor.record_image(img)

    phase2_report = monitor.check_drift()
    print("Phase 2 Result:")
    print(f"  Drift Detected:     {phase2_report['drift_detected']}")
    print(f"  Trigger Retraining: {phase2_report['trigger_retraining']}")
    print(f"  Reasons:            {phase2_report['reasons']}")
    print(f"  Brightness KS p-val: {phase2_report['details']['brightness']['p_value']:.4e}")
    print(f"  Contrast KS p-val:   {phase2_report['details']['contrast']['p_value']:.4e}")

    demo_results = {
        "phase1_clean_stream": phase1_report,
        "phase2_shifted_stream": phase2_report,
    }

    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, "w") as f:
        json.dump(demo_results, f, indent=2)

    print(f"\nDrift demo results saved to {report_output}")
    return demo_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift monitoring for steel surface defect inspection")
    parser.add_argument("--build-baseline", action="store_true", help="Build baseline statistics from training set")
    parser.add_argument("--demo", action="store_true", help="Run live drift demo with clean vs shifted streams")
    args = parser.parse_args()

    if args.build_baseline:
        build_baseline_distribution()
    elif args.demo:
        run_drift_demo()
    else:
        build_baseline_distribution()
        run_drift_demo()
