"""
High-performance ONNX Runtime Predictor for Steel Defect Inspection.
Handles image preprocessing, letterbox scaling, ONNX session execution,
and Non-Maximum Suppression (NMS) post-processing with calibrated confidence thresholds.
"""

import os
import time
import numpy as np
import cv2
import onnxruntime as ort

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


class ONNXPredictor:
    """
    Production-grade ONNX Runtime Predictor for YOLOv8 defect detection.
    """

    def __init__(
        self,
        model_path: str = "models/onnx/model_int8.onnx",
        conf_threshold: float = 0.28,
        iou_threshold: float = 0.45,
        imgsz: int = 224,
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.classes = CLASSES

        # Fallback to FP32 or PyTorch if INT8 doesn't exist yet
        if not os.path.exists(model_path):
            alt_fp32 = "models/onnx/model_fp32.onnx"
            if os.path.exists(alt_fp32):
                self.model_path = alt_fp32
            else:
                self.model_path = model_path

        self.session = None
        self._init_session(num_threads)

    def _init_session(self, num_threads: int):
        """Initializes ONNX Runtime inference session with CPU optimizations."""
        if os.path.exists(self.model_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            # Dynamically sync input resolution from model graph
            if len(self.input_shape) >= 4 and isinstance(self.input_shape[2], int):
                self.imgsz = self.input_shape[2]
            print(f"ONNXPredictor loaded: {self.model_path} (input shape: {self.input_shape}, imgsz: {self.imgsz})")
        else:
            print(f"Warning: Model file not found at {self.model_path}. Predictor will run in mock/standby mode.")

    def preprocess(self, image: np.ndarray):
        """
        Resizes and normalizes an input image to match the ONNX model input dimensions.
        Returns:
            blob (np.ndarray): Shape (1, 3, imgsz, imgsz), float32 in [0, 1]
            scale (float): Scale factor for bounding box rescaling
            pad (tuple): Padding (pad_w, pad_h)
        """
        h_orig, w_orig = image.shape[:2]
        target_size = self.imgsz

        # Letterbox scaling
        r = min(target_size / h_orig, target_size / w_orig)
        new_unpad = (int(round(w_orig * r)), int(round(h_orig * r)))
        dw = target_size - new_unpad[0]
        dh = target_size - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w_orig, h_orig) != new_unpad:
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        if len(padded.shape) == 2:
            padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        elif padded.shape[2] == 3:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        blob = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, r, (dw, dh)

    def postprocess(self, output: np.ndarray, orig_shape: tuple, ratio: float, pad: tuple) -> list:
        """
        Decodes raw YOLOv8 ONNX outputs via Non-Maximum Suppression (NMS).
        Returns list of detection dictionaries:
            [{class_id, class_name, confidence, bbox: [xmin, ymin, xmax, ymax]}]
        """
        predictions = np.squeeze(output, axis=0).T  # Transpose to (anchors, 4 + classes)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0] - boxes[:, 2] / 2.0
        y1 = boxes[:, 1] - boxes[:, 3] / 2.0
        x2 = boxes[:, 0] + boxes[:, 2] / 2.0
        y2 = boxes[:, 1] + boxes[:, 3] / 2.0

        dw, dh = pad
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        h_orig, w_orig = orig_shape
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences.tolist(), score_threshold=self.conf_threshold, nms_threshold=self.iou_threshold
        )

        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                c_id = int(class_ids[idx])
                c_name = self.classes[c_id] if c_id < len(self.classes) else "unknown"
                conf = float(confidences[idx])
                box = [
                    round(float(x1[idx]), 1),
                    round(float(y1[idx]), 1),
                    round(float(x2[idx]), 1),
                    round(float(y2[idx]), 1),
                ]
                detections.append({
                    "class_id": c_id,
                    "class_name": c_name,
                    "confidence": round(conf, 4),
                    "bbox": box,
                })

        return detections

    def predict(self, image: np.ndarray) -> dict:
        """
        End-to-end inference on a single image.
        Returns detections, inference latency (ms), and summary statistics.
        """
        t0 = time.perf_counter()
        orig_shape = image.shape[:2]

        if self.session is None:
            t1 = time.perf_counter()
            return {
                "detections": [],
                "count": 0,
                "timing_ms": {"total": round((t1 - t0) * 1000.0, 2)},
                "model_status": "standby_no_weights",
            }

        blob, ratio, pad = self.preprocess(image)
        t_pre = time.perf_counter()

        outputs = self.session.run(None, {self.input_name: blob})
        t_infer = time.perf_counter()

        detections = self.postprocess(outputs[0], orig_shape, ratio, pad)
        t_post = time.perf_counter()

        total_latency = (t_post - t0) * 1000.0
        preprocess_latency = (t_pre - t0) * 1000.0
        inference_latency = (t_infer - t_pre) * 1000.0
        postprocess_latency = (t_post - t_infer) * 1000.0

        return {
            "detections": detections,
            "count": len(detections),
            "timing_ms": {
                "total": round(total_latency, 2),
                "preprocess": round(preprocess_latency, 2),
                "inference": round(inference_latency, 2),
                "postprocess": round(postprocess_latency, 2),
            },
            "model_path": self.model_path,
        }
