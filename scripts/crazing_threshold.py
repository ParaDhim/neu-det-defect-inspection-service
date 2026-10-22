"""Per-class confidence threshold for `crazing`, tuned on the VALIDATION split only.

1. On val (conf=0.001) take the crazing F1-vs-confidence curve and pick its argmax.
2. Evaluate val and test once at that crazing threshold; other classes keep the global
   0.22 operating point (Ultralytics NMS is per-class, so per-class thresholds are independent).
"""
import json

import numpy as np
from ultralytics import YOLO

PATH, IMGSZ, GLOBAL = "models/onnx/model_edge_fp32.onnx", 224, 0.22
Y = "data/processed/neu-det/dataset.yaml"
kw = dict(data=Y, imgsz=IMGSZ, rect=False, verbose=False, workers=0)

r = YOLO(PATH).val(split="val", conf=0.001, plots=False, **kw)
f1 = np.asarray(r.box.f1_curve)[0]
px = np.linspace(0, 1, f1.shape[0])
thr = float(px[int(f1.argmax())])
print(f"RESULT crazing val-F1-optimal conf = {thr:.3f} (val crazing F1 {f1.max():.3f}; at {GLOBAL}: {f1[int(GLOBAL*(len(px)-1))]:.3f})")

out = {"crazing_threshold": round(thr, 3), "global_threshold": GLOBAL}
for split in ("val", "test"):
    for tag, conf in (("global", GLOBAL), ("crazing_tuned", thr)):
        rr = YOLO(PATH).val(split=split, conf=conf, iou=0.45, plots=True, project="runs/crz", name=f"{split}_{tag}", exist_ok=True, **kw)
        m = rr.confusion_matrix.matrix
        tp = np.diag(m)[:6]; fp = m[:6].sum(1) - tp; fn = m[:, :6].sum(0) - tp
        out[f"{split}_{tag}"] = {"tp": tp.astype(int).tolist(), "fp": fp.astype(int).tolist(), "fn": fn.astype(int).tolist()}
        print(f"RESULT {split} {tag} conf={conf:.3f} crazing tp/fp/fn = {int(tp[0])}/{int(fp[0])}/{int(fn[0])}", flush=True)

# combined operating point: crazing at tuned threshold, other classes at the global one
for split in ("val", "test"):
    g, c = out[f"{split}_global"], out[f"{split}_crazing_tuned"]
    tp, fp, fn = (np.array(g[k]) for k in ("tp", "fp", "fn"))
    tp[0], fp[0], fn[0] = c["tp"][0], c["fp"][0], c["fn"][0]
    p, rc = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    out[f"{split}_combined"] = {"macro_p": round(float(p.mean()), 3), "macro_r": round(float(rc.mean()), 3),
                                "micro_p": round(float(tp.sum() / (tp.sum() + fp.sum())), 3),
                                "micro_r": round(float(tp.sum() / (tp.sum() + fn.sum())), 3)}
    print("RESULT", split, "combined", out[f"{split}_combined"])
json.dump(out, open("reports/crazing_threshold.json", "w"), indent=2)
