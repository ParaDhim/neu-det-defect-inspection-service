"""Evaluate models/seeds/*.pt (val, test, shifted test) and aggregate mean +/- std (ddof=1)."""
import json
from pathlib import Path

import numpy as np

from eval_all_runs import CLEAN, SHIFTED, evaluate

IMGSZ = {"01": 200, "06": 224, "07": 224, "10": 224, "11": 640}
res = {}
for key, imgsz in IMGSZ.items():
    res[key] = {"val": [], "test": [], "shifted": [], "test_50_95": [], "test_crazing_ap50": [], "test_per_class": [], "shifted_per_class": []}
    for seed in (0, 1, 2):
        w = f"models/seeds/{key}_s{seed}.pt"
        v, t, sh = evaluate(w, CLEAN, imgsz, "val"), evaluate(w, CLEAN, imgsz), evaluate(w, SHIFTED, imgsz)
        res[key]["val"].append(v["mAP50"]); res[key]["test"].append(t["mAP50"])
        res[key]["shifted"].append(sh["mAP50"]); res[key]["test_crazing_ap50"].append(t["per_class_ap50"][0]); res[key]["test_per_class"].append(t["per_class_ap50"]); res[key]["shifted_per_class"].append(sh["per_class_ap50"]); res[key]["test_50_95"].append(t["mAP50_95"])
        print("RESULT", key, seed, v["mAP50"], t["mAP50"], sh["mAP50"], flush=True)
for key, d in res.items():
    d["summary"] = {k: [round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2)]
                    for k, v in d.items() if isinstance(v, list) and "per_class" not in k}
    for k in ("test_per_class", "shifted_per_class"):
        a = np.array(d[k])
        d["summary"][k] = [[round(float(m), 2), round(float(sd), 2)] for m, sd in zip(a.mean(0), a.std(0, ddof=1))]
Path("reports/seed_eval_conf001.json").write_text(json.dumps(res, indent=2))
