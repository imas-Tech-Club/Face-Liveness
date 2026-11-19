"""Evalúa APCER/BPCER (ISO/IEC 30107-3) y latencia.

Estructura esperada de data/:
  data/real/*.jpg
  data/print/*.jpg      (foto impresa)
  data/screen/*.jpg     (foto/video en celular o monitor)
Uso: python scripts/evaluate.py [--threshold 0.5]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.liveness import LivenessEngine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()

eng = LivenessEngine(threshold=a.threshold)
lat = []


def run(folder):
    res = []
    for p in sorted(glob.glob(os.path.join(a.data, folder, "*"))):
        img = cv2.imread(p)
        if img is None:
            continue
        r = eng.predict(img)
        lat.append(r.latency_ms)
        res.append(r.label)
    return res


real = run("real")
if real:
    # BPCER: reales rechazados (attack o sin rostro)
    print(f"BPCER (real): {sum(l != 'real' for l in real) / len(real):.1%}  n={len(real)}")
for atk in ("print", "screen"):
    r = run(atk)
    if r:
        # APCER: ataques aceptados como reales
        print(f"APCER ({atk}): {sum(l == 'real' for l in r) / len(r):.1%}  n={len(r)}")
if lat:
    print(f"Latencia: media {np.mean(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms")
