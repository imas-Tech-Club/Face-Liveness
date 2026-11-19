"""Prueba rápida: corre el pipeline sobre una imagen. Uso: python scripts/smoke_test.py [ruta]
Sin ruta usa la foto de ejemplo 'astronaut' de scikit-image (persona real, foto digital)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.liveness as L  # noqa: E402

if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
else:
    from skimage import data
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)

for div in (1.0, 255.0):
    L.PIXEL_DIV = div
    eng = L.LivenessEngine()
    r = eng.predict(img)
    print(f"PIXEL_DIV={div:>5}: {r.label} score={r.score:.3f} {r.latency_ms:.0f} ms")
