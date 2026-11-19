"""Pipeline de liveness pasivo: detección de rostro -> recorte -> MiniFASNet (ONNX) -> decisión.

Modelos: Silent-Face-Anti-Spoofing (MiniFASNetV2 escala 2.7 y MiniFASNetV1SE escala 4.0),
exportados a ONNX. Entrada 80x80 BGR en rango 0-255, salida de 3 clases (índice 1 = real).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
# (archivo, escala del recorte)
MODEL_SPECS = [("MiniFASNetV2.onnx", 2.7)]
# Divisor de píxeles: 1.0 (rango 0-255, como el repo original) o 255.0 (según README de HF). Se valida abajo.
PIXEL_DIV = float(os.getenv("LIVENESS_PIXEL_DIV", "1.0"))
INPUT_SIZE = 80
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))


@dataclass
class Result:
    label: str  # "real" | "attack" | "no_face"
    score: float  # probabilidad de "real"
    latency_ms: float


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def crop_scaled(img: np.ndarray, box: tuple[int, int, int, int], scale: float) -> np.ndarray:
    """Recorta alrededor del bbox ampliado `scale` veces (limitado a la imagen) y reescala a 80x80."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    scale = min((h - 1) / bh, (w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x + bw / 2, y + bh / 2
    x1 = max(0, cx - nw / 2)
    y1 = max(0, cy - nh / 2)
    x2 = min(w - 1, cx + nw / 2)
    y2 = min(h - 1, cy + nh / 2)
    crop = img[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))


class LivenessEngine:
    def __init__(self, models_dir: str = MODELS_DIR, threshold: float = THRESHOLD):
        self.threshold = threshold
        yunet_path = os.path.join(models_dir, YUNET_FILE)
        # YuNet (preferido); si falta el archivo, se usa Haar como respaldo.
        self.yunet = (
            cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.7, 0.3, 5000)
            if os.path.exists(yunet_path) and not os.getenv("LIVENESS_USE_HAAR")
            else None
        )
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.models = []
        for name, scale in MODEL_SPECS:
            path = os.path.join(models_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Falta {path}. Corré `python scripts/download_models.py` (ver README)."
                )
            self.models.append((ort.InferenceSession(path, providers=["CPUExecutionProvider"]), scale))

    def detect_face(self, img: np.ndarray):
        if self.yunet is not None:
            h, w = img.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(img)
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda r: r[2] * r[3])
            x, y, bw, bh = (int(v) for v in f[:4])
            x, y = max(x, 0), max(y, 0)
            bw, bh = min(bw, w - x), min(bh, h - y)
            return (x, y, bw, bh) if bw > 10 and bh > 10 else None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

    def predict(self, img: np.ndarray) -> Result:
        t0 = time.perf_counter()
        box = self.detect_face(img)
        if box is None:
            return Result("no_face", 0.0, (time.perf_counter() - t0) * 1000)
        probs = np.zeros(3, dtype=np.float32)
        for sess, scale in self.models:
            crop = crop_scaled(img, box, scale)
            inp = (crop.astype(np.float32) / PIXEL_DIV).transpose(2, 0, 1)[None]  # NCHW, BGR
            out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
            probs += _softmax(out)
        probs /= len(self.models)
        score = float(probs[1])
        label = "real" if score >= self.threshold else "attack"
        return Result(label, score, (time.perf_counter() - t0) * 1000)
