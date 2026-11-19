from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import os

from app.liveness import LivenessEngine

app = FastAPI(title="Face Liveness API")
_engine: LivenessEngine | None = None


def engine() -> LivenessEngine:
    global _engine
    if _engine is None:
        _engine = LivenessEngine()
    return _engine


@app.get("/health")
def health():
    return {"status": "ok"}


WINDOW = int(os.getenv("LIVENESS_WINDOW", "5"))  # frames promediados por sesión
_sessions: dict[str, deque] = {}


@app.post("/liveness")
async def liveness(frame: UploadFile = File(...), session_id: str | None = Form(None)):
    # Privacidad: la imagen se procesa en memoria y nunca se guarda ni se loguea.
    data = np.frombuffer(await frame.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida")
    eng = engine()
    r = eng.predict(img)
    score, label = r.score, r.label
    if session_id and r.label != "no_face":
        if len(_sessions) > 1000:  # evita crecimiento sin límite
            _sessions.clear()
        win = _sessions.setdefault(session_id, deque(maxlen=WINDOW))
        win.append(r.score)
        score = float(np.mean(win))
        label = "real" if score >= eng.threshold else "attack"
    return {"score": round(score, 4), "label": label, "latency_ms": round(r.latency_ms, 1)}


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "demo"), html=True), name="demo")
