# Prototipo MVP funcionando: qué se hizo y próximos pasos

## Qué se hizo en este commit

Primer prototipo end-to-end de liveness pasivo, siguiendo el plan del MVP (modelo preentrenado, sin entrenar):

- **Pipeline** (`app/liveness.py`): webcam → detección de rostro con **YuNet** (Haar de respaldo) → recorte 2.7× a 80×80 → **MiniFASNetV2** (ONNX Runtime) → decisión `real` / `attack` / `no_face` según umbral.
- **API** (`app/main.py`): FastAPI con `POST /liveness` → `{score, label, latency_ms}` y `GET /health`. Promedia el score de los últimos 5 frames si se manda `session_id`. Las imágenes no se guardan.
- **Demo web** (`demo/index.html`): usa la webcam y muestra REAL/ATTACK en vivo.
- **Evaluación** (`scripts/evaluate.py`): calcula BPCER, APCER por ataque (`print`, `screen`) y latencia.
- **Utilidades:** `scripts/download_models.py`, `scripts/smoke_test.py`, un test en `tests/`.
- **Docs:** `docs/LIMITACIONES.md` (borrador de limitaciones, privacidad y tabla de métricas).

**Decisiones:**
- Solo se usa MiniFASNetV2 (único ONNX confiable disponible, Apache-2.0).
- La normalización de entrada es rango 0–255 (la que usa el repo original). Con `/255` una persona real daba `attack`. Configurable con `LIVENESS_PIXEL_DIV`.
- Se fijó `opencv-python-headless<5` (OpenCV 5 no trae el detector Haar).
- El paper PixelJev (2609.29283) y el I-FAS (2501.01720) **no** se usan: son modelos multimodales pesados, fuera del alcance del MVP.

**Estado de verificación:** solo se probó con una foto de ejemplo de una persona real (da `real`) y el test unitario pasa. **Todavía no se midió contra ataques.**

## Cómo correrlo

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_models.py
.venv\Scripts\python -m uvicorn app.main:app
```
Abrir http://127.0.0.1:8000 y permitir la cámara.

## Próximos pasos

- [ ] **Datos + evaluación:** armar el set propio con consentimiento de quienes aparecen (aprox. 50 imágenes por clase) en `data/real`, `data/print` (foto impresa) y `data/screen` (foto/video en celular o monitor). Variar luz, distancia y cámara. No se versiona (`.gitignore`).
- [ ] **Datos + evaluación:** correr `scripts/evaluate.py` y completar la tabla de `docs/LIMITACIONES.md`.
- [ ] **Datos + evaluación:** definir el APCER tolerable por tipo de ataque (el BPCER objetivo es < 10 %).
- [ ] **Modelos:** comparar YuNet vs Haar (`LIVENESS_USE_HAAR=1`) con `evaluate.py`.
- [ ] **Modelos:** ajustar `LIVENESS_THRESHOLD` y `LIVENESS_WINDOW` con los resultados.
- [ ] **Modelos:** si el APCER no alcanza, convertir `MiniFASNetV1SE` (escala 4.0) a ONNX y promediar con V2.
- [ ] **Datasets públicos (opcional):** evaluar CASIA-MFSD, Replay-Attack o MSU-MFSD como referencia extra. Ver la tabla del repo [DeepFAS](https://github.com/ZitongYu/DeepFAS) y revisar licencias y acceso.
- [ ] **API/demo:** probar la demo con webcam común (cara real, foto impresa, pantalla de celular) y confirmar latencia < 500 ms.
- [ ] **API/demo:** verificar que otra app puede llamar a `POST /liveness`.
- [ ] **Líder:** completar el apartado "Cómo correrlo" del README y revisar el informe de limitaciones.

## Criterio de terminado

Decide en < 500 ms, BPCER < 10 %, APCER por debajo del umbral definido para impresa y pantalla, API llamable desde otra app e informe honesto de limitaciones.

## Fuera de alcance

Máscaras, deepfakes, inyección de imágenes y cámaras virtuales, fine-tuning, desafío activo (parpadeo/giro), SDK en el cliente.
