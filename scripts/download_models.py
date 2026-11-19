"""Descarga los modelos ONNX a models/ (Apache-2.0).

- MiniFASNetV2: exportación de minivision-ai/Silent-Face-Anti-Spoofing (HF: garciafido).
- YuNet: detector de rostros de OpenCV Zoo.
"""
import os
import urllib.request

MODELS = {
    "MiniFASNetV2.onnx": "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
    "face_detection_yunet_2023mar.onnx": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
}
dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(dest, exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join(dest, name)
    if os.path.exists(path):
        print("Ya existe", name)
        continue
    print("Descargando", name)
    urllib.request.urlretrieve(url, path)
