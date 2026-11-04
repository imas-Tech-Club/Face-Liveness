# Face Liveness Detection

Motor de detección de vida facial (face liveness) que distingue a una persona real de intentos de suplantación con
fotos, pantallas o videos, y se ofrece por una API REST. Proyecto del club de alumnos de Informática de UADE.

## Estado

En planificación. Todavía no hay código para correr.

## MVP

El MVP busca responder una pregunta: ¿podemos distinguir en tiempo real a una persona real de una foto o una
pantalla, y ofrecerlo detrás de una API que otra aplicación pueda llamar?

- **Cómo funciona:** webcam, detección del rostro, recorte, modelo de liveness pasivo y decisión real o ataque.
- **Modelo:** preentrenado y usado tal cual, sin entrenamiento propio. El candidato principal es MiniFASNet.
- **Ataques cubiertos:** foto impresa y foto o video mostrado en un celular o un monitor.
- **API:** un endpoint de FastAPI que recibe cuadros y devuelve `{score, label}`, más una demo web que lo consume.
- **Evaluación:** set de prueba propio, con APCER y BPCER por tipo de ataque (ISO/IEC 30107-3) y latencia.
- **Terminado cuando:** decide en menos de 500 ms, rechaza a menos del 10 % de las personas reales y deja pasar
  pocos ataques de cada tipo, por debajo del umbral que fije el equipo.
- **Fuera de alcance:** máscaras, deepfakes, inyección de imágenes o cámaras virtuales, fine-tuning, desafío activo
  (parpadeo o giro) y SDK en el cliente.

El plan completo, con roles, cronograma, datasets y limitaciones, está en el
[documento del equipo](https://docs.google.com/document/d/15Zb2KnBQdbEGCwkuIVotcWF1z9f6M8xO2aKvtGtsQLg/edit). Si
algo de esta sección cambia, se cambia primero ahí.

## Cómo correrlo

Se va a completar cuando exista el prototipo. El stack previsto es Python, OpenCV, ONNX Runtime y FastAPI.
