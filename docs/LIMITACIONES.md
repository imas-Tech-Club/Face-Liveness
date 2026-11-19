# Limitaciones y privacidad (borrador, completar con resultados reales)

## Fuera de alcance (declarado)
Máscaras 3D, deepfakes, inyección de imágenes y cámaras virtuales, desafío activo (parpadeo/giro), SDK cliente,
certificación formal y análisis legal profundo.

## Privacidad
Las imágenes se procesan en memoria y **no se guardan ni se registran**. Solo se devuelve `{score, label}`.
Las puntuaciones por sesión se mantienen en memoria (ventana de pocos frames). Las imágenes de `data/` son del
set de prueba propio, con consentimiento de quienes aparecen, y no se versionan (`.gitignore`).

## Limitaciones conocidas (a validar con el set propio)
- Modelo preentrenado en otros datasets: puede generalizar mal a nuestras cámaras e iluminación.
- Detector Haar: sensible a poses y poca luz.
- Sin protección contra reproducción de frames enviados directamente a la API.

## Resultados
| Métrica | Objetivo | Medido |
|---|---|---|
| Latencia | < 500 ms | _pendiente_ |
| BPCER | < 10 % | _pendiente_ |
| APCER impresa | definir | _pendiente_ |
| APCER pantalla | definir | _pendiente_ |
