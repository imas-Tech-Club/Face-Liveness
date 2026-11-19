import numpy as np

from app.liveness import crop_scaled, INPUT_SIZE


def test_crop_scaled_shape_and_bounds():
    img = np.zeros((480, 640, 3), np.uint8)
    out = crop_scaled(img, (500, 300, 100, 100), 4.0)  # bbox cerca del borde
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, 3)
