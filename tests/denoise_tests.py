import numpy as np

from rendering.denoise import edge_aware_denoise, linear_to_display
from tests.utils import run_tests


def test_linear_to_display_clamps_and_gamma_corrects():
    linear = np.array([[[-1.0, 0.25, 4.0]]], dtype=np.float32)
    display = linear_to_display(linear)
    assert display.shape == linear.shape
    assert display[0, 0, 0] == 0.0
    assert display[0, 0, 1] == 0.5
    assert display[0, 0, 2] == 1.0


def test_edge_aware_denoise_preserves_shape_and_finite_values():
    pixels = np.zeros((5, 5, 3), dtype=np.float32)
    pixels[:, :] = [0.2, 0.2, 0.2]
    pixels[2, 2] = [1.0, 1.0, 1.0]

    filtered = edge_aware_denoise(pixels, radius=1, sigma_color=0.2, amount=0.8)
    assert filtered.shape == pixels.shape
    assert np.isfinite(filtered).all()
    assert filtered[2, 2, 0] > filtered[0, 0, 0]


def test_edge_aware_denoise_can_be_disabled():
    pixels = np.random.default_rng(1).random((4, 4, 3), dtype=np.float32)
    assert edge_aware_denoise(pixels, radius=0) is pixels
    assert edge_aware_denoise(pixels, amount=0.0) is pixels


if __name__ == "__main__":
    run_tests([
        test_linear_to_display_clamps_and_gamma_corrects,
        test_edge_aware_denoise_preserves_shape_and_finite_values,
        test_edge_aware_denoise_can_be_disabled,
    ])
