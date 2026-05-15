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


def test_edge_aware_denoise_none_guides_match_legacy_call():
    pixels = np.random.default_rng(2).random((4, 5, 3), dtype=np.float32)
    legacy = edge_aware_denoise(pixels, 1, 0.25, 0.6)
    explicit_none = edge_aware_denoise(
        pixels,
        radius=1,
        sigma_color=0.25,
        amount=0.6,
        normal=None,
        albedo=None,
        depth=None,
    )

    assert np.allclose(explicit_none, legacy)


def _boundary_pixels():
    pixels = np.zeros((3, 5, 3), dtype=np.float32)
    pixels[:, 2:] = 1.0
    return pixels


def _assert_guide_preserves_boundary(**guide_kwargs):
    pixels = _boundary_pixels()
    unguided = edge_aware_denoise(
        pixels,
        radius=1,
        sigma_color=10.0,
        amount=1.0,
    )
    guided = edge_aware_denoise(
        pixels,
        radius=1,
        sigma_color=10.0,
        amount=1.0,
        **guide_kwargs,
    )

    assert guided[1, 1, 0] < unguided[1, 1, 0] * 0.25
    assert guided[1, 2, 0] > unguided[1, 2, 0] + 0.20


def test_edge_aware_denoise_uses_normal_guide():
    normal = np.zeros((3, 5, 3), dtype=np.float32)
    normal[:, :2] = [0.0, 1.0, 0.0]
    normal[:, 2:] = [0.0, -1.0, 0.0]

    _assert_guide_preserves_boundary(normal=normal, sigma_normal=0.10)


def test_edge_aware_denoise_uses_albedo_guide():
    albedo = np.zeros((3, 5, 3), dtype=np.float32)
    albedo[:, :2] = [0.1, 0.1, 0.1]
    albedo[:, 2:] = [0.9, 0.9, 0.9]

    _assert_guide_preserves_boundary(albedo=albedo, sigma_albedo=0.05)


def test_edge_aware_denoise_uses_depth_guide():
    depth = np.ones((3, 5), dtype=np.float32)
    depth[:, 2:] = 10.0

    _assert_guide_preserves_boundary(depth=depth, sigma_depth=0.10)


def test_edge_aware_denoise_rejects_mismatched_guide_shape():
    pixels = np.zeros((3, 5, 3), dtype=np.float32)
    normal = np.zeros((3, 4, 3), dtype=np.float32)

    try:
        edge_aware_denoise(pixels, normal=normal)
    except ValueError as exc:
        assert "normal guide" in str(exc)
    else:
        assert False, "expected mismatched guide shape to raise ValueError"


if __name__ == "__main__":
    run_tests([
        test_linear_to_display_clamps_and_gamma_corrects,
        test_edge_aware_denoise_preserves_shape_and_finite_values,
        test_edge_aware_denoise_can_be_disabled,
        test_edge_aware_denoise_none_guides_match_legacy_call,
        test_edge_aware_denoise_uses_normal_guide,
        test_edge_aware_denoise_uses_albedo_guide,
        test_edge_aware_denoise_uses_depth_guide,
        test_edge_aware_denoise_rejects_mismatched_guide_shape,
    ])
