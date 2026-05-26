from rendering.taichi_wavefront_renderer import TaichiWavefrontRenderer
from rendering.taichi.fields import MAX_H, MAX_W
from tests.utils import run_tests


def test_taichi_wavefront_rejects_width_larger_than_field_capacity():
    try:
        TaichiWavefrontRenderer._validate_image_size(MAX_W + 1, 1)
    except ValueError as exc:
        assert "exceeds field capacity" in str(exc)
    else:
        raise AssertionError("Expected oversized width to raise ValueError")


def test_taichi_wavefront_rejects_height_larger_than_field_capacity():
    try:
        TaichiWavefrontRenderer._validate_image_size(1, MAX_H + 1)
    except ValueError as exc:
        assert "exceeds field capacity" in str(exc)
    else:
        raise AssertionError("Expected oversized height to raise ValueError")


if __name__ == "__main__":
    run_tests([
        test_taichi_wavefront_rejects_width_larger_than_field_capacity,
        test_taichi_wavefront_rejects_height_larger_than_field_capacity,
    ])
