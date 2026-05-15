from core import Color
from rendering.sampling import clamp_color_sample, mis_power_weight, pixel_sample_offset
from tests.utils import run_tests, approx_eq


def test_pixel_sample_offset_is_inside_pixel():
    for frame in range(8):
        x, y = pixel_sample_offset(12, 34, frame)
        assert 0.0 <= x < 1.0
        assert 0.0 <= y < 1.0


def test_pixel_sample_offset_changes_by_frame():
    a = pixel_sample_offset(4, 5, 0)
    b = pixel_sample_offset(4, 5, 1)
    assert a != b


def test_pixel_sample_offset_changes_by_pixel():
    a = pixel_sample_offset(4, 5, 0)
    b = pixel_sample_offset(5, 5, 0)
    assert a != b


def test_clamp_color_sample_can_be_disabled():
    color = Color(20, -1, 2)
    assert clamp_color_sample(color, None) is color
    assert clamp_color_sample(color, 0) is color


def test_clamp_color_sample_limits_extremes():
    clamped = clamp_color_sample(Color(20, -1, 2), 10.0)
    assert approx_eq(clamped.r, 10.0)
    assert approx_eq(clamped.g, 0.0)
    assert approx_eq(clamped.b, 2.0)


def test_mis_power_weight_handles_equal_pdfs():
    assert approx_eq(mis_power_weight(2.0, 2.0), 0.5)


def test_mis_power_weight_handles_zero_pdfs():
    assert approx_eq(mis_power_weight(0.0, 4.0), 0.0)
    assert approx_eq(mis_power_weight(4.0, 0.0), 1.0)
    assert approx_eq(mis_power_weight(0.0, 0.0), 0.0)


if __name__ == "__main__":
    tests = [
        test_pixel_sample_offset_is_inside_pixel,
        test_pixel_sample_offset_changes_by_frame,
        test_pixel_sample_offset_changes_by_pixel,
        test_clamp_color_sample_can_be_disabled,
        test_clamp_color_sample_limits_extremes,
        test_mis_power_weight_handles_equal_pdfs,
        test_mis_power_weight_handles_zero_pdfs,
    ]
    run_tests(tests)
