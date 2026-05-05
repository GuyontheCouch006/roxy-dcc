# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Image — dimensions, aspect ratio, pixel buffer shape,
#              write/read, out-of-bounds safety, and clear.
# ============================================

from rendering.image import Image
from core import Color
from tests.utils import run_tests, approx_eq


# ─── Construction ─────────────────────────────────────────────────────────────

def test_width_stored():
    img = Image(320, 240)
    assert img.width == 320

def test_height_stored():
    img = Image(320, 240)
    assert img.height == 240

def test_aspect_ratio():
    img = Image(200, 100)
    assert approx_eq(img.aspect_ratio, 2.0)

def test_square_aspect_ratio():
    img = Image(100, 100)
    assert approx_eq(img.aspect_ratio, 1.0)

def test_pixels_shape():
    img = Image(80, 60)
    assert img.pixels.shape == (60, 80, 3)

def test_pixels_initially_black():
    img = Image(10, 10)
    assert img.pixels.max() == 0.0


# ─── write_pixel / read_pixel ─────────────────────────────────────────────────

def test_write_and_read_pixel():
    img = Image(10, 10)
    img.write_pixel(3, 4, Color(0.2, 0.5, 0.8))
    px = img.read_pixel(3, 4)
    assert approx_eq(px[0], 0.2 ** 0.5)
    assert approx_eq(px[1], 0.5 ** 0.5)
    assert approx_eq(px[2], 0.8 ** 0.5)

def test_write_pixel_does_not_affect_others():
    img = Image(10, 10)
    img.write_pixel(5, 5, Color(1, 0, 0))
    px = img.read_pixel(0, 0)
    assert approx_eq(px[0], 0.0)

def test_write_pixel_out_of_bounds_does_not_raise():
    img = Image(10, 10)
    img.write_pixel(-1, 0, Color(1, 0, 0))
    img.write_pixel(0, -1, Color(1, 0, 0))
    img.write_pixel(10, 0, Color(1, 0, 0))
    img.write_pixel(0, 10, Color(1, 0, 0))

def test_write_pixel_overwrites():
    img = Image(10, 10)
    img.write_pixel(1, 1, Color(1, 0, 0))
    img.write_pixel(1, 1, Color(0, 0, 1))
    px = img.read_pixel(1, 1)
    assert approx_eq(px[2], 1.0)
    assert approx_eq(px[0], 0.0)


# ─── clear ────────────────────────────────────────────────────────────────────

def test_clear_resets_pixels_to_black():
    img = Image(10, 10)
    img.write_pixel(5, 5, Color(1, 0.5, 0.25))
    img.clear()
    assert img.pixels.max() == 0.0


# ─── repr ─────────────────────────────────────────────────────────────────────

def test_repr_contains_dimensions():
    img = Image(800, 600)
    r = repr(img)
    assert "800" in r and "600" in r


if __name__ == "__main__":
    tests = [
        test_width_stored,
        test_height_stored,
        test_aspect_ratio,
        test_square_aspect_ratio,
        test_pixels_shape,
        test_pixels_initially_black,
        test_write_and_read_pixel,
        test_write_pixel_does_not_affect_others,
        test_write_pixel_out_of_bounds_does_not_raise,
        test_write_pixel_overwrites,
        test_clear_resets_pixels_to_black,
        test_repr_contains_dimensions,
    ]
    run_tests(tests)
