# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for core utility functions — safe_div, clamp, lerp,
#              angle conversion, and random unit vector sampling.
# ============================================

import math
from core import Vec3
from core.utils import (
    safe_div, clamp, lerp, degrees_to_radians, radians_to_degrees,
    random_unit_vector, random_cosine_hemisphere,
)
from tests.utils import run_tests, approx_eq


# ─── safe_div ─────────────────────────────────────────────────────────────────

def test_safe_div_normal():
    assert approx_eq(safe_div(10, 2), 5.0)

def test_safe_div_negative():
    assert approx_eq(safe_div(-6, 3), -2.0)

def test_safe_div_near_zero_positive_numerator():
    assert safe_div(1, 0) == float('inf')

def test_safe_div_near_zero_negative_numerator():
    assert safe_div(-1, 0) == float('-inf')

def test_safe_div_zero_numerator_near_zero_denom():
    assert safe_div(0, 0) == float('inf')


# ─── clamp ────────────────────────────────────────────────────────────────────

def test_clamp_within_range():
    assert approx_eq(clamp(0.5, 0.0, 1.0), 0.5)

def test_clamp_below_lo():
    assert approx_eq(clamp(-1.0, 0.0, 1.0), 0.0)

def test_clamp_above_hi():
    assert approx_eq(clamp(2.0, 0.0, 1.0), 1.0)

def test_clamp_at_lo():
    assert approx_eq(clamp(0.0, 0.0, 1.0), 0.0)

def test_clamp_at_hi():
    assert approx_eq(clamp(1.0, 0.0, 1.0), 1.0)


# ─── lerp ─────────────────────────────────────────────────────────────────────

def test_lerp_t_zero_returns_a():
    assert approx_eq(lerp(2.0, 8.0, 0.0), 2.0)

def test_lerp_t_one_returns_b():
    assert approx_eq(lerp(2.0, 8.0, 1.0), 8.0)

def test_lerp_t_half_returns_midpoint():
    assert approx_eq(lerp(0.0, 10.0, 0.5), 5.0)

def test_lerp_t_quarter():
    assert approx_eq(lerp(0.0, 4.0, 0.25), 1.0)


# ─── degrees_to_radians ───────────────────────────────────────────────────────

def test_degrees_to_radians_180():
    assert approx_eq(degrees_to_radians(180), math.pi)

def test_degrees_to_radians_90():
    assert approx_eq(degrees_to_radians(90), math.pi / 2)

def test_degrees_to_radians_0():
    assert approx_eq(degrees_to_radians(0), 0.0)

def test_degrees_to_radians_360():
    assert approx_eq(degrees_to_radians(360), 2 * math.pi)


# ─── radians_to_degrees ───────────────────────────────────────────────────────

def test_radians_to_degrees_pi():
    assert approx_eq(radians_to_degrees(math.pi), 180.0)

def test_radians_to_degrees_half_pi():
    assert approx_eq(radians_to_degrees(math.pi / 2), 90.0)

def test_radians_to_degrees_zero():
    assert approx_eq(radians_to_degrees(0), 0.0)

def test_degrees_radians_roundtrip():
    assert approx_eq(radians_to_degrees(degrees_to_radians(45)), 45.0)


# ─── random_unit_vector ───────────────────────────────────────────────────────

def test_random_unit_vector_is_unit_length():
    for _ in range(10):
        v = random_unit_vector()
        assert approx_eq(v.length(), 1.0), f"Expected unit length, got {v.length()}"

def test_random_unit_vector_varies():
    vectors = [random_unit_vector() for _ in range(5)]
    # All five identical would be astronomically unlikely.
    unique = {(round(v.x, 6), round(v.y, 6), round(v.z, 6)) for v in vectors}
    assert len(unique) > 1


def test_random_cosine_hemisphere_is_unit_length():
    normal = Vec3(0, 1, 0)
    for _ in range(10):
        v = random_cosine_hemisphere(normal)
        assert approx_eq(v.length(), 1.0, eps=1e-5)


def test_random_cosine_hemisphere_stays_above_surface():
    normal = Vec3(0, 1, 0)
    for _ in range(10):
        v = random_cosine_hemisphere(normal)
        assert v.dot(normal) >= 0.0


if __name__ == "__main__":
    tests = [
        test_safe_div_normal,
        test_safe_div_negative,
        test_safe_div_near_zero_positive_numerator,
        test_safe_div_near_zero_negative_numerator,
        test_safe_div_zero_numerator_near_zero_denom,
        test_clamp_within_range,
        test_clamp_below_lo,
        test_clamp_above_hi,
        test_clamp_at_lo,
        test_clamp_at_hi,
        test_lerp_t_zero_returns_a,
        test_lerp_t_one_returns_b,
        test_lerp_t_half_returns_midpoint,
        test_lerp_t_quarter,
        test_degrees_to_radians_180,
        test_degrees_to_radians_90,
        test_degrees_to_radians_0,
        test_degrees_to_radians_360,
        test_radians_to_degrees_pi,
        test_radians_to_degrees_half_pi,
        test_radians_to_degrees_zero,
        test_degrees_radians_roundtrip,
        test_random_unit_vector_is_unit_length,
        test_random_unit_vector_varies,
        test_random_cosine_hemisphere_is_unit_length,
        test_random_cosine_hemisphere_stays_above_surface,
    ]
    run_tests(tests)
