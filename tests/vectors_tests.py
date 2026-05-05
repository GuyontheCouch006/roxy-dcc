# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Vec2, Vec3, Color, and Ray — arithmetic, dot/cross,
#              length, normalization, equality, and Ray.at().
# ============================================

import math
from core import Vec3, Ray, Color
from core.vectors import Vec2
from tests.utils import run_tests, approx_eq, vec3_approx_eq


# ─── Vec2 ─────────────────────────────────────────────────────────────────────

def test_vec2_add():
    assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)

def test_vec2_sub():
    assert Vec2(5, 3) - Vec2(2, 1) == Vec2(3, 2)

def test_vec2_mul():
    assert Vec2(2, 3) * 4 == Vec2(8, 12)

def test_vec2_rmul():
    assert 4 * Vec2(2, 3) == Vec2(8, 12)

def test_vec2_div():
    v = Vec2(4, 6) / 2
    assert approx_eq(v.x, 2) and approx_eq(v.y, 3)

def test_vec2_neg():
    assert -Vec2(1, -2) == Vec2(-1, 2)

def test_vec2_dot():
    assert approx_eq(Vec2(1, 0).dot(Vec2(0, 1)), 0)
    assert approx_eq(Vec2(1, 0).dot(Vec2(1, 0)), 1)

def test_vec2_length():
    assert approx_eq(Vec2(3, 4).length(), 5.0)

def test_vec2_length_sq():
    assert approx_eq(Vec2(3, 4).length_sq(), 25.0)

def test_vec2_normalize():
    n = Vec2(3, 0).normalize()
    assert approx_eq(n.length(), 1.0)
    assert approx_eq(n.x, 1.0)

def test_vec2_normalize_zero_returns_zero():
    n = Vec2(0, 0).normalize()
    assert approx_eq(n.x, 0) and approx_eq(n.y, 0)

def test_vec2_eq():
    assert Vec2(1, 2) == Vec2(1, 2)
    assert Vec2(1, 2) != Vec2(1, 3)

def test_vec2_hash_equal_vecs():
    assert hash(Vec2(1, 2)) == hash(Vec2(1, 2))


# ─── Vec3 ─────────────────────────────────────────────────────────────────────

def test_vec3_add():
    assert Vec3(1, 2, 3) + Vec3(4, 5, 6) == Vec3(5, 7, 9)

def test_vec3_sub():
    assert Vec3(5, 5, 5) - Vec3(1, 2, 3) == Vec3(4, 3, 2)

def test_vec3_mul():
    assert Vec3(1, 2, 3) * 3 == Vec3(3, 6, 9)

def test_vec3_rmul():
    assert 3 * Vec3(1, 2, 3) == Vec3(3, 6, 9)

def test_vec3_div():
    v = Vec3(2, 4, 6) / 2
    assert vec3_approx_eq(v, Vec3(1, 2, 3))

def test_vec3_neg():
    assert -Vec3(1, -2, 3) == Vec3(-1, 2, -3)

def test_vec3_dot_perpendicular():
    assert approx_eq(Vec3(1, 0, 0).dot(Vec3(0, 1, 0)), 0)

def test_vec3_dot_parallel():
    assert approx_eq(Vec3(1, 0, 0).dot(Vec3(1, 0, 0)), 1)

def test_vec3_cross_right_hand_rule():
    # x cross y = z
    result = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
    assert vec3_approx_eq(result, Vec3(0, 0, 1))

def test_vec3_cross_anticommutative():
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert vec3_approx_eq(a.cross(b), -(b.cross(a)))

def test_vec3_length():
    assert approx_eq(Vec3(1, 2, 2).length(), 3.0)

def test_vec3_length_sq():
    assert approx_eq(Vec3(1, 2, 2).length_sq(), 9.0)

def test_vec3_normalize_is_unit():
    n = Vec3(3, 1, 4).normalize()
    assert approx_eq(n.length(), 1.0)

def test_vec3_normalize_direction_preserved():
    n = Vec3(0, 5, 0).normalize()
    assert vec3_approx_eq(n, Vec3(0, 1, 0))

def test_vec3_normalize_zero_returns_zero():
    n = Vec3(0, 0, 0).normalize()
    assert vec3_approx_eq(n, Vec3(0, 0, 0))

def test_vec3_eq():
    assert Vec3(1, 2, 3) == Vec3(1, 2, 3)
    assert Vec3(1, 2, 3) != Vec3(1, 2, 4)

def test_vec3_eq_wrong_type():
    assert Vec3(1, 2, 3) != "not a vec"

def test_vec3_hash_equal_vecs():
    assert hash(Vec3(1, 2, 3)) == hash(Vec3(1, 2, 3))


# ─── Color ────────────────────────────────────────────────────────────────────

def test_color_rgb_properties():
    c = Color(0.2, 0.5, 0.9)
    assert approx_eq(c.r, 0.2)
    assert approx_eq(c.g, 0.5)
    assert approx_eq(c.b, 0.9)

def test_color_add_returns_color():
    result = Color(0.1, 0.2, 0.3) + Color(0.4, 0.5, 0.6)
    assert isinstance(result, Color)
    assert vec3_approx_eq(result, Color(0.5, 0.7, 0.9))

def test_color_mul_returns_color():
    result = Color(0.5, 0.5, 0.5) * 2
    assert isinstance(result, Color)
    assert vec3_approx_eq(result, Color(1.0, 1.0, 1.0))

def test_color_repr():
    r = repr(Color(1, 0, 0))
    assert "Color" in r and "r=" in r


# ─── Ray ──────────────────────────────────────────────────────────────────────

def test_ray_direction_normalized():
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 3, 0))
    assert approx_eq(ray.direction.length(), 1.0)

def test_ray_at_t_zero_is_origin():
    origin = Vec3(1, 2, 3)
    ray = Ray(origin, Vec3(0, 1, 0))
    assert vec3_approx_eq(ray.at(0), origin)

def test_ray_at_t_one():
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
    assert vec3_approx_eq(ray.at(1), Vec3(0, 0, 1))

def test_ray_at_t_five():
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    assert vec3_approx_eq(ray.at(5), Vec3(0, 0, 0))

def test_ray_repr():
    ray = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
    assert "Ray" in repr(ray)


if __name__ == "__main__":
    tests = [
        test_vec2_add, test_vec2_sub, test_vec2_mul, test_vec2_rmul,
        test_vec2_div, test_vec2_neg, test_vec2_dot, test_vec2_length,
        test_vec2_length_sq, test_vec2_normalize,
        test_vec2_normalize_zero_returns_zero, test_vec2_eq,
        test_vec2_hash_equal_vecs,
        test_vec3_add, test_vec3_sub, test_vec3_mul, test_vec3_rmul,
        test_vec3_div, test_vec3_neg, test_vec3_dot_perpendicular,
        test_vec3_dot_parallel, test_vec3_cross_right_hand_rule,
        test_vec3_cross_anticommutative, test_vec3_length, test_vec3_length_sq,
        test_vec3_normalize_is_unit, test_vec3_normalize_direction_preserved,
        test_vec3_normalize_zero_returns_zero, test_vec3_eq, test_vec3_eq_wrong_type,
        test_vec3_hash_equal_vecs,
        test_color_rgb_properties, test_color_add_returns_color,
        test_color_mul_returns_color, test_color_repr,
        test_ray_direction_normalized, test_ray_at_t_zero_is_origin,
        test_ray_at_t_one, test_ray_at_t_five, test_ray_repr,
    ]
    run_tests(tests)
