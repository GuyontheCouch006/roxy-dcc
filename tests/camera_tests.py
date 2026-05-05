# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Camera — default orientation, aspect ratio, get_ray
#              direction, shoot pixel-to-ray conversion, and property setters.
# ============================================

import math
from scene.camera import Camera
from core import Vec3, Ray
from tests.utils import run_tests, approx_eq, vec3_approx_eq


W, H = 100, 100


def _camera(**kwargs):
    return Camera(width=W, height=H, **kwargs)


# ─── Defaults ─────────────────────────────────────────────────────────────────

def test_default_position_is_origin():
    cam = _camera()
    assert vec3_approx_eq(cam.position, Vec3(0, 0, 0))

def test_default_forward_is_neg_z():
    cam = _camera()
    assert vec3_approx_eq(cam.forward, Vec3(0, 0, -1))

def test_default_fov_is_90():
    cam = _camera()
    assert cam.fov == 90

def test_aspect_ratio_square():
    cam = Camera(width=100, height=100)
    assert approx_eq(cam.aspect_ratio, 1.0)

def test_aspect_ratio_wide():
    cam = Camera(width=200, height=100)
    assert approx_eq(cam.aspect_ratio, 2.0)


# ─── get_ray ──────────────────────────────────────────────────────────────────

def test_get_ray_center_points_forward():
    cam = _camera()
    ray = cam.get_ray(0, 0)
    assert vec3_approx_eq(ray.direction, Vec3(0, 0, -1))

def test_get_ray_direction_is_unit():
    cam = _camera()
    for u, v in [(0, 0), (1, 0), (0, 1), (-1, -1)]:
        ray = cam.get_ray(u, v)
        assert approx_eq(ray.direction.length(), 1.0)

def test_get_ray_origin_is_camera_position():
    pos = Vec3(1, 2, 3)
    cam = Camera(position=pos, width=W, height=H)
    ray = cam.get_ray(0, 0)
    assert vec3_approx_eq(ray.origin, pos)

def test_get_ray_positive_u_tilts_right():
    cam = _camera()
    ray = cam.get_ray(1, 0)
    assert ray.direction.x > 0

def test_get_ray_positive_v_tilts_up():
    cam = _camera()
    ray = cam.get_ray(0, 1)
    assert ray.direction.y > 0

def test_get_ray_returns_ray():
    cam = _camera()
    assert isinstance(cam.get_ray(0, 0), Ray)


# ─── shoot ────────────────────────────────────────────────────────────────────

def test_shoot_returns_ray():
    cam = _camera()
    assert isinstance(cam.shoot(50, 50, W, H), Ray)

def test_shoot_direction_is_unit():
    cam = _camera()
    ray = cam.shoot(50, 50, W, H)
    assert approx_eq(ray.direction.length(), 1.0)

def test_shoot_top_left_points_up_left():
    cam = _camera()
    ray = cam.shoot(0, 0, W, H)
    assert ray.direction.x < 0
    assert ray.direction.y > 0

def test_shoot_bottom_right_points_down_right():
    cam = _camera()
    ray = cam.shoot(99, 99, W, H)
    assert ray.direction.x > 0
    assert ray.direction.y < 0

def test_shoot_wider_fov_deviates_more():
    narrow = Camera(fov=30, width=W, height=H)
    wide = Camera(fov=120, width=W, height=H)
    fwd = Vec3(0, 0, -1)
    r_narrow = narrow.shoot(0, 0, W, H)
    r_wide = wide.shoot(0, 0, W, H)
    assert r_wide.direction.dot(fwd) < r_narrow.direction.dot(fwd)


# ─── Property setters ─────────────────────────────────────────────────────────

def test_fov_setter():
    cam = _camera()
    cam.fov = 45
    assert cam.fov == 45

def test_position_setter():
    cam = _camera()
    cam.position = Vec3(5, 0, 0)
    assert vec3_approx_eq(cam.position, Vec3(5, 0, 0))

def test_set_width_height_updates_aspect():
    cam = Camera(width=100, height=100)
    cam.set_width_height(200, 100)
    assert approx_eq(cam.aspect_ratio, 2.0)


# ─── repr ─────────────────────────────────────────────────────────────────────

def test_repr_contains_camera():
    cam = _camera()
    assert "Camera" in repr(cam)


if __name__ == "__main__":
    tests = [
        test_default_position_is_origin,
        test_default_forward_is_neg_z,
        test_default_fov_is_90,
        test_aspect_ratio_square,
        test_aspect_ratio_wide,
        test_get_ray_center_points_forward,
        test_get_ray_direction_is_unit,
        test_get_ray_origin_is_camera_position,
        test_get_ray_positive_u_tilts_right,
        test_get_ray_positive_v_tilts_up,
        test_get_ray_returns_ray,
        test_shoot_returns_ray,
        test_shoot_direction_is_unit,
        test_shoot_top_left_points_up_left,
        test_shoot_bottom_right_points_down_right,
        test_shoot_wider_fov_deviates_more,
        test_fov_setter,
        test_position_setter,
        test_set_width_height_updates_aspect,
        test_repr_contains_camera,
    ]
    run_tests(tests)
