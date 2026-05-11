# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Primitive implementations — Sphere bounds, normals,
#              and ray intersection (hit, miss, inside).
#              Primitives return PrimitiveHit(t, normal, uv); front_face
#              correction is Shape's responsibility and tested in scene_object_tests.
# ============================================

import math
from scene import Sphere
from scene.primitives import PrimitiveHit
from core import Vec3, Ray
from tests.utils import run_tests, approx_eq, vec3_approx_eq


# --- local_bounds ---

def test_unit_sphere_bounds():
    s = Sphere()
    b = s.local_bounds()
    assert vec3_approx_eq(b.min, Vec3(-1, -1, -1))
    assert vec3_approx_eq(b.max, Vec3( 1,  1,  1))

def test_custom_radius_bounds():
    s = Sphere(radius=3.0)
    b = s.local_bounds()
    assert vec3_approx_eq(b.min, Vec3(-3, -3, -3))
    assert vec3_approx_eq(b.max, Vec3( 3,  3,  3))


# --- normal_at ---

def test_normal_at_is_unit_length():
    s = Sphere()
    n = s.normal_at(Vec3(1, 0, 0))
    assert approx_eq(n.length(), 1.0), f"Expected unit normal, got length {n.length()}"

def test_normal_at_front():
    s = Sphere()
    n = s.normal_at(Vec3(1, 0, 0))
    assert vec3_approx_eq(n, Vec3(1, 0, 0))

def test_normal_at_top():
    s = Sphere()
    n = s.normal_at(Vec3(0, 1, 0))
    assert vec3_approx_eq(n, Vec3(0, 1, 0))


# --- intersect: hits ---

def test_ray_hits_sphere_front():
    s = Sphere()
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert hit is not None
    assert isinstance(hit, PrimitiveHit)
    assert approx_eq(hit.t, 4.0), f"Expected t=4, got {hit.t}"

def test_hit_point_on_surface():
    s = Sphere()
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    point = ray.at(hit.t)
    assert vec3_approx_eq(point, Vec3(0, 0, -1))

def test_front_face_outward_normal():
    # PrimitiveHit carries the outward surface normal (no front_face correction).
    s = Sphere()
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert vec3_approx_eq(hit.normal, Vec3(0, 0, -1))


# --- intersect: inside sphere ---

def test_ray_inside_sphere_hits_back():
    s = Sphere()
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert hit is not None
    assert approx_eq(hit.t, 1.0), f"Expected t=1, got {hit.t}"

def test_ray_inside_sphere_outward_normal_points_forward():
    # Inside hit: the outward normal at z=+1 is (0,0,1) — points away from center.
    s = Sphere()
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert vec3_approx_eq(hit.normal, Vec3(0, 0, 1))


# --- intersect: misses ---

def test_ray_misses_sphere():
    s = Sphere()
    ray = Ray(Vec3(0, 5, -5), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert hit is None

def test_ray_behind_sphere_misses():
    s = Sphere()
    ray = Ray(Vec3(0, 0, 5), Vec3(0, 0, 1))
    hit = s.intersect(ray)
    assert hit is None


# --- repr ---

def test_repr():
    s = Sphere(radius=2.5)
    assert "Sphere" in repr(s) and "2.5" in repr(s)


if __name__ == "__main__":
    tests = [
        test_unit_sphere_bounds,
        test_custom_radius_bounds,
        test_normal_at_is_unit_length,
        test_normal_at_front,
        test_normal_at_top,
        test_ray_hits_sphere_front,
        test_hit_point_on_surface,
        test_front_face_outward_normal,
        test_ray_inside_sphere_hits_back,
        test_ray_inside_sphere_outward_normal_points_forward,
        test_ray_misses_sphere,
        test_ray_behind_sphere_misses,
        test_repr,
    ]
    run_tests(tests)
