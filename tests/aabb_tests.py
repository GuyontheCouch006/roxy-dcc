# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for AABB — ray intersection (hit, miss, behind), union, and centroid.
# ============================================

from core import AABB, Vec3, Ray
from tests.utils import run_tests, approx_eq

def test_ray_hits_box():
    box = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
    ray = Ray(Vec3(-1, 0.5, 0.5), Vec3(1, 0, 0))  # Ray pointing towards the box
    t = box.intersect(ray)
    assert t is not None, "Expected ray to hit the box"
    assert approx_eq(t, 1), f"Expected hit at t=1, got {t}"

def test_ray_misses_box():
    box = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
    ray = Ray(Vec3(-1, 2, 0.5), Vec3(1, 0, 0))  # Ray above the box
    t = box.intersect(ray)
    assert t is None, "Expected ray to miss the box"

def test_ray_behind_box():
    box = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
    ray = Ray(Vec3(2, 0.5, 0.5), Vec3(1, 0, 0))  # Ray pointing away from the box
    t = box.intersect(ray)
    assert t is None, "Expected ray to miss the box since it's behind"

def test_union_of_boxes():
    box1 = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
    box2 = AABB(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5))
    union_box = box1.union(box2)
    assert union_box.min == Vec3(0, 0, 0), f"Expected min to be (0, 0, 0), got {union_box.min}"
    assert union_box.max == Vec3(1.5, 1.5, 1.5), f"Expected max to be (1.5, 1.5, 1.5), got {union_box.max}"

def test_centroid():
    box = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
    centroid = box.centroid()
    assert centroid == Vec3(1, 1, 1), f"Expected centroid to be (1, 1, 1), got {centroid}"  


if __name__ == "__main__":
    tests = [
        test_ray_hits_box,
        test_ray_misses_box,
        test_ray_behind_box,
        test_union_of_boxes,
        test_centroid
    ]

    run_tests(tests)