# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for HitRecord — from_ray construction, front/back face
#              detection, normal orientation, ordering, and bool conversion.
# ============================================

from core import Vec3, Ray, HitRecord
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _ray_toward_z():
    return Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))


# ─── from_ray construction ────────────────────────────────────────────────────

def test_from_ray_point_on_ray():
    ray = _ray_toward_z()
    hit = HitRecord.from_ray(ray, 4.0, Vec3(0, 0, -1))
    assert vec3_approx_eq(hit.point, Vec3(0, 0, -1))

def test_from_ray_stores_t():
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1))
    assert approx_eq(hit.t, 4.0)

def test_from_ray_stores_material():
    sentinel = object()
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1), material=sentinel)
    assert hit.material is sentinel

def test_from_ray_default_material_is_none():
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1))
    assert hit.material is None


# ─── Front face ───────────────────────────────────────────────────────────────

def test_front_face_when_ray_opposes_normal():
    # Ray going +z, outward normal going -z — ray hits the outside.
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1))
    assert hit.front_face is True

def test_front_face_normal_matches_outward():
    outward = Vec3(0, 0, -1)
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, outward)
    assert vec3_approx_eq(hit.normal, outward)


# ─── Back face ────────────────────────────────────────────────────────────────

def test_back_face_when_ray_same_side_as_normal():
    # Ray going +z, outward normal also going +z — ray is inside the surface.
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, 1))
    assert hit.front_face is False

def test_back_face_normal_is_flipped():
    # Normal should be flipped to oppose the ray direction.
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, 1))
    assert vec3_approx_eq(hit.normal, Vec3(0, 0, -1))


# ─── Ordering (by t) ──────────────────────────────────────────────────────────

def test_ordering_lt():
    near = HitRecord.from_ray(_ray_toward_z(), 2.0, Vec3(0, 0, -1))
    far = HitRecord.from_ray(_ray_toward_z(), 5.0, Vec3(0, 0, -1))
    assert near < far

def test_ordering_gt():
    near = HitRecord.from_ray(_ray_toward_z(), 2.0, Vec3(0, 0, -1))
    far = HitRecord.from_ray(_ray_toward_z(), 5.0, Vec3(0, 0, -1))
    assert far > near

def test_ordering_eq():
    h1 = HitRecord.from_ray(_ray_toward_z(), 3.0, Vec3(0, 0, -1))
    h2 = HitRecord.from_ray(_ray_toward_z(), 3.0, Vec3(0, 0, -1))
    assert h1 == h2

def test_min_returns_closest():
    near = HitRecord.from_ray(_ray_toward_z(), 2.0, Vec3(0, 0, -1))
    far = HitRecord.from_ray(_ray_toward_z(), 5.0, Vec3(0, 0, -1))
    assert min(near, far) is near


# ─── Bool ─────────────────────────────────────────────────────────────────────

def test_bool_true_for_positive_t():
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1))
    assert bool(hit) is True

def test_bool_false_for_zero_t():
    hit = HitRecord.from_ray(_ray_toward_z(), 0.0, Vec3(0, 0, -1))
    assert bool(hit) is False

def test_bool_false_for_negative_t():
    hit = HitRecord.from_ray(_ray_toward_z(), -1.0, Vec3(0, 0, -1))
    assert bool(hit) is False


# ─── Repr ─────────────────────────────────────────────────────────────────────

def test_repr_contains_hit_record():
    hit = HitRecord.from_ray(_ray_toward_z(), 4.0, Vec3(0, 0, -1))
    assert "HitRecord" in repr(hit)


if __name__ == "__main__":
    tests = [
        test_from_ray_point_on_ray,
        test_from_ray_stores_t,
        test_from_ray_stores_material,
        test_from_ray_default_material_is_none,
        test_front_face_when_ray_opposes_normal,
        test_front_face_normal_matches_outward,
        test_back_face_when_ray_same_side_as_normal,
        test_back_face_normal_is_flipped,
        test_ordering_lt,
        test_ordering_gt,
        test_ordering_eq,
        test_min_returns_closest,
        test_bool_true_for_positive_t,
        test_bool_false_for_zero_t,
        test_bool_false_for_negative_t,
        test_repr_contains_hit_record,
    ]
    run_tests(tests)
