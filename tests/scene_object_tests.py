# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for SceneObject — defaults, property delegation, shape/material
#              assignment, and world-space ray intersection.
# ============================================

from scene import Sphere, Diffuse, SceneObject
from core import Vec3, Ray, Color, RotationOrder
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _sphere_obj(**kwargs):
    """Return a unit Sphere SceneObject with a white Diffuse material."""
    return SceneObject(shape=Sphere(), material=Diffuse(Color(1, 1, 1)), **kwargs)


def _ray(origin, direction):
    return Ray(Vec3(*origin), Vec3(*direction))


# ─── Defaults ─────────────────────────────────────────────────────────────────

def test_default_name():
    obj = _sphere_obj()
    assert obj.name == ""

def test_default_visible():
    obj = _sphere_obj()
    assert obj.visible is True

def test_default_renderable():
    obj = _sphere_obj()
    assert obj.renderable is True

def test_default_selectable():
    obj = _sphere_obj()
    assert obj.selectable is True

def test_default_translation_is_zero():
    obj = _sphere_obj()
    assert vec3_approx_eq(obj.translation, Vec3(0, 0, 0))

def test_default_scale_is_one():
    obj = _sphere_obj()
    assert vec3_approx_eq(obj.scale, Vec3(1, 1, 1))

def test_default_rotation_is_zero():
    obj = _sphere_obj()
    assert vec3_approx_eq(obj.rotation, Vec3(0, 0, 0))


# ─── Property setters ─────────────────────────────────────────────────────────

def test_name_setter():
    obj = _sphere_obj()
    obj.name = "MySphere"
    assert obj.name == "MySphere"

def test_visible_setter():
    obj = _sphere_obj()
    obj.visible = False
    assert obj.visible is False

def test_renderable_setter():
    obj = _sphere_obj()
    obj.renderable = False
    assert obj.renderable is False

def test_selectable_setter():
    obj = _sphere_obj()
    obj.selectable = False
    assert obj.selectable is False

def test_translation_setter_delegates_to_transform():
    obj = _sphere_obj()
    obj.translation = Vec3(3, 0, 0)
    assert vec3_approx_eq(obj.translation, Vec3(3, 0, 0))

def test_scale_setter_delegates_to_transform():
    obj = _sphere_obj()
    obj.scale = Vec3(2, 2, 2)
    assert vec3_approx_eq(obj.scale, Vec3(2, 2, 2))

def test_rotation_setter_delegates_to_transform():
    obj = _sphere_obj()
    obj.rotation = Vec3(0, 45, 0)
    assert vec3_approx_eq(obj.rotation, Vec3(0, 45, 0))

def test_shape_setter_updates_transform_shape():
    obj = _sphere_obj()
    new_shape = Sphere(radius=3.0)
    obj.shape = new_shape
    assert obj.shape is new_shape
    assert obj.transform.shape is new_shape

def test_material_setter():
    obj = _sphere_obj()
    new_mat = Diffuse(Color(0.5, 0.2, 0.8))
    obj.material = new_mat
    assert obj.material is new_mat


# ─── Ray intersection ─────────────────────────────────────────────────────────

def test_intersect_ray_hits_sphere():
    obj = _sphere_obj()
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert hit is not None

def test_intersect_ray_misses_sphere():
    obj = _sphere_obj()
    hit = obj.intersect(_ray((0, 5, -5), (0, 0, 1)))
    assert hit is None

def test_intersect_ray_behind_sphere():
    obj = _sphere_obj()
    hit = obj.intersect(_ray((0, 0, 5), (0, 0, 1)))
    assert hit is None

def test_intersect_hit_has_material():
    mat = Diffuse(Color(0.8, 0.2, 0.4))
    obj = SceneObject(shape=Sphere(), material=mat)
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert hit.material is mat

def test_intersect_hit_normal_is_unit_length():
    obj = _sphere_obj()
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert approx_eq(hit.normal.length(), 1.0), \
        f"Expected unit normal, got length {hit.normal.length()}"

def test_intersect_front_face():
    obj = _sphere_obj()
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert hit.front_face

def test_intersect_translated_sphere_hit():
    obj = _sphere_obj(translation=Vec3(5, 0, 0))
    hit = obj.intersect(_ray((5, 0, -5), (0, 0, 1)))
    assert hit is not None

def test_intersect_translated_sphere_miss():
    obj = _sphere_obj(translation=Vec3(5, 0, 0))
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert hit is None

def test_intersect_scaled_sphere_hit_at_correct_distance():
    # Scale sphere to radius 2 — ray from (0,0,-5) should hit closer (at z=-2).
    obj = _sphere_obj(scale=Vec3(2, 2, 2))
    hit = obj.intersect(_ray((0, 0, -5), (0, 0, 1)))
    assert hit is not None
    assert approx_eq(hit.t, 3.0), f"Expected t=3, got {hit.t}"


# ─── Repr ─────────────────────────────────────────────────────────────────────

def test_repr_contains_name():
    obj = _sphere_obj(name="Orb")
    assert "Orb" in repr(obj)

def test_repr_contains_scene_object():
    obj = _sphere_obj()
    assert "SceneObject" in repr(obj)


if __name__ == "__main__":
    tests = [
        test_default_name,
        test_default_visible,
        test_default_renderable,
        test_default_selectable,
        test_default_translation_is_zero,
        test_default_scale_is_one,
        test_default_rotation_is_zero,
        test_name_setter,
        test_visible_setter,
        test_renderable_setter,
        test_selectable_setter,
        test_translation_setter_delegates_to_transform,
        test_scale_setter_delegates_to_transform,
        test_rotation_setter_delegates_to_transform,
        test_shape_setter_updates_transform_shape,
        test_material_setter,
        test_intersect_ray_hits_sphere,
        test_intersect_ray_misses_sphere,
        test_intersect_ray_behind_sphere,
        test_intersect_hit_has_material,
        test_intersect_hit_normal_is_unit_length,
        test_intersect_front_face,
        test_intersect_translated_sphere_hit,
        test_intersect_translated_sphere_miss,
        test_intersect_scaled_sphere_hit_at_correct_distance,
        test_repr_contains_name,
        test_repr_contains_scene_object,
    ]
    run_tests(tests)
