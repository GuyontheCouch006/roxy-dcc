# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Transform — defaults, dirty flag lifecycle, translation,
#              scale, rotation, rotation order, pivot, and inverse correctness.
# ============================================

import math
from core import Transform, Vec3, RotationOrder
from tests.utils import run_tests, approx_eq, vec3_approx_eq


# --- defaults ---

def test_default_translation():
    t = Transform()
    assert vec3_approx_eq(t.translation, Vec3(0, 0, 0))

def test_default_scale():
    t = Transform()
    assert vec3_approx_eq(t.scale, Vec3(1, 1, 1))

def test_default_rotation():
    t = Transform()
    assert vec3_approx_eq(t.rotation, Vec3(0, 0, 0))

def test_default_not_dirty():
    t = Transform()
    assert not t.dirty

def test_default_matrix_is_identity():
    t = Transform()
    identity = t.world_matrix
    for i in range(4):
        for j in range(4):
            expected = 1.0 if i == j else 0.0
            assert approx_eq(identity.rows[i][j], expected), \
                f"[{i}][{j}] expected {expected}, got {identity.rows[i][j]}"


# --- dirty flag ---

def test_setter_marks_dirty():
    t = Transform()
    t.translation = Vec3(1, 0, 0)
    assert t.dirty

def test_accessing_world_matrix_clears_dirty():
    t = Transform()
    t.translation = Vec3(1, 0, 0)
    _ = t.world_matrix
    assert not t.dirty

def test_accessing_world_inverse_clears_dirty():
    t = Transform()
    t.scale = Vec3(2, 2, 2)
    _ = t.world_inverse_matrix
    assert not t.dirty

def test_scale_setter_marks_dirty():
    t = Transform()
    t.scale = Vec3(2, 2, 2)
    assert t.dirty

def test_rotation_setter_marks_dirty():
    t = Transform()
    t.rotation = Vec3(0, 90, 0)
    assert t.dirty

def test_pivot_setter_marks_dirty():
    t = Transform()
    t.pivot = Vec3(1, 0, 0)
    assert t.dirty

def test_rotation_order_setter_marks_dirty():
    t = Transform()
    t.rotation_order = RotationOrder.ZYX
    assert t.dirty


# --- translation ---

def test_translation_moves_point():
    t = Transform(translation=Vec3(3, 0, 0))
    result = t.world_matrix.transform_point(Vec3(0, 0, 0))
    assert vec3_approx_eq(result, Vec3(3, 0, 0)), f"Expected (3,0,0), got {result}"

def test_translation_does_not_move_vector():
    t = Transform(translation=Vec3(3, 0, 0))
    result = t.world_matrix.transform_vector(Vec3(1, 0, 0))
    assert vec3_approx_eq(result, Vec3(1, 0, 0)), f"Vector should be unaffected by translation"


# --- scale ---

def test_scale_stretches_point():
    t = Transform(scale=Vec3(2, 3, 4))
    result = t.world_matrix.transform_point(Vec3(1, 1, 1))
    assert vec3_approx_eq(result, Vec3(2, 3, 4)), f"Expected (2,3,4), got {result}"

def test_uniform_scale():
    t = Transform(scale=Vec3(5, 5, 5))
    result = t.world_matrix.transform_point(Vec3(1, 0, 0))
    assert vec3_approx_eq(result, Vec3(5, 0, 0)), f"Expected (5,0,0), got {result}"


# --- rotation ---

def test_rotation_z_90_moves_x_to_y():
    t = Transform(rotation=Vec3(0, 0, 90))
    result = t.world_matrix.transform_point(Vec3(1, 0, 0))
    assert approx_eq(result.x, 0.0) and approx_eq(result.y, 1.0) and approx_eq(result.z, 0.0), \
        f"Expected (0,1,0), got {result}"

def test_rotation_x_90_moves_y_to_z():
    t = Transform(rotation=Vec3(90, 0, 0))
    result = t.world_matrix.transform_point(Vec3(0, 1, 0))
    assert approx_eq(result.x, 0.0) and approx_eq(result.y, 0.0) and approx_eq(result.z, 1.0), \
        f"Expected (0,0,1), got {result}"

def test_rotation_order_affects_result():
    r = Vec3(30, 45, 60)
    t_xyz = Transform(rotation=r, rotation_order=RotationOrder.XYZ)
    t_zyx = Transform(rotation=r, rotation_order=RotationOrder.ZYX)
    m_xyz = t_xyz.world_matrix
    m_zyx = t_zyx.world_matrix
    # At least one element should differ between the two orderings
    any_diff = any(
        not approx_eq(m_xyz.rows[i][j], m_zyx.rows[i][j])
        for i in range(4) for j in range(4)
    )
    assert any_diff, "Different rotation orders should produce different matrices"


# --- pivot ---

def test_pivot_shifts_rotation_center():
    # Rotate 90° around Z with pivot at (1,0,0).
    # The pivot moves to itself, so a point at (1,0,0) stays at (1,0,0).
    t = Transform(rotation=Vec3(0, 0, 90), pivot=Vec3(1, 0, 0))
    result = t.world_matrix.transform_point(Vec3(1, 0, 0))
    assert vec3_approx_eq(result, Vec3(1, 0, 0)), f"Pivot point should stay fixed: got {result}"


# --- inverse ---

def test_inverse_undoes_translation():
    t = Transform(translation=Vec3(5, -3, 2))
    point = Vec3(1, 1, 1)
    transformed = t.world_matrix.transform_point(point)
    restored = t.world_inverse_matrix.transform_point(transformed)
    assert vec3_approx_eq(restored, point), f"Expected {point}, got {restored}"

def test_inverse_undoes_scale():
    t = Transform(scale=Vec3(2, 4, 8))
    point = Vec3(1, 1, 1)
    transformed = t.world_matrix.transform_point(point)
    restored = t.world_inverse_matrix.transform_point(transformed)
    assert vec3_approx_eq(restored, point), f"Expected {point}, got {restored}"

def test_inverse_undoes_rotation():
    t = Transform(rotation=Vec3(30, 45, 60))
    point = Vec3(1, 2, 3)
    transformed = t.world_matrix.transform_point(point)
    restored = t.world_inverse_matrix.transform_point(transformed)
    assert vec3_approx_eq(restored, point), f"Expected {point}, got {restored}"


# --- repr ---

def test_repr_contains_components():
    t = Transform(translation=Vec3(1, 2, 3))
    r = repr(t)
    assert "Transform" in r and "1" in r


if __name__ == "__main__":
    tests = [
        test_default_translation,
        test_default_scale,
        test_default_rotation,
        test_default_not_dirty,
        test_default_matrix_is_identity,
        test_setter_marks_dirty,
        test_accessing_world_matrix_clears_dirty,
        test_accessing_world_inverse_clears_dirty,
        test_scale_setter_marks_dirty,
        test_rotation_setter_marks_dirty,
        test_pivot_setter_marks_dirty,
        test_rotation_order_setter_marks_dirty,
        test_translation_moves_point,
        test_translation_does_not_move_vector,
        test_scale_stretches_point,
        test_uniform_scale,
        test_rotation_z_90_moves_x_to_y,
        test_rotation_x_90_moves_y_to_z,
        test_rotation_order_affects_result,
        test_pivot_shifts_rotation_center,
        test_inverse_undoes_translation,
        test_inverse_undoes_scale,
        test_inverse_undoes_rotation,
        test_repr_contains_components,
    ]
    run_tests(tests)
