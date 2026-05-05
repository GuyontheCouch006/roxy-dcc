# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Mat4x4 — arithmetic, point/vector transforms, translation,
#              scale, rotation, composition order, inversion, and transpose.
# ============================================

from core import Mat4x4, Vec3
from tests.utils import run_tests, approx_eq



def test_id_times_id():
    identity = Mat4x4.identity()
    result = identity * identity
    for i in range(4):
        for j in range(4):
            assert approx_eq(result.rows[i][j], identity.rows[i][j]), f"Expected {identity.rows[i][j]}, got {result.rows[i][j]}"

def test_id_times_scalar():
    identity = Mat4x4.identity()
    scalar = 2
    result = identity * scalar
    for i in range(4):
        for j in range(4):
            expected = identity.rows[i][j] * scalar
            assert approx_eq(result.rows[i][j], expected), f"Expected {expected}, got {result.rows[i][j]}"

def test_scalar_times_id():
    identity = Mat4x4.identity()
    scalar = 2
    result = scalar * identity
    for i in range(4):
        for j in range(4):
            expected = identity.rows[i][j] * scalar
            assert approx_eq(result.rows[i][j], expected), f"Expected {expected}, got {result.rows[i][j]}"

def test_id_times_vec3():
    identity = Mat4x4.identity()
    vec = Vec3(1, 2, 3)
    result = identity.transform_vector(vec)
    assert approx_eq(result.x, vec.x), f"Expected {vec.x}, got {result.x}"
    assert approx_eq(result.y, vec.y), f"Expected {vec.y}, got {result.y}"
    assert approx_eq(result.z, vec.z), f"Expected {vec.z}, got {result.z}" 

def test_transform_point():
    identity = Mat4x4.identity()
    point = Vec3(1, 2, 3)
    result = identity.transform_point(point)
    assert approx_eq(result.x, point.x), f"Expected {point.x}, got {result.x}"
    assert approx_eq(result.y, point.y), f"Expected {point.y}, got {result.y}"
    assert approx_eq(result.z, point.z), f"Expected {point.z}, got {result.z}"  

def test_transform_vector():
    identity = Mat4x4.identity()
    vector = Vec3(1, 2, 3)
    result = identity.transform_vector(vector)
    assert approx_eq(result.x, vector.x), f"Expected {vector.x}, got {result.x}"
    assert approx_eq(result.y, vector.y), f"Expected {vector.y}, got {result.y}"
    assert approx_eq(result.z, vector.z), f"Expected {vector.z}, got {result.z}"    

def test_translation_moves_point():
    t = Mat4x4.translation(5, 0, 0)
    result = t.transform_point(Vec3(0, 0, 0))
    assert approx_eq(result.x, 5), f"Expected 5, got {result.x}"
    assert approx_eq(result.y, 0), f"Expected 0, got {result.y}"
    assert approx_eq(result.z, 0), f"Expected 0, got {result.z}"

def test_translation_does_not_move_vector():
    t = Mat4x4.translation(5, 0, 0)
    result = t.transform_vector(Vec3(1, 0, 0))
    assert approx_eq(result.x, 1), f"Expected 1, got {result.x}"
    assert approx_eq(result.y, 0), f"Expected 0, got {result.y}"
    assert approx_eq(result.z, 0), f"Expected 0, got {result.z}"

def test_scale():
    s = Mat4x4.scale(2, 3, 4)
    result = s.transform_point(Vec3(1, 1, 1))
    assert approx_eq(result.x, 2), f"Expected 2, got {result.x}"
    assert approx_eq(result.y, 3), f"Expected 3, got {result.y}"
    assert approx_eq(result.z, 4), f"Expected 4, got {result.z}"

def test_rotation_z_90():
    r = Mat4x4.rotation_z(90)
    result = r.transform_point(Vec3(1, 0, 0))
    assert approx_eq(result.x, 0), f"Expected 0, got {result.x}"
    assert approx_eq(result.y, 1), f"Expected 1, got {result.y}"
    assert approx_eq(result.z, 0), f"Expected 0, got {result.z}"

def test_composition_order_matters():
    # translate then scale vs scale then translate — must be different
    T = Mat4x4.translation(2, 0, 0)
    S = Mat4x4.scale(3, 3, 3)
    p = Vec3(1, 0, 0)
    ts = (T * S).transform_point(p)   # scale first, then translate
    st = (S * T).transform_point(p)   # translate first, then scale
    assert not (approx_eq(ts.x, st.x) and approx_eq(ts.y, st.y)), \
        "T*S and S*T should produce different results"

def test_inverse():
    T = Mat4x4.translation(3, 5, -2)
    R = Mat4x4.rotation_y(45)
    S = Mat4x4.scale(2, 2, 2)
    M = T * R * S
    M_inv = M.inverse()
    result = M * M_inv
    identity = Mat4x4.identity()
    for i in range(4):
        for j in range(4):
            assert approx_eq(result.rows[i][j], identity.rows[i][j]), \
                f"M * M_inv should be identity at [{i}][{j}]"

def test_transpose_twice():
    T = Mat4x4.translation(1, 2, 3)
    assert T.transpose().transpose() == T, "Transpose twice should equal original"

if __name__ == "__main__":
    tests = [
        test_id_times_id,
        test_id_times_scalar,
        test_scalar_times_id,
        test_id_times_vec3,
        test_transform_point,
        test_transform_vector,
        test_translation_moves_point,
        test_translation_does_not_move_vector,
        test_scale,
        test_rotation_z_90,
        test_composition_order_matters,
        test_inverse,
        test_transpose_twice,
    ]

    run_tests(tests)