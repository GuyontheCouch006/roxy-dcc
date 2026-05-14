import taichi as ti

from scene.primitives import Sphere, Plane, Cube
from scene.mesh import Triangle
from rendering.taichi.fields import (
    _obj_type, _obj_center, _obj_radius,
    _obj_normal, _obj_offset, _obj_extra, _n_objects,
    _obj_v0, _obj_v1, _obj_v2,
    _bvh_aabb_min, _bvh_aabb_max, _bvh_left, _bvh_right,
    _bvh_tri_start, _bvh_tri_count, _bvh_n_tris,
    _bvh_v0, _bvh_v1, _bvh_v2,
    _bvh_n0, _bvh_n1, _bvh_n2,
    _bvh_mat_idx,
)


@ti.func
def aabb_hit(ro, rd, aabb_min, aabb_max):
    inv_d = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        if ti.abs(rd[k]) > 1e-10:
            inv_d[k] = 1.0 / rd[k]
        else:
            inv_d[k] = 1e30

    t_min = (aabb_min - ro) * inv_d
    t_max = (aabb_max - ro) * inv_d

    t1 = ti.min(t_min, t_max)
    t2 = ti.max(t_min, t_max)

    t_enter = ti.max(t1[0], ti.max(t1[1], t1[2]))
    t_exit  = ti.min(t2[0], ti.min(t2[1], t2[2]))

    return t_exit >= t_enter and t_exit >= 0.001


@ti.func
def bvh_triangle_intersect(ro, rd, v0, v1, v2):
    """Möller–Trumbore. Returns (t, u, v); t=-1 on miss."""
    e1 = v1 - v0
    e2 = v2 - v0
    h  = rd.cross(e2)
    a  = e1.dot(h)
    t  = -1.0
    u  = 0.0
    v  = 0.0
    if ti.abs(a) > 1e-8:
        f = 1.0 / a
        s = ro - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(e1)
            v = f * rd.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                tc = f * e2.dot(q)
                if tc > 0.001:
                    t = tc
    return t, u, v


@ti.func
def bvh_intersect(ro, rd):
    """Traverse BVH with an explicit stack. Returns (t, tri_idx, u, v)."""
    # Stack depth 64 handles trees up to depth 32 (2 children pushed per level).
    stack     = ti.Vector.zero(ti.i32, 64)
    stack_ptr = 0
    stack[0]  = 0   # start at root

    closest_t   = 1e9
    closest_idx = -1
    closest_u   = 0.0
    closest_v   = 0.0

    while stack_ptr >= 0:
        node_idx  = stack[stack_ptr]
        stack_ptr -= 1

        if not aabb_hit(ro, rd, _bvh_aabb_min[node_idx], _bvh_aabb_max[node_idx]):
            continue

        if _bvh_tri_count[node_idx] > 0:   # leaf
            for i in range(_bvh_tri_start[node_idx],
                           _bvh_tri_start[node_idx] + _bvh_tri_count[node_idx]):
                t, u, v = bvh_triangle_intersect(
                    ro, rd, _bvh_v0[i], _bvh_v1[i], _bvh_v2[i])
                if 0.001 < t < closest_t:
                    closest_t   = t
                    closest_idx = i
                    closest_u   = u
                    closest_v   = v
        else:                               # interior — push both children
            stack_ptr += 1
            stack[stack_ptr] = _bvh_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = _bvh_right[node_idx]

    return closest_t, closest_idx, closest_u, closest_v


@ti.func
def bvh_occluded(ro, rd, max_t):
    """Return 1 if any BVH triangle blocks the ray before max_t."""
    occluded = 0

    stack     = ti.Vector.zero(ti.i32, 64)
    stack_ptr = 0
    stack[0]  = 0

    while stack_ptr >= 0 and occluded == 0:
        node_idx  = stack[stack_ptr]
        stack_ptr -= 1

        if not aabb_hit(ro, rd, _bvh_aabb_min[node_idx], _bvh_aabb_max[node_idx]):
            continue

        if _bvh_tri_count[node_idx] > 0:
            for i in range(_bvh_tri_start[node_idx],
                           _bvh_tri_start[node_idx] + _bvh_tri_count[node_idx]):
                t, u, v = bvh_triangle_intersect(
                    ro, rd, _bvh_v0[i], _bvh_v1[i], _bvh_v2[i])
                if 0.001 < t < max_t:
                    occluded = 1
        else:
            stack_ptr += 1
            stack[stack_ptr] = _bvh_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = _bvh_right[node_idx]

    return occluded


@ti.func
def scene_occluded(ro, rd, max_t):
    """Any-hit shadow query. Returns 1 if a primitive or BVH tri blocks the ray."""
    occluded = 0

    n = _n_objects[None]
    for i in range(n):
        obj_t = -1.0

        if _obj_type[i] == 0:
            obj_t = Sphere.taichi_intersect(ro, rd, _obj_center[i], _obj_radius[i])
        elif _obj_type[i] == 1:
            obj_t = Plane.taichi_intersect(ro, rd, _obj_normal[i], _obj_offset[i])
        elif _obj_type[i] == 2:
            obj_t = Cube.taichi_intersect(ro, rd, _obj_center[i], _obj_extra[i])
        elif _obj_type[i] == 4:
            obj_t = Triangle.taichi_intersect(ro, rd, _obj_v0[i], _obj_v1[i], _obj_v2[i])

        if 0.001 < obj_t < max_t:
            occluded = 1

    if occluded == 0 and _bvh_n_tris[None] > 0:
        occluded = bvh_occluded(ro, rd, max_t)

    return occluded


@ti.func
def scene_intersect(ro, rd):
    closest_t      = 1e9
    closest_normal = ti.Vector([0.0, 1.0, 0.0])
    closest_idx    = -1
    is_bvh_hit     = 0
    bvh_mat_idx    = 0

    # ── Primitives — linear scan ─────────────────────────────────────────────
    n = _n_objects[None]
    for i in range(n):
        obj_t      = -1.0
        obj_normal = ti.Vector([0.0, 0.0, 0.0])

        if _obj_type[i] == 0:   # Sphere
            obj_t = Sphere.taichi_intersect(ro, rd, _obj_center[i], _obj_radius[i])
            if obj_t > 0.001:
                hit_p      = ro + rd * obj_t
                obj_normal = (hit_p - _obj_center[i]).normalized()
        elif _obj_type[i] == 1:  # Plane
            obj_t = Plane.taichi_intersect(ro, rd, _obj_normal[i], _obj_offset[i])
            if obj_t > 0.001:
                obj_normal = _obj_normal[i]
        elif _obj_type[i] == 2:  # Cube
            obj_t = Cube.taichi_intersect(ro, rd, _obj_center[i], _obj_extra[i])
            if obj_t > 0.001:
                hit_p      = ro + rd * obj_t
                obj_normal = Cube.taichi_normal(hit_p, _obj_center[i], _obj_extra[i])
        elif _obj_type[i] == 4:  # Triangle (legacy linear slot)
            obj_t = Triangle.taichi_intersect(ro, rd, _obj_v0[i], _obj_v1[i], _obj_v2[i])
            if obj_t > 0.001:
                obj_normal = Triangle.taichi_normal(_obj_v0[i], _obj_v1[i], _obj_v2[i])

        if obj_t > 0.001 and obj_t < closest_t:
            closest_t      = obj_t
            closest_normal = obj_normal
            closest_idx    = i

    # ── Mesh triangles — BVH traversal ───────────────────────────────────────
    if _bvh_n_tris[None] > 0:
        t, tri_idx, u, v = bvh_intersect(ro, rd)
        if 0.001 < t < closest_t:
            closest_t   = t
            closest_idx = tri_idx
            is_bvh_hit  = 1
            bvh_mat_idx = _bvh_mat_idx[tri_idx]
            w = 1.0 - u - v
            closest_normal = (
                _bvh_n0[tri_idx] * w +
                _bvh_n1[tri_idx] * u +
                _bvh_n2[tri_idx] * v
            ).normalized()

    return closest_t, closest_normal, closest_idx, is_bvh_hit, bvh_mat_idx
