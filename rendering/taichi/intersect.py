import taichi as ti

from scene.shapes import Sphere, Plane, Cube
from rendering.taichi.fields import (
    _obj_type, _obj_center, _obj_radius,
    _obj_normal, _obj_offset, _obj_extra, _n_objects,
)


@ti.func
def scene_intersect(ro, rd):
    best_t      = -1.0
    best_normal = ti.Vector([0.0, 0.0, 0.0])
    best_idx    = -1

    n = _n_objects[None]
    for i in range(n):
        obj_t      = -1.0
        obj_normal = ti.Vector([0.0, 0.0, 0.0])

        if _obj_type[i] == 0:  # Sphere
            obj_t = Sphere.taichi_intersect(ro, rd, _obj_center[i], _obj_radius[i])
            if obj_t > 0.0:
                hit_p = ro + rd * obj_t
                obj_normal = (hit_p - _obj_center[i]).normalized()
        elif _obj_type[i] == 1:  # Plane
            obj_t = Plane.taichi_intersect(ro, rd, _obj_normal[i], _obj_offset[i])
            if obj_t > 0.0:
                obj_normal = _obj_normal[i]
        elif _obj_type[i] == 2:  # Cube
            obj_t = Cube.taichi_intersect(ro, rd, _obj_center[i], _obj_extra[i])
            if obj_t > 0.0:
                hit_p = ro + rd * obj_t
                obj_normal = Cube.taichi_normal(hit_p, _obj_center[i], _obj_extra[i])

        if obj_t > 0.001:
            if best_t < 0.0 or obj_t < best_t:
                best_t      = obj_t
                best_normal = obj_normal
                best_idx    = i

    return best_t, best_normal, best_idx
