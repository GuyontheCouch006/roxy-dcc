import math
import taichi as ti

from rendering.taichi.fields import (
    _pixels, _obj_albedo, _obj_mat_type,
    _obj_roughness, _obj_ior, _obj_emission, 
    _accumulator, _frame_count,
)
from rendering.taichi.sky import sky_color
from rendering.taichi.camera import get_ray_direction
from rendering.taichi.intersect import scene_intersect
from rendering.taichi.scatter import (
    diffuse_scatter, metal_scatter, dielectric_scatter
)


@ti.func
def trace(ro, rd, max_depth, use_sky, bg_color):
    color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    current_ior = 1.0  # start in air

    for depth in range(max_depth):
        t, normal, idx = scene_intersect(ro, rd)

        if t < 0.0:
            if use_sky:
                color = throughput * sky_color(rd)
            else:
                color = throughput * bg_color
            break

        front_face = normal.dot(rd) < 0.0
        if not front_face:
            normal = -normal

        hit_p = ro + rd * t
        mat_type = _obj_mat_type[idx]
        albedo = _obj_albedo[idx]

        if mat_type == 3:  # emissive — add light and terminate
            color += throughput * albedo * _obj_emission[idx]
            break

        elif mat_type == 0:  # diffuse
            rd = diffuse_scatter(normal)
            throughput *= albedo

        elif mat_type == 1:  # metal
            rd = metal_scatter(rd, normal, _obj_roughness[idx])
            throughput *= albedo

        elif mat_type == 2:  # dielectric
            obj_ior = _obj_ior[idx]
            eta = 1.0 # taichi doesn't like uninitialized variables, so we have to set it to something first
            current_ior = 1.0 # taichi doesn't like uninitialized variables, so we have to set it to something first
            if front_face:
                eta = current_ior / obj_ior
                current_ior = obj_ior
            else:
                eta = current_ior / 1.0
                current_ior = 1.0
            rd = dielectric_scatter(rd, normal, eta)
            throughput *= albedo

        # Russian roulette
        if depth > 2:
            rr = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
            if ti.random() > rr:
                break
            throughput /= rr

        ro = hit_p + normal * 1e-4

    return color


@ti.kernel
def render_kernel(
    W: int, H: int,
    fov_tan: float, aspect: float,
    max_depth: int,
    use_sky: int,
    bg_color: ti.types.vector(3, ti.f32),
    cam_pos:   ti.types.vector(3, ti.f32),
    cam_fwd:   ti.types.vector(3, ti.f32),
    cam_right: ti.types.vector(3, ti.f32),
    cam_up:    ti.types.vector(3, ti.f32),
):
    frame = _frame_count[None]
    for y, x in ti.ndrange(H, W):
        accum = ti.Vector([0.0, 0.0, 0.0])
        rd = get_ray_direction(x, y, W, H, fov_tan, aspect,
                                cam_fwd, cam_right, cam_up)
        sample = trace(cam_pos, rd, max_depth, use_sky, bg_color)

        # running average
        if frame == 0:
            _accumulator[y, x] = sample
        else:
            _accumulator[y, x] = (_accumulator[y, x] * frame + sample) / (frame + 1)

        # color = accum / samples
        _pixels[y, x] = ti.sqrt(ti.min(_accumulator[y, x], 1.0))
