import math
import taichi as ti

from rendering.taichi.fields import (
    _pixels, _obj_albedo, _obj_mat_type,
    _obj_roughness, _obj_ior, _obj_emission,
    _accumulator, _frame_count,
    _mat_type, _mat_albedo, _mat_roughness, _mat_ior, _mat_emission,
)
from rendering.taichi.sky import sky_color
from rendering.taichi.camera import get_ray_direction
from rendering.taichi.intersect import scene_intersect
from rendering.taichi.scatter import (
    diffuse_scatter, metal_scatter, dielectric_scatter, glossy_scatter,
)


@ti.func
def trace(ro, rd, max_depth, use_sky, bg_color):
    color       = ti.Vector([0.0, 0.0, 0.0])
    throughput  = ti.Vector([1.0, 1.0, 1.0])
    current_ior = 1.0

    for depth in range(max_depth):
        t, normal, idx, is_bvh, bvh_mat = scene_intersect(ro, rd)

        if t < 0.0 or t >= 1e9:
            if use_sky:
                color = throughput * sky_color(rd)
            else:
                color = throughput * bg_color
            break

        front_face = normal.dot(rd) < 0.0
        if not front_face:
            normal = -normal

        hit_p = ro + rd * t

        # ── Material lookup ───────────────────────────────────────────────────
        mat_type  = 0
        albedo    = ti.Vector([0.8, 0.8, 0.8])
        roughness = 0.0
        ior       = 1.0
        emission  = 0.0

        if is_bvh:
            mat_type  = _mat_type[bvh_mat]
            albedo    = _mat_albedo[bvh_mat]
            roughness = _mat_roughness[bvh_mat]
            ior       = _mat_ior[bvh_mat]
            emission  = _mat_emission[bvh_mat]
        else:
            mat_type  = _obj_mat_type[idx]
            albedo    = _obj_albedo[idx]
            roughness = _obj_roughness[idx]
            ior       = _obj_ior[idx]
            emission  = _obj_emission[idx]

        # ── Shading ───────────────────────────────────────────────────────────
        if mat_type == 3:   # emissive
            color += throughput * albedo * emission
            break

        elif mat_type == 0:  # diffuse
            rd = diffuse_scatter(normal)
            throughput *= albedo

        elif mat_type == 1:  # metal
            rd = metal_scatter(rd, normal, roughness)
            throughput *= albedo

        elif mat_type == 4:  # glossy
            rd = glossy_scatter(rd, normal, roughness)
            throughput *= albedo

        elif mat_type == 2:  # dielectric
            eta = 1.0
            if front_face:
                eta = current_ior / ior
                current_ior = ior
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
        rd = get_ray_direction(x, y, W, H, fov_tan, aspect,
                               cam_fwd, cam_right, cam_up)
        sample = trace(cam_pos, rd, max_depth, use_sky, bg_color)

        if frame == 0:
            _accumulator[y, x] = sample
        else:
            _accumulator[y, x] = (_accumulator[y, x] * frame + sample) / (frame + 1)

        _pixels[y, x] = ti.sqrt(ti.min(_accumulator[y, x], 1.0))
