import math
import taichi as ti

from rendering.taichi.fields import (
    _pixels, _obj_albedo, _obj_mat_type,
    _obj_roughness, _obj_ior, _obj_emission,
    _accumulator, _frame_count, _ray_count,
    _mat_type, _mat_albedo, _mat_roughness, _mat_ior, _mat_emission,
    _light_center, _light_radius, _light_albedo, _light_emission, _n_lights,
)
from rendering.taichi.sky import sky_color
from rendering.taichi.camera import get_ray_direction
from rendering.taichi.intersect import scene_intersect, scene_occluded
from rendering.taichi.scatter import (
    diffuse_scatter, metal_scatter, dielectric_scatter, glossy_scatter,
)


@ti.func
def direct_light_sample(hit_p, normal, albedo, direct_light_mode):
    direct = ti.Vector([0.0, 0.0, 0.0])
    light_count = _n_lights[None]
    if light_count > 0:
        selected_light = ti.min(ti.cast(ti.random() * light_count, ti.i32), light_count - 1)

        for i in range(light_count):
            if direct_light_mode == 1 or i == selected_light:
                to_light = _light_center[i] - hit_p
                dist2 = ti.max(to_light.dot(to_light), 1e-6)
                dist = ti.sqrt(dist2)
                light_dir = to_light / dist
                ndotl = normal.dot(light_dir)

                if ndotl > 0.0:
                    radius = _light_radius[i]
                    max_t = dist - radius * 1.001
                    if max_t > 0.001:
                        shadow_origin = hit_p + normal * 1e-4
                        ti.atomic_add(_ray_count[None], 1)
                        if scene_occluded(shadow_origin, light_dir, max_t) == 0:
                            solid_angle_scale = radius * radius / ti.max(dist2, radius * radius)
                            sample_weight = 1.0
                            if direct_light_mode == 0:
                                sample_weight = ti.cast(light_count, ti.f32)
                            direct += (
                                albedo *
                                _light_albedo[i] *
                                _light_emission[i] *
                                ndotl *
                                solid_angle_scale *
                                sample_weight
                            )

    return direct


@ti.func
def trace(ro, rd, max_depth, use_sky, bg_color, direct_light_mode):
    color       = ti.Vector([0.0, 0.0, 0.0])
    throughput  = ti.Vector([1.0, 1.0, 1.0])
    current_ior = 1.0

    for depth in range(max_depth):
        ti.atomic_add(_ray_count[None], 1)
        t, normal, idx, is_bvh, bvh_mat = scene_intersect(ro, rd)

        if t < 0.0 or t >= 1e9:
            if use_sky:
                color += throughput * sky_color(rd)
            else:
                color += throughput * bg_color
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
            color += throughput * direct_light_sample(hit_p, normal, albedo, direct_light_mode)
            rd = diffuse_scatter(normal)
            throughput *= albedo

        elif mat_type == 1:  # metal
            rd = metal_scatter(rd, normal, roughness)
            throughput *= albedo

        elif mat_type == 4:  # glossy
            color += throughput * direct_light_sample(hit_p, normal, albedo, direct_light_mode) * roughness
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
    direct_light_mode: int,
    sample_clamp: float,
    bg_color: ti.types.vector(3, ti.f32),
    cam_pos:   ti.types.vector(3, ti.f32),
    cam_fwd:   ti.types.vector(3, ti.f32),
    cam_right: ti.types.vector(3, ti.f32),
    cam_up:    ti.types.vector(3, ti.f32),
):
    frame = _frame_count[None]
    for y, x in ti.ndrange(H, W):
        rd = get_ray_direction(x, y, W, H, frame, fov_tan, aspect,
                               cam_fwd, cam_right, cam_up)
        sample = trace(cam_pos, rd, max_depth, use_sky, bg_color, direct_light_mode)
        if sample_clamp > 0.0:
            limit = ti.Vector([sample_clamp, sample_clamp, sample_clamp])
            sample = ti.min(ti.max(sample, ti.Vector([0.0, 0.0, 0.0])), limit)

        if frame == 0:
            _accumulator[y, x] = sample
        else:
            _accumulator[y, x] = (_accumulator[y, x] * frame + sample) / (frame + 1)

        _pixels[y, x] = ti.sqrt(ti.min(_accumulator[y, x], 1.0))
