import math
import taichi as ti

from rendering.taichi.fields import (
    _pixels, _obj_albedo, _obj_mat_type,
    _obj_roughness, _obj_ior, _obj_emission,
    _accumulator, _normal_accumulator, _albedo_accumulator, _depth_accumulator,
    _frame_count, _ray_count,
    _mat_type, _mat_albedo, _mat_roughness, _mat_ior, _mat_emission,
    _mat_texture,
    _bvh_uv0, _bvh_uv1, _bvh_uv2, _bvh_has_uv,
    _tex_pixels, _tex_width, _tex_height, _tex_flip_v,
    _light_center, _light_radius, _light_albedo, _light_emission, _n_lights,
)
from rendering.taichi.sky import sky_color
from rendering.taichi.camera import get_ray_direction
from rendering.taichi.intersect import scene_intersect, scene_occluded
from rendering.taichi.scatter import (
    diffuse_scatter, metal_scatter, dielectric_scatter, glossy_scatter,
)


@ti.func
def _sphere_light_sample(hit_p, center, radius):
    to_center = center - hit_p
    dist2 = ti.max(to_center.dot(to_center), 1e-6)
    dist = ti.sqrt(dist2)
    axis = to_center / dist

    sin2_theta_max = ti.min(radius * radius / dist2, 1.0)
    cos_theta_max = ti.sqrt(ti.max(0.0, 1.0 - sin2_theta_max))
    cos_theta = 1.0 - ti.random() * (1.0 - cos_theta_max)
    sin_theta = ti.sqrt(ti.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * ti.random()

    helper = ti.Vector([1.0, 0.0, 0.0])
    if ti.abs(axis[0]) > 0.9:
        helper = ti.Vector([0.0, 1.0, 0.0])
    bitangent = axis.cross(helper).normalized()
    tangent = bitangent.cross(axis)

    light_dir = (
        tangent * (ti.cos(phi) * sin_theta) +
        bitangent * (ti.sin(phi) * sin_theta) +
        axis * cos_theta
    ).normalized()

    oc = hit_p - center
    half_b = oc.dot(light_dir)
    c = oc.dot(oc) - radius * radius
    discriminant = ti.max(half_b * half_b - c, 0.0)
    t_light = -half_b - ti.sqrt(discriminant)

    # Lambertian 1/pi folded into solid angle. For small lights this tends
    # toward radius^2 / distance^2, matching the old center-sample scale.
    solid_angle_over_pi = 2.0 * (1.0 - cos_theta_max)
    return light_dir, t_light, solid_angle_over_pi


@ti.func
def direct_light_sample(hit_p, normal, albedo, direct_light_mode):
    direct = ti.Vector([0.0, 0.0, 0.0])
    light_count = _n_lights[None]
    if light_count > 0:
        selected_light = ti.min(ti.cast(ti.random() * light_count, ti.i32), light_count - 1)

        for i in range(light_count):
            if direct_light_mode == 1 or i == selected_light:
                radius = _light_radius[i]
                light_dir, t_light, solid_angle_scale = _sphere_light_sample(
                    hit_p, _light_center[i], radius)
                ndotl = normal.dot(light_dir)

                if ndotl > 0.0:
                    max_t = t_light * 0.999
                    if max_t > 0.001:
                        shadow_origin = hit_p + normal * 1e-4
                        ti.atomic_add(_ray_count[None], 1)
                        if scene_occluded(shadow_origin, light_dir, max_t) == 0:
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
def _sample_texture(tex_idx, uv):
    result = ti.Vector([1.0, 1.0, 1.0])
    width = _tex_width[tex_idx]
    height = _tex_height[tex_idx]

    if width > 0 and height > 0:
        u = uv[0] - ti.floor(uv[0])
        v = uv[1] - ti.floor(uv[1])
        if _tex_flip_v[tex_idx] != 0:
            v = 1.0 - v

        x = u * ti.cast(width - 1, ti.f32)
        y = v * ti.cast(height - 1, ti.f32)
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(x0 + 1, width - 1)
        y1 = ti.min(y0 + 1, height - 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        c00 = _tex_pixels[tex_idx, y0, x0].cast(ti.f32) / 255.0
        c10 = _tex_pixels[tex_idx, y0, x1].cast(ti.f32) / 255.0
        c01 = _tex_pixels[tex_idx, y1, x0].cast(ti.f32) / 255.0
        c11 = _tex_pixels[tex_idx, y1, x1].cast(ti.f32) / 255.0
        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        result = c0 * (1.0 - ty) + c1 * ty

    return result


@ti.func
def _bvh_albedo(tri_idx, mat_idx, bary_u, bary_v, base_albedo):
    albedo = base_albedo
    tex_idx = _mat_texture[mat_idx]
    if tex_idx >= 0 and _bvh_has_uv[tri_idx] != 0:
        w = 1.0 - bary_u - bary_v
        uv = _bvh_uv0[tri_idx] * w + _bvh_uv1[tri_idx] * bary_u + _bvh_uv2[tri_idx] * bary_v
        albedo = base_albedo * _sample_texture(tex_idx, uv)
    return albedo


@ti.func
def trace(ro, rd, max_depth, use_sky, bg_color,
          direct_light_mode, direct_light_max_depth):
    color       = ti.Vector([0.0, 0.0, 0.0])
    throughput  = ti.Vector([1.0, 1.0, 1.0])
    current_ior = 1.0
    first_normal = ti.Vector([0.0, 0.0, 0.0])
    first_albedo = ti.Vector([0.0, 0.0, 0.0])
    first_depth = 0.0

    for depth in range(max_depth):
        ti.atomic_add(_ray_count[None], 1)
        t, normal, idx, is_bvh, bvh_mat, bary_u, bary_v = scene_intersect(ro, rd)

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
            albedo    = _bvh_albedo(idx, bvh_mat, bary_u, bary_v, albedo)
        else:
            mat_type  = _obj_mat_type[idx]
            albedo    = _obj_albedo[idx]
            roughness = _obj_roughness[idx]
            ior       = _obj_ior[idx]
            emission  = _obj_emission[idx]

        if depth == 0:
            first_normal = normal
            first_albedo = albedo
            first_depth = t

        # ── Shading ───────────────────────────────────────────────────────────
        if mat_type == 3:   # emissive
            color += throughput * albedo * emission
            break

        elif mat_type == 0:  # diffuse
            if depth < direct_light_max_depth:
                color += throughput * direct_light_sample(
                    hit_p, normal, albedo, direct_light_mode)
            rd = diffuse_scatter(normal)
            throughput *= albedo

        elif mat_type == 1:  # metal
            rd = metal_scatter(rd, normal, roughness)
            throughput *= albedo

        elif mat_type == 4:  # glossy
            if depth < direct_light_max_depth:
                color += throughput * direct_light_sample(
                    hit_p, normal, albedo, direct_light_mode) * roughness
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

    return color, first_normal, first_albedo, first_depth


@ti.kernel
def render_kernel(
    W: int, H: int,
    fov_tan: float, aspect: float,
    max_depth: int,
    use_sky: int,
    direct_light_mode: int,
    direct_light_max_depth: int,
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
        sample, normal_sample, albedo_sample, depth_sample = trace(
            cam_pos, rd, max_depth, use_sky, bg_color,
            direct_light_mode, direct_light_max_depth)
        if sample_clamp > 0.0:
            limit = ti.Vector([sample_clamp, sample_clamp, sample_clamp])
            sample = ti.min(ti.max(sample, ti.Vector([0.0, 0.0, 0.0])), limit)

        if frame == 0:
            _accumulator[y, x] = sample
            _normal_accumulator[y, x] = normal_sample
            _albedo_accumulator[y, x] = albedo_sample
            _depth_accumulator[y, x] = depth_sample
        else:
            _accumulator[y, x] = (_accumulator[y, x] * frame + sample) / (frame + 1)
            _normal_accumulator[y, x] = (
                _normal_accumulator[y, x] * frame + normal_sample
            ) / (frame + 1)
            _albedo_accumulator[y, x] = (
                _albedo_accumulator[y, x] * frame + albedo_sample
            ) / (frame + 1)
            _depth_accumulator[y, x] = (
                _depth_accumulator[y, x] * frame + depth_sample
            ) / (frame + 1)

        _pixels[y, x] = ti.sqrt(ti.min(_accumulator[y, x], 1.0))
