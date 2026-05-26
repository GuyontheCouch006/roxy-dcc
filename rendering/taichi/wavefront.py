import taichi as ti

from rendering.taichi.fields import (
    _pixels, _accumulator,
    _normal_accumulator, _albedo_accumulator, _depth_accumulator,
    _obj_albedo, _obj_mat_type, _obj_roughness, _obj_ior, _obj_emission,
    _mat_type, _mat_albedo, _mat_roughness, _mat_ior, _mat_emission,
    _light_center, _light_radius, _light_albedo, _light_emission, _n_lights,
    _ray_count,
    _wf_ro, _wf_rd, _wf_throughput, _wf_ior, _wf_active, _wf_color,
    _wf_ray_count,
    _wf_curr_indices, _wf_next_indices, _wf_curr_count, _wf_next_count,
    _wf_shadow_ro, _wf_shadow_rd, _wf_shadow_max_t, _wf_shadow_contrib,
    _wf_shadow_pixel, _wf_shadow_count,
    _wf_first_normal, _wf_first_albedo, _wf_first_depth,
    _wf_hit_t, _wf_hit_u, _wf_hit_v,
    _wf_hit_tri, _wf_hit_is_bvh, _wf_hit_bvh_mat, _wf_hit_normal,
    MAX_RAYS,
)
from rendering.taichi.camera import get_ray_direction
from rendering.taichi.intersect import scene_intersect, scene_occluded
from rendering.taichi.trace import direct_light_sample, _sphere_light_sample, _bvh_albedo
from rendering.taichi.scatter import diffuse_scatter, metal_scatter, dielectric_scatter, glossy_scatter
from rendering.taichi.sky import sky_color


@ti.func
def _direct_light_enqueue_or_inline(
    pixel_idx, hit_p, normal, albedo, throughput,
    direct_light_mode, count_rays, split_direct_light, strength,
):
    if split_direct_light and direct_light_mode == 0:
        light_count = _n_lights[None]
        if light_count > 0:
            light_idx = ti.min(ti.cast(ti.random() * light_count, ti.i32), light_count - 1)
            light_dir, t_light, solid_angle_scale = _sphere_light_sample(
                hit_p, _light_center[light_idx], _light_radius[light_idx])
            ndotl = normal.dot(light_dir)

            if ndotl > 0.0:
                max_t = t_light * 0.999
                if max_t > 0.001:
                    slot = ti.atomic_add(_wf_shadow_count[None], 1)
                    if slot < MAX_RAYS:
                        if count_rays:
                            _wf_ray_count[pixel_idx] += 1
                        sample_weight = ti.cast(light_count, ti.f32)
                        _wf_shadow_pixel[slot] = pixel_idx
                        _wf_shadow_ro[slot] = hit_p + normal * 1e-4
                        _wf_shadow_rd[slot] = light_dir
                        _wf_shadow_max_t[slot] = max_t
                        _wf_shadow_contrib[slot] = (
                            throughput *
                            albedo *
                            _light_albedo[light_idx] *
                            _light_emission[light_idx] *
                            ndotl *
                            solid_angle_scale *
                            sample_weight *
                            strength
                        )
    else:
        direct, shadow_rays = direct_light_sample(
            hit_p, normal, albedo, direct_light_mode)
        _wf_color[pixel_idx] += throughput * direct * strength
        if count_rays:
            _wf_ray_count[pixel_idx] += shadow_rays


@ti.kernel
def wf_generate(
    W: int, H: int, frame: int,
    fov_tan: float, aspect: float,
    count_rays: int,
    compact_rays: int,
    cam_pos:   ti.types.vector(3, ti.f32),
    cam_fwd:   ti.types.vector(3, ti.f32),
    cam_right: ti.types.vector(3, ti.f32),
    cam_up:    ti.types.vector(3, ti.f32),
):
    if compact_rays:
        _wf_curr_count[None] = W * H
        _wf_next_count[None] = 0
    for y, x in ti.ndrange(H, W):
        i = y * W + x
        if compact_rays:
            _wf_curr_indices[i] = i
        rd = get_ray_direction(x, y, W, H, frame, fov_tan, aspect,
                               cam_fwd, cam_right, cam_up)
        _wf_ro[i]           = cam_pos
        _wf_rd[i]           = rd
        _wf_throughput[i]   = ti.Vector([1.0, 1.0, 1.0])
        _wf_ior[i]          = 1.0
        _wf_active[i]       = 1
        _wf_color[i]        = ti.Vector([0.0, 0.0, 0.0])
        _wf_first_normal[i] = ti.Vector([0.0, 0.0, 0.0])
        _wf_first_albedo[i] = ti.Vector([0.0, 0.0, 0.0])
        _wf_first_depth[i]  = 0.0
        if count_rays:
            _wf_ray_count[i] = 0


@ti.kernel
def wf_traverse_full(W: int, H: int, count_rays: int):
    for y, x in ti.ndrange(H, W):
        i = y * W + x
        if _wf_active[i]:
            if count_rays:
                _wf_ray_count[i] += 1
            t, normal, idx, is_bvh, bvh_mat, u, v = scene_intersect(
                _wf_ro[i], _wf_rd[i])
            _wf_hit_t[i]       = t
            _wf_hit_normal[i]  = normal
            _wf_hit_tri[i]     = idx
            _wf_hit_is_bvh[i]  = is_bvh
            _wf_hit_bvh_mat[i] = bvh_mat
            _wf_hit_u[i]       = u
            _wf_hit_v[i]       = v


@ti.kernel
def wf_traverse(W: int, H: int, count_rays: int):
    for q in range(_wf_curr_count[None]):
        i = _wf_curr_indices[q]
        if _wf_active[i]:
            if count_rays:
                _wf_ray_count[i] += 1
            t, normal, idx, is_bvh, bvh_mat, u, v = scene_intersect(
                _wf_ro[i], _wf_rd[i])
            _wf_hit_t[i]       = t
            _wf_hit_normal[i]  = normal
            _wf_hit_tri[i]     = idx
            _wf_hit_is_bvh[i]  = is_bvh
            _wf_hit_bvh_mat[i] = bvh_mat
            _wf_hit_u[i]       = u
            _wf_hit_v[i]       = v


@ti.kernel
def wf_shade_full(
    W: int, H: int,
    depth_idx: int,
    use_sky: int,
    bg_color: ti.types.vector(3, ti.f32),
    direct_light_mode: int,
    direct_light_max_depth: int,
    max_depth: int,
    count_rays: int,
    split_direct_light: int,
):
    _wf_shadow_count[None] = 0
    for y, x in ti.ndrange(H, W):
        i = y * W + x
        if _wf_active[i] == 0:
            continue

        t      = _wf_hit_t[i]
        rd     = _wf_rd[i]
        normal = _wf_hit_normal[i]

        front_face = normal.dot(rd) < 0.0
        if not front_face:
            normal = -normal

        if t < 0.0 or t >= 1e9:
            if use_sky:
                _wf_color[i] += _wf_throughput[i] * sky_color(rd)
            else:
                _wf_color[i] += _wf_throughput[i] * bg_color
            _wf_active[i] = 0
            continue

        ro    = _wf_ro[i]
        hit_p = ro + rd * t

        idx     = _wf_hit_tri[i]
        is_bvh  = _wf_hit_is_bvh[i]
        bvh_mat = _wf_hit_bvh_mat[i]
        bary_u  = _wf_hit_u[i]
        bary_v  = _wf_hit_v[i]

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

        if depth_idx == 0:
            _wf_first_normal[i] = normal
            _wf_first_albedo[i] = albedo
            _wf_first_depth[i]  = t

        throughput = _wf_throughput[i]

        if mat_type == 3:
            _wf_color[i] += throughput * albedo * emission
            _wf_active[i] = 0
        else:
            new_rd = ti.Vector([0.0, 0.0, 1.0])

            if mat_type == 0:
                if depth_idx < direct_light_max_depth:
                    _direct_light_enqueue_or_inline(
                        i, hit_p, normal, albedo, throughput,
                        direct_light_mode, count_rays, split_direct_light, 1.0,
                    )
                new_rd = diffuse_scatter(normal)
                throughput *= albedo
            elif mat_type == 1:
                new_rd = metal_scatter(rd, normal, roughness)
                throughput *= albedo
            elif mat_type == 4:
                if depth_idx < direct_light_max_depth:
                    _direct_light_enqueue_or_inline(
                        i, hit_p, normal, albedo, throughput,
                        direct_light_mode, count_rays, split_direct_light, roughness,
                    )
                new_rd = glossy_scatter(rd, normal, roughness)
                throughput *= albedo
            elif mat_type == 2:
                current_ior = _wf_ior[i]
                eta = 1.0
                if front_face:
                    eta = current_ior / ior
                    _wf_ior[i] = ior
                else:
                    eta = current_ior / 1.0
                    _wf_ior[i] = 1.0
                new_rd = dielectric_scatter(rd, normal, eta)
                throughput *= albedo

            if depth_idx > 2:
                rr = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
                if ti.random() > rr:
                    _wf_active[i] = 0
                else:
                    throughput /= rr

            if _wf_active[i] != 0:
                if depth_idx >= max_depth - 1:
                    _wf_active[i] = 0
                else:
                    _wf_throughput[i] = throughput
                    _wf_ro[i] = hit_p + normal * 1e-4
                    _wf_rd[i] = new_rd


@ti.kernel
def wf_shade(
    W: int, H: int,
    depth_idx: int,
    use_sky: int,
    bg_color: ti.types.vector(3, ti.f32),
    direct_light_mode: int,
    direct_light_max_depth: int,
    max_depth: int,
    count_rays: int,
    split_direct_light: int,
):
    _wf_next_count[None] = 0
    _wf_shadow_count[None] = 0
    for q in range(_wf_curr_count[None]):
        i = _wf_curr_indices[q]
        if _wf_active[i] == 0:
            continue

        t      = _wf_hit_t[i]
        rd     = _wf_rd[i]
        normal = _wf_hit_normal[i]

        front_face = normal.dot(rd) < 0.0
        if not front_face:
            normal = -normal

        if t < 0.0 or t >= 1e9:
            if use_sky:
                _wf_color[i] += _wf_throughput[i] * sky_color(rd)
            else:
                _wf_color[i] += _wf_throughput[i] * bg_color
            _wf_active[i] = 0
            continue

        ro    = _wf_ro[i]
        hit_p = ro + rd * t

        idx     = _wf_hit_tri[i]
        is_bvh  = _wf_hit_is_bvh[i]
        bvh_mat = _wf_hit_bvh_mat[i]
        bary_u  = _wf_hit_u[i]
        bary_v  = _wf_hit_v[i]

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

        if depth_idx == 0:
            _wf_first_normal[i] = normal
            _wf_first_albedo[i] = albedo
            _wf_first_depth[i]  = t

        throughput = _wf_throughput[i]

        if mat_type == 3:   # emissive
            _wf_color[i] += throughput * albedo * emission
            _wf_active[i] = 0
        else:
            new_rd = ti.Vector([0.0, 0.0, 1.0])

            if mat_type == 0:   # diffuse
                if depth_idx < direct_light_max_depth:
                    _direct_light_enqueue_or_inline(
                        i, hit_p, normal, albedo, throughput,
                        direct_light_mode, count_rays, split_direct_light, 1.0,
                    )
                new_rd = diffuse_scatter(normal)
                throughput *= albedo
            elif mat_type == 1:  # metal
                new_rd = metal_scatter(rd, normal, roughness)
                throughput *= albedo
            elif mat_type == 4:  # glossy
                if depth_idx < direct_light_max_depth:
                    _direct_light_enqueue_or_inline(
                        i, hit_p, normal, albedo, throughput,
                        direct_light_mode, count_rays, split_direct_light, roughness,
                    )
                new_rd = glossy_scatter(rd, normal, roughness)
                throughput *= albedo
            elif mat_type == 2:  # dielectric
                current_ior = _wf_ior[i]
                eta = 1.0
                if front_face:
                    eta = current_ior / ior
                    _wf_ior[i] = ior
                else:
                    eta = current_ior / 1.0
                    _wf_ior[i] = 1.0
                new_rd = dielectric_scatter(rd, normal, eta)
                throughput *= albedo

            # Russian roulette
            if depth_idx > 2:
                rr = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
                if ti.random() > rr:
                    _wf_active[i] = 0
                else:
                    throughput /= rr

            if _wf_active[i] != 0:
                if depth_idx >= max_depth - 1:
                    _wf_active[i] = 0
                else:
                    _wf_throughput[i] = throughput
                    _wf_ro[i] = hit_p + normal * 1e-4
                    _wf_rd[i] = new_rd
                    slot = ti.atomic_add(_wf_next_count[None], 1)
                    if slot < MAX_RAYS:
                        _wf_next_indices[slot] = i
                    else:
                        _wf_active[i] = 0


@ti.kernel
def wf_resolve_shadows():
    for slot in range(ti.min(_wf_shadow_count[None], MAX_RAYS)):
        pixel_idx = _wf_shadow_pixel[slot]
        if scene_occluded(_wf_shadow_ro[slot], _wf_shadow_rd[slot], _wf_shadow_max_t[slot]) == 0:
            _wf_color[pixel_idx] += _wf_shadow_contrib[slot]


@ti.kernel
def wf_swap_queues():
    next_count = ti.min(_wf_next_count[None], MAX_RAYS)
    for q in range(next_count):
        _wf_curr_indices[q] = _wf_next_indices[q]
    _wf_curr_count[None] = next_count
    _wf_next_count[None] = 0


@ti.kernel
def wf_accumulate(W: int, H: int, frame: int, sample_clamp: float, count_rays: int):
    for y, x in ti.ndrange(H, W):
        i = y * W + x
        sample = _wf_color[i]
        if sample_clamp > 0.0:
            clamp_v = ti.Vector([sample_clamp, sample_clamp, sample_clamp])
            sample = ti.min(ti.max(sample, ti.Vector([0.0, 0.0, 0.0])), clamp_v)

        first_normal = _wf_first_normal[i]
        first_albedo = _wf_first_albedo[i]
        first_depth  = _wf_first_depth[i]

        if frame == 0:
            _accumulator[y, x]        = sample
            _normal_accumulator[y, x] = first_normal
            _albedo_accumulator[y, x] = first_albedo
            _depth_accumulator[y, x]  = first_depth
        else:
            _accumulator[y, x] = (
                _accumulator[y, x] * frame + sample
            ) / (frame + 1)
            _normal_accumulator[y, x] = (
                _normal_accumulator[y, x] * frame + first_normal
            ) / (frame + 1)
            _albedo_accumulator[y, x] = (
                _albedo_accumulator[y, x] * frame + first_albedo
            ) / (frame + 1)
            _depth_accumulator[y, x] = (
                _depth_accumulator[y, x] * frame + first_depth
            ) / (frame + 1)

        _pixels[y, x] = ti.sqrt(ti.min(_accumulator[y, x], 1.0))
        if count_rays:
            ti.atomic_add(_ray_count[None], _wf_ray_count[i])
