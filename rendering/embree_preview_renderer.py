import math
import time

import numpy as np

from rendering.denoise import linear_to_display
from rendering.intersector import EmbreeIntersector
from rendering.ray_tracer import _collect_emissive_sphere_lights
from rendering.render_stats import RenderStats


_SHADOW_EPSILON = 1e-4
_MAX_PREVIEW_TEXTURE_SIZE = 2048


class EmbreePreviewRenderer:
    """Batched Embree direct-light preview renderer.

    This is a first wavefront-style renderer: it traces primary rays in one
    Embree batch and shadow rays in one batch per light. It is intentionally
    simpler than the recursive Python path tracer, but it proves the app can
    render through Embree without initializing the Taichi renderer.
    """

    def __init__(self, world, image, viewport, *, intersector=None,
                 startup_progress=None):
        self._world = world
        self._image = image
        self._viewport = viewport
        self._camera = world.active_camera
        self._intersector = intersector or EmbreeIntersector(world)
        self._scene = self._intersector.triangle_scene
        self._startup_progress = startup_progress
        self._last_stats = None
        self._last_ray_count = 0
        self._texture_cache = {}

    def render(self):
        W, H = self._image.width, self._image.height
        start = time.perf_counter()

        if self._startup_progress:
            self._startup_progress.step(
                "Tracing Embree preview",
                "Batching primary rays and direct-light shadow rays.",
            )

        origins, directions = _primary_rays(self._camera, W, H)
        raw = self._intersector.intersect_raw_arrays(origins, directions)
        colors = self._background(directions)
        rays_cast = len(origins)

        hit_mask = raw["hit"]
        if np.any(hit_mask):
            hit_indices = np.nonzero(hit_mask)[0]
            prim_ids = raw["tri_id"][hit_mask]
            t = raw["t"][hit_mask]
            u = raw["u"][hit_mask]
            v = raw["v"][hit_mask]
            hit_origins = origins[hit_mask]
            hit_dirs = directions[hit_mask]
            points = hit_origins + hit_dirs * t[:, None]
            normals = self._normals(prim_ids, u, v, hit_dirs)
            albedo = self._albedo(prim_ids, u, v)

            shaded = self._emission(prim_ids)
            direct, shadow_rays = self._direct_lighting(points, normals, albedo)
            rays_cast += shadow_rays
            colors[hit_indices] = shaded + direct

        pixels = colors.reshape((H, W, 3))
        self._image.pixels[:] = linear_to_display(pixels)
        self._last_ray_count = rays_cast
        self._last_stats = RenderStats(
            width=W,
            height=H,
            samples_requested=1,
            samples_rendered=1,
            max_depth=1,
            rays_cast=rays_cast,
            elapsed_seconds=time.perf_counter() - start,
            primitive_count=self._scene.skipped_primitives,
            bvh_triangles=self._scene.triangle_count,
            bvh_materials=len(self._scene.materials),
        )

        if self._startup_progress:
            self._startup_progress.close()
            self._startup_progress = None

        print("\n" + self._last_stats.format_report())
        if self._viewport:
            self._viewport.update(self._image)
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    def _background(self, directions):
        if not self._world.use_sky:
            color = np.asarray(list(self._world.background_color), dtype=np.float32)
            return np.repeat(color[None, :], len(directions), axis=0)

        t = 0.5 * (directions[:, 1:2] + 1.0)
        white = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
        blue = np.asarray([[0.5, 0.7, 1.0]], dtype=np.float32)
        return white * (1.0 - t) + blue * t

    def _normals(self, prim_ids, u, v, directions):
        rows = self._scene.normals[prim_ids]
        w = 1.0 - u - v
        normals = rows[:, 0] * w[:, None] + rows[:, 1] * u[:, None] + rows[:, 2] * v[:, None]
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        normals[~valid] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        back_facing = np.einsum("ij,ij->i", directions, normals) > 0.0
        normals[back_facing] *= -1.0
        return normals.astype(np.float32, copy=False)

    def _uvs(self, prim_ids, u, v):
        rows = self._scene.uvs[prim_ids]
        w = 1.0 - u - v
        return rows[:, 0] * w[:, None] + rows[:, 1] * u[:, None] + rows[:, 2] * v[:, None]

    def _albedo(self, prim_ids, u, v):
        material_ids = self._scene.material_idx[prim_ids]
        uvs = self._uvs(prim_ids, u, v)
        has_uv = self._scene.has_uv[prim_ids]
        albedo = np.zeros((len(prim_ids), 3), dtype=np.float32)

        for material_id in np.unique(material_ids):
            mask = material_ids == material_id
            material = self._scene.materials[int(material_id)]
            texture = getattr(material, "_albedo_texture", None)
            if texture is None:
                albedo[mask] = np.asarray(list(material._albedo), dtype=np.float32)
                continue

            local_indices = np.nonzero(mask)[0]
            albedo[local_indices] = np.asarray(list(material._albedo), dtype=np.float32)
            textured = local_indices[has_uv[local_indices]]
            if len(textured):
                albedo[textured] = self._sample_texture(texture, uvs[textured])

        return albedo

    def _sample_texture(self, texture, uvs):
        key = (id(texture), _MAX_PREVIEW_TEXTURE_SIZE)
        pixels = self._texture_cache.get(key)
        if pixels is None:
            if getattr(texture, "path", None) is not None:
                pixels = texture.load_pixels_u8(
                    max_size=_MAX_PREVIEW_TEXTURE_SIZE,
                ).astype(np.float32) / 255.0
            else:
                pixels = texture._load_pixels()
            self._texture_cache[key] = pixels
        return _sample_pixels(pixels, uvs, flip_v=texture.flip_v)

    def _emission(self, prim_ids):
        material_ids = self._scene.material_idx[prim_ids]
        emission = np.zeros((len(prim_ids), 3), dtype=np.float32)
        for material_id in np.unique(material_ids):
            mask = material_ids == material_id
            material = self._scene.materials[int(material_id)]
            if hasattr(material, "emitted"):
                emission[mask] = np.asarray(list(material.emitted()), dtype=np.float32)
        return emission

    def _direct_lighting(self, points, normals, albedo):
        lights = _collect_emissive_sphere_lights(self._world)
        direct = np.zeros_like(albedo)
        rays_cast = 0

        for light in lights:
            light_pos = np.asarray(list(light.center), dtype=np.float32)
            to_light = light_pos[None, :] - points
            dist2 = np.einsum("ij,ij->i", to_light, to_light)
            valid = dist2 > 1e-12
            if not np.any(valid):
                continue

            dist = np.sqrt(dist2, dtype=np.float32)
            light_dirs = np.zeros_like(to_light)
            light_dirs[valid] = to_light[valid] / dist[valid, None]
            ndotl = np.maximum(0.0, np.einsum("ij,ij->i", normals, light_dirs))
            candidates = valid & (ndotl > 0.0)
            if not np.any(candidates):
                continue

            candidate_indices = np.nonzero(candidates)[0]
            shadow_origins = points[candidates] + normals[candidates] * _SHADOW_EPSILON
            max_t = np.maximum(dist[candidates] - _SHADOW_EPSILON, 0.0).astype(np.float32)
            blocked = self._intersector.occluded_raw_arrays(
                shadow_origins,
                light_dirs[candidates],
                max_t,
            )
            rays_cast += len(candidate_indices)

            visible_indices = candidate_indices[~blocked]
            if len(visible_indices) == 0:
                continue

            light_color = np.asarray(list(light.color), dtype=np.float32)
            attenuation = light.intensity * ndotl[visible_indices] / np.maximum(
                dist2[visible_indices],
                1e-6,
            )
            direct[visible_indices] += (
                albedo[visible_indices]
                * light_color[None, :]
                * attenuation[:, None]
                * (1.0 / math.pi)
            )

        return direct, rays_cast

    @property
    def last_ray_count(self):
        return self._last_ray_count

    @property
    def last_stats(self):
        return self._last_stats


def _primary_rays(camera, width, height):
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    u = xs / width * 2.0 - 1.0
    v = 1.0 - ys / height * 2.0
    half_fov = np.tan(np.radians(camera.fov) / 2.0)
    ndc_x = u * camera.aspect_ratio * half_fov
    ndc_y = v * half_fov

    forward = np.asarray(list(camera.forward), dtype=np.float32)
    right = np.asarray(list(camera.right), dtype=np.float32)
    up = np.asarray(list(camera.up), dtype=np.float32)
    directions = (
        forward[None, None, :]
        + right[None, None, :] * ndc_x[:, :, None]
        + up[None, None, :] * ndc_y[:, :, None]
    )
    directions = directions.reshape((-1, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    origins = np.repeat(
        np.asarray([list(camera.position)], dtype=np.float32),
        len(directions),
        axis=0,
    )
    return origins, directions.astype(np.float32, copy=False)


def _sample_pixels(pixels, uvs, flip_v=True):
    pixels = np.asarray(pixels, dtype=np.float32)
    uvs = np.asarray(uvs, dtype=np.float32).reshape((-1, 2))
    h, w = pixels.shape[:2]
    if h == 0 or w == 0:
        return np.ones((len(uvs), 3), dtype=np.float32)

    u = np.mod(uvs[:, 0], 1.0)
    v = np.mod(uvs[:, 1], 1.0)
    if flip_v:
        v = 1.0 - v

    x = u * float(w - 1)
    y = v * float(h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (x - x0).reshape((-1, 1))
    ty = (y - y0).reshape((-1, 1))

    c00 = pixels[y0, x0, :3]
    c10 = pixels[y0, x1, :3]
    c01 = pixels[y1, x0, :3]
    c11 = pixels[y1, x1, :3]
    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return np.clip(c0 * (1.0 - ty) + c1 * ty, 0.0, 1.0).astype(np.float32)
