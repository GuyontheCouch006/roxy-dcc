import time

import numpy as np

from rendering.denoise import linear_to_display
from rendering.embree_preview_renderer import (
    EmbreePreviewRenderer,
    _SHADOW_EPSILON,
    _primary_rays,
    _sample_pixels,
)
from rendering.intersector import EmbreeIntersector
from rendering.render_stats import RenderStats


_MAT_DIFFUSE = 0
_MAT_METAL = 1
_MAT_DIELECTRIC = 2
_MAT_EMISSIVE = 3
_MAT_GLOSSY = 4


class EmbreeWavefrontRenderer(EmbreePreviewRenderer):
    """Queue-based Embree path tracer.

    Rays are stored as NumPy arrays and processed depth-by-depth. Each wave does
    one batched Embree query, shades the hit batch, accumulates emission/direct
    light, then compacts surviving scatter rays into the next queue.
    """

    def __init__(self, world, image, viewport, *, intersector=None,
                 samples=4, max_depth=3, direct_light_max_depth=1,
                 sample_clamp=10.0, seed=7, startup_progress=None):
        super().__init__(
            world,
            image,
            viewport,
            intersector=intersector or EmbreeIntersector(world),
            startup_progress=startup_progress,
        )
        self._samples = int(samples)
        self._max_depth = int(max_depth)
        self._direct_light_max_depth = int(direct_light_max_depth)
        self._sample_clamp = sample_clamp
        self._seed = int(seed)

    def render(self):
        W, H = self._image.width, self._image.height
        render_start = time.perf_counter()
        rng = np.random.default_rng(self._seed)

        if self._startup_progress:
            self._startup_progress.step(
                "Tracing Embree wavefront",
                "Processing ray queues through batched Embree intersections.",
            )

        accum = np.zeros((H * W, 3), dtype=np.float32)
        rays_cast = 0
        frames_rendered = 0

        for sample in range(self._samples):
            sample_color, sample_rays = self._trace_sample(W, H, rng)
            if self._sample_clamp is not None and self._sample_clamp > 0.0:
                sample_color = np.minimum(sample_color, float(self._sample_clamp))
            accum += sample_color
            rays_cast += sample_rays
            frames_rendered = sample + 1

            display = linear_to_display((accum / frames_rendered).reshape((H, W, 3)))
            self._image.pixels[:] = display
            if self._viewport:
                self._viewport.update(self._image)
                self._viewport.poll_events()
                if self._viewport.should_close:
                    break

            print(f"\r  sample {frames_rendered}/{self._samples}", end='', flush=True)

        elapsed = time.perf_counter() - render_start
        self._last_ray_count = rays_cast
        self._last_stats = RenderStats(
            width=W,
            height=H,
            samples_requested=self._samples,
            samples_rendered=frames_rendered,
            max_depth=self._max_depth,
            rays_cast=rays_cast,
            elapsed_seconds=elapsed,
            primitive_count=self._scene.skipped_primitives,
            bvh_triangles=self._scene.triangle_count,
            bvh_materials=len(self._scene.materials),
        )

        if self._startup_progress:
            self._startup_progress.close()
            self._startup_progress = None

        print("\n" + self._last_stats.format_report())
        if self._viewport:
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    def _trace_sample(self, W, H, rng):
        origins, directions = _primary_rays(self._camera, W, H)
        pixel_idx = np.arange(W * H, dtype=np.int32)
        throughput = np.ones((W * H, 3), dtype=np.float32)
        radiance = np.zeros((W * H, 3), dtype=np.float32)
        rays_cast = 0

        for depth in range(self._max_depth):
            if len(origins) == 0:
                break

            raw = self._intersector.intersect_raw_arrays(origins, directions)
            rays_cast += len(origins)

            hit_mask = raw["hit"]
            miss_mask = ~hit_mask
            if np.any(miss_mask):
                miss_pixels = pixel_idx[miss_mask]
                radiance[miss_pixels] += (
                    throughput[miss_mask] * self._background(directions[miss_mask])
                )

            if not np.any(hit_mask):
                break

            hit_pixels = pixel_idx[hit_mask]
            hit_origins = origins[hit_mask]
            hit_dirs = directions[hit_mask]
            hit_throughput = throughput[hit_mask]
            prim_ids = raw["tri_id"][hit_mask]
            t = raw["t"][hit_mask]
            u = raw["u"][hit_mask]
            v = raw["v"][hit_mask]

            points = hit_origins + hit_dirs * t[:, None]
            normals = self._normals(prim_ids, u, v, hit_dirs)
            albedo = self._albedo(prim_ids, u, v)
            material_ids = self._scene.material_idx[prim_ids]
            mat_types = self._material_types(material_ids)
            roughness = self._roughness(material_ids)

            emission = self._emission(prim_ids)
            radiance[hit_pixels] += hit_throughput * emission

            if depth < self._direct_light_max_depth:
                direct_mask = (mat_types == _MAT_DIFFUSE) | (mat_types == _MAT_GLOSSY)
                if np.any(direct_mask):
                    direct, shadow_rays = self._direct_lighting(
                        points[direct_mask],
                        normals[direct_mask],
                        albedo[direct_mask],
                    )
                    rays_cast += shadow_rays
                    glossy_direct = mat_types[direct_mask] == _MAT_GLOSSY
                    if np.any(glossy_direct):
                        direct[glossy_direct] *= roughness[direct_mask][glossy_direct, None]
                    radiance[hit_pixels[direct_mask]] += (
                        hit_throughput[direct_mask] * direct
                    )

            scatter_mask = mat_types != _MAT_EMISSIVE
            scatter_mask &= mat_types != _MAT_DIELECTRIC
            if not np.any(scatter_mask) or depth == self._max_depth - 1:
                break

            next_dirs = np.zeros_like(points)
            next_throughput = hit_throughput * albedo

            diffuse_mask = scatter_mask & (mat_types == _MAT_DIFFUSE)
            if np.any(diffuse_mask):
                next_dirs[diffuse_mask] = _random_cosine_hemisphere(
                    normals[diffuse_mask],
                    rng,
                )

            metal_mask = scatter_mask & (mat_types == _MAT_METAL)
            if np.any(metal_mask):
                next_dirs[metal_mask] = _metal_scatter(
                    hit_dirs[metal_mask],
                    normals[metal_mask],
                    roughness[metal_mask],
                    rng,
                )

            glossy_mask = scatter_mask & (mat_types == _MAT_GLOSSY)
            if np.any(glossy_mask):
                choose_diffuse = rng.random(np.count_nonzero(glossy_mask)) < roughness[glossy_mask]
                glossy_indices = np.nonzero(glossy_mask)[0]
                diffuse_indices = glossy_indices[choose_diffuse]
                metal_indices = glossy_indices[~choose_diffuse]
                if len(diffuse_indices):
                    next_dirs[diffuse_indices] = _random_cosine_hemisphere(
                        normals[diffuse_indices],
                        rng,
                    )
                if len(metal_indices):
                    next_dirs[metal_indices] = _metal_scatter(
                        hit_dirs[metal_indices],
                        normals[metal_indices],
                        roughness[metal_indices],
                        rng,
                    )

            valid_scatter = scatter_mask & (
                np.linalg.norm(next_dirs, axis=1) > 1e-12
            )
            valid_scatter &= (
                np.einsum("ij,ij->i", next_dirs, normals) > 0.0
            )
            if not np.any(valid_scatter):
                break

            origin_offset = normals[valid_scatter] * _SHADOW_EPSILON
            origins = points[valid_scatter] + origin_offset
            directions = _normalize_rows(next_dirs[valid_scatter])
            throughput = next_throughput[valid_scatter]
            pixel_idx = hit_pixels[valid_scatter]

        return radiance, rays_cast

    def _material_types(self, material_ids):
        mat_types = np.zeros(len(material_ids), dtype=np.int32)
        for material_id in np.unique(material_ids):
            mask = material_ids == material_id
            mat_types[mask] = self._scene.materials[int(material_id)].taichi_type_id()
        return mat_types

    def _roughness(self, material_ids):
        roughness = np.zeros(len(material_ids), dtype=np.float32)
        for material_id in np.unique(material_ids):
            material = self._scene.materials[int(material_id)]
            params = material.taichi_params()
            value = float(params[0]) if params else 0.0
            roughness[material_ids == material_id] = value
        return roughness


def _random_cosine_hemisphere(normals, rng):
    count = len(normals)
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    phi = 2.0 * np.pi * r1
    radius = np.sqrt(r2)
    local_x = np.cos(phi) * radius
    local_y = np.sin(phi) * radius
    local_z = np.sqrt(np.maximum(0.0, 1.0 - r2))

    helper = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    use_y = np.abs(normals[:, 0]) > 0.9
    helper[use_y] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    bitangent = _normalize_rows(np.cross(normals, helper))
    tangent = _normalize_rows(np.cross(bitangent, normals))
    directions = (
        tangent * local_x[:, None]
        + bitangent * local_y[:, None]
        + normals * local_z[:, None]
    )
    return _normalize_rows(directions)


def _metal_scatter(directions, normals, roughness, rng):
    reflected = directions - 2.0 * np.einsum(
        "ij,ij->i",
        directions,
        normals,
    )[:, None] * normals
    scattered = reflected + _random_unit_vectors(len(directions), rng) * roughness[:, None]
    return _normalize_rows(scattered)


def _random_unit_vectors(count, rng):
    vectors = rng.normal(size=(count, 3)).astype(np.float32)
    return _normalize_rows(vectors)


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=1)
    valid = lengths > 1e-12
    out = np.zeros_like(values)
    out[valid] = values[valid] / lengths[valid, None]
    return out
