# ============================================
# Author: Seth
# Date: May 2026
# Version: 2.0
# Description: GPU-accelerated renderer using Taichi.
#              Delegates to rendering/taichi/ package for all GPU logic.
# ============================================

import math
import time
import taichi as ti

import core.timing as timing
from rendering.taichi import render_kernel, extract_scene
from rendering.taichi.fields import (
    _pixels, _accumulator, _normal_accumulator, _albedo_accumulator,
    _depth_accumulator, _frame_count, _ray_count,
)
from rendering.denoise import edge_aware_denoise, linear_to_display
from rendering.render_stats import RenderStats


class TaichiRenderer:
    """GPU ray tracer using Taichi Metal backend."""

    def __init__(self, world, image, viewport, samples=16, max_depth=4,
                 direct_light_mode="one", denoise=False,
                 denoise_radius=1, denoise_sigma=0.08, denoise_amount=0.8,
                 sample_clamp=10.0, direct_light_max_depth=1,
                 startup_progress=None):
        self._world     = world
        self._image     = image
        self._viewport  = viewport
        self._samples   = samples
        self._max_depth = max_depth
        self._direct_light_mode = direct_light_mode
        self._direct_light_mode_id = self._direct_light_mode_to_id(direct_light_mode)
        if direct_light_max_depth is None:
            direct_light_max_depth = max_depth
        self._direct_light_max_depth = max(0, int(direct_light_max_depth))
        self._denoise = denoise
        self._denoise_radius = denoise_radius
        self._denoise_sigma = denoise_sigma
        self._denoise_amount = denoise_amount
        self._sample_clamp = sample_clamp
        self._camera    = world.active_camera
        self._last_ray_count = 0
        self._last_stats = None
        self._startup_progress = startup_progress

    @staticmethod
    def _direct_light_mode_to_id(mode):
        if mode in (0, "one", "random", "sample"):
            return 0
        if mode in (1, "all", "final"):
            return 1
        raise ValueError("direct_light_mode must be 'one' or 'all'")

    @timing.timer("first frame (JIT compile)", tag="taichi")
    def _jit_frame(self, W, H, fov_tan, aspect, use_sky, bg_color,
                   cam_pos, cam_fwd, cam_right, cam_up):
        render_kernel(W, H, fov_tan, aspect, self._max_depth, use_sky,
                      self._direct_light_mode_id, self._direct_light_max_depth,
                      self._sample_clamp, int(timing.LEVEL >= 1), bg_color,
                      cam_pos, cam_fwd, cam_right, cam_up)

    @timing.timer("denoise", tag="render")
    def _denoised_pixels(self, W, H):
        linear_pixels = _accumulator.to_numpy()[:H, :W]
        linear_pixels = edge_aware_denoise(
            linear_pixels,
            radius=self._denoise_radius,
            sigma_color=self._denoise_sigma,
            amount=self._denoise_amount,
            normal=_normal_accumulator.to_numpy()[:H, :W],
            albedo=_albedo_accumulator.to_numpy()[:H, :W],
            depth=_depth_accumulator.to_numpy()[:H, :W],
        )
        return linear_to_display(linear_pixels)

    def _copy_display_pixels(self, W, H, apply_denoise=False):
        if apply_denoise and self._denoise:
            self._image.pixels[:] = self._denoised_pixels(W, H)
        else:
            self._image.pixels[:] = _pixels.to_numpy()[:H, :W]

    @timing.timer("render", tag="render")
    def render(self):
        W, H = self._image.width, self._image.height
        cam  = self._camera

        fov_tan = math.tan(math.radians(cam.fov) / 2)
        aspect  = cam.aspect_ratio

        cam_pos   = ti.Vector(list(cam.position))
        cam_fwd   = ti.Vector(list(cam.forward))
        cam_right = ti.Vector(list(cam.right))
        cam_up    = ti.Vector(list(cam.up))

        world    = self._world
        use_sky  = int(world._use_sky)
        bg_color = list(world._background_color)
        render_start = time.perf_counter()

        if self._startup_progress:
            self._startup_progress.step(
                "Extracting scene for GPU",
                "Flattening objects, materials, lights, and textures.",
            )
        scene_stats = extract_scene(world)
        _frame_count[None] = 0
        total_rays_cast = 0

        args = (W, H, fov_tan, aspect, use_sky, bg_color,
                cam_pos, cam_fwd, cam_right, cam_up)

        if self._startup_progress:
            self._startup_progress.step(
                "Compiling first GPU frame",
                "Taichi is preparing kernels and uploading render buffers.",
            )
        if timing.LEVEL >= 1:
            _ray_count[None] = 0
        self._jit_frame(*args)
        if self._startup_progress:
            self._startup_progress.close()
            self._startup_progress = None
        if timing.LEVEL >= 1:
            total_rays_cast += int(_ray_count[None])
        _frame_count[None] = 1
        self._copy_display_pixels(W, H)
        if self._viewport:
            self._viewport.update(self._image)
            self._viewport.poll_events()

        target_samples = self._samples
        t_loop_start = time.perf_counter()
        frames_rendered = 1

        for frame in range(1, target_samples):
            if self._viewport and self._viewport.should_close:
                break
            count_rays = int(timing.LEVEL >= 1)
            if count_rays:
                _ray_count[None] = 0
            render_kernel(W, H, fov_tan, aspect, self._max_depth, use_sky,
                          self._direct_light_mode_id, self._direct_light_max_depth,
                          self._sample_clamp, count_rays, bg_color,
                          cam_pos, cam_fwd, cam_right, cam_up)
            if count_rays:
                total_rays_cast += int(_ray_count[None])
            _frame_count[None] = frame + 1
            frames_rendered = frame + 1

            if frame % 4 == 0 or frame == target_samples - 1:
                self._image.pixels[:] = _pixels.to_numpy()[:H, :W]
                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()

            print(f"\r  sample {frame+1}/{target_samples}", end='', flush=True)

        if timing.LEVEL >= 1 and frames_rendered > 1:
            print()
            loop_elapsed = time.perf_counter() - t_loop_start
            avg_ms = loop_elapsed / (frames_rendered - 1) * 1000
            print(timing._fmt("render", f"{frames_rendered - 1} steady-state frames",
                               loop_elapsed, f"avg {avg_ms:.1f} ms/frame"))

        self._copy_display_pixels(W, H, apply_denoise=True)
        self._last_ray_count = total_rays_cast
        self._last_stats = RenderStats(
            width=W,
            height=H,
            samples_requested=self._samples,
            samples_rendered=frames_rendered,
            max_depth=self._max_depth,
            rays_cast=total_rays_cast,
            elapsed_seconds=time.perf_counter() - render_start,
            primitive_count=scene_stats.get('primitive_count', 0),
            bvh_nodes=scene_stats.get('bvh_nodes', 0),
            bvh_triangles=scene_stats.get('bvh_triangles', 0),
            bvh_materials=scene_stats.get('bvh_materials', 0),
            bvh_leaf_size=scene_stats.get('bvh_leaf_size', 0),
        )
        print("\n" + self._last_stats.format_report())
        if self._viewport:
            self._viewport.update(self._image)
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    @property
    def last_ray_count(self):
        return self._last_ray_count

    @property
    def last_stats(self):
        return self._last_stats

    def __repr__(self):
        return (f"TaichiRenderer(samples={self._samples}, max_depth={self._max_depth}, "
                f"direct_light_mode={self._direct_light_mode!r})")
