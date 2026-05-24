import math
import time
import taichi as ti

import core.timing as timing
from rendering.taichi import extract_scene
from rendering.taichi.fields import (
    _pixels, _accumulator, _normal_accumulator, _albedo_accumulator,
    _depth_accumulator, _frame_count, _ray_count,
)
from rendering.taichi.wavefront import (
    wf_generate, wf_traverse_full, wf_traverse,
    wf_shade_full, wf_shade, wf_resolve_shadows, wf_swap_queues, wf_accumulate,
)
from rendering.denoise import edge_aware_denoise, linear_to_display
from rendering.render_stats import RenderStats


class TaichiWavefrontRenderer:
    """GPU path tracer using a wavefront architecture (Taichi Metal backend).

    Separates BVH traversal and shading into distinct kernels, reducing
    register pressure per kernel and improving GPU occupancy.
    """

    def __init__(self, world, image, viewport, samples=16, max_depth=4,
                 direct_light_mode="one", denoise=False,
                 denoise_radius=1, denoise_sigma=0.08, denoise_amount=0.8,
                 sample_clamp=10.0, direct_light_max_depth=1,
                 startup_progress=None, count_rays=True, compact_rays=False,
                 split_direct_light=True):
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
        self._last_timing = {}
        self._startup_progress = startup_progress
        self._count_rays = count_rays
        self._compact_rays = compact_rays
        self._split_direct_light = split_direct_light

    @staticmethod
    def _direct_light_mode_to_id(mode):
        if mode in (0, "one", "random", "sample"):
            return 0
        if mode in (1, "all", "final"):
            return 1
        raise ValueError("direct_light_mode must be 'one' or 'all'")

    def _run_sample(self, W, H, frame, fov_tan, aspect, use_sky, bg_color,
                    cam_pos, cam_fwd, cam_right, cam_up):
        count_rays = int(self._count_rays)
        compact_rays = int(self._compact_rays)
        split_direct_light = int(self._split_direct_light)
        wf_generate(W, H, frame, fov_tan, aspect, count_rays, compact_rays,
                    cam_pos, cam_fwd, cam_right, cam_up)
        for depth_idx in range(self._max_depth):
            if self._compact_rays:
                wf_traverse(W, H, count_rays)
                wf_shade(
                    W, H, depth_idx, use_sky, bg_color,
                    self._direct_light_mode_id,
                    self._direct_light_max_depth,
                    self._max_depth,
                    count_rays,
                    split_direct_light,
                )
                if self._split_direct_light:
                    wf_resolve_shadows()
                wf_swap_queues()
            else:
                wf_traverse_full(W, H, count_rays)
                wf_shade_full(
                    W, H, depth_idx, use_sky, bg_color,
                    self._direct_light_mode_id,
                    self._direct_light_max_depth,
                    self._max_depth,
                    count_rays,
                    split_direct_light,
                )
                if self._split_direct_light:
                    wf_resolve_shadows()
        wf_accumulate(W, H, frame, self._sample_clamp, count_rays)

    @timing.timer("first frame (JIT compile)", tag="taichi")
    def _jit_frame(self, W, H, fov_tan, aspect, use_sky, bg_color,
                   cam_pos, cam_fwd, cam_right, cam_up):
        self._run_sample(W, H, 0, fov_tan, aspect, use_sky, bg_color,
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
        bg_color = ti.Vector(list(world._background_color))
        render_start = time.perf_counter()

        if self._startup_progress:
            self._startup_progress.step(
                "Extracting scene for GPU",
                "Flattening objects, materials, lights, and textures.",
            )
        extract_start = time.perf_counter()
        scene_stats = extract_scene(world)
        extract_seconds = time.perf_counter() - extract_start
        _frame_count[None] = 0
        total_rays_cast = 0

        if self._startup_progress:
            self._startup_progress.step(
                "Compiling first GPU frame",
                "Taichi is preparing wavefront kernels.",
            )
        if self._count_rays:
            _ray_count[None] = 0
        jit_start = time.perf_counter()
        self._jit_frame(W, H, fov_tan, aspect, use_sky, bg_color,
                        cam_pos, cam_fwd, cam_right, cam_up)
        jit_seconds = time.perf_counter() - jit_start
        if self._count_rays:
            total_rays_cast += int(_ray_count[None])
        if self._startup_progress:
            self._startup_progress.close()
            self._startup_progress = None

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
            if self._count_rays:
                _ray_count[None] = 0
            self._run_sample(W, H, frame, fov_tan, aspect, use_sky, bg_color,
                             cam_pos, cam_fwd, cam_right, cam_up)
            if self._count_rays:
                total_rays_cast += int(_ray_count[None])
            _frame_count[None] = frame + 1
            frames_rendered = frame + 1

            if frame % 4 == 0 or frame == target_samples - 1:
                self._image.pixels[:] = _pixels.to_numpy()[:H, :W]
                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()

            print(f"\r  sample {frame+1}/{target_samples}", end='', flush=True)

        loop_elapsed = time.perf_counter() - t_loop_start
        if frames_rendered > 1:
            print()
            avg_ms = loop_elapsed / (frames_rendered - 1) * 1000
            print(timing._fmt("render", f"{frames_rendered - 1} steady-state frames",
                               loop_elapsed, f"avg {avg_ms:.1f} ms/frame"))

        self._copy_display_pixels(W, H, apply_denoise=True)
        elapsed = time.perf_counter() - render_start
        self._last_ray_count = total_rays_cast
        self._last_timing = {
            'extract_seconds': extract_seconds,
            'jit_seconds': jit_seconds,
            'steady_seconds': loop_elapsed,
            'steady_frames': max(0, frames_rendered - 1),
            'total_seconds': elapsed,
        }
        self._last_stats = RenderStats(
            width=W,
            height=H,
            samples_requested=self._samples,
            samples_rendered=frames_rendered,
            max_depth=self._max_depth,
            rays_cast=total_rays_cast,
            elapsed_seconds=elapsed,
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

    @property
    def last_timing(self):
        return self._last_timing

    def __repr__(self):
        return (f"TaichiWavefrontRenderer(samples={self._samples}, "
                f"max_depth={self._max_depth}, "
                f"compact_rays={self._compact_rays}, "
                f"split_direct_light={self._split_direct_light}, "
                f"direct_light_mode={self._direct_light_mode!r})")
