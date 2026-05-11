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
from rendering.taichi.fields import _pixels, _frame_count


class TaichiRenderer:
    """GPU ray tracer using Taichi Metal backend."""

    def __init__(self, world, image, viewport, samples=16, max_depth=4):
        self._world     = world
        self._image     = image
        self._viewport  = viewport
        self._samples   = samples
        self._max_depth = max_depth
        self._camera    = world.active_camera

    @timing.timer("first frame (JIT compile)", tag="taichi")
    def _jit_frame(self, W, H, fov_tan, aspect, use_sky, bg_color,
                   cam_pos, cam_fwd, cam_right, cam_up):
        render_kernel(W, H, fov_tan, aspect, self._max_depth, use_sky, bg_color,
                      cam_pos, cam_fwd, cam_right, cam_up)

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

        extract_scene(world)
        _frame_count[None] = 0

        args = (W, H, fov_tan, aspect, use_sky, bg_color,
                cam_pos, cam_fwd, cam_right, cam_up)

        self._jit_frame(*args)
        _frame_count[None] = 1
        self._image.pixels[:] = _pixels.to_numpy()[:H, :W]
        if self._viewport:
            self._viewport.update(self._image)
            self._viewport.poll_events()

        target_samples = self._samples
        t_loop_start = time.perf_counter()
        frames_rendered = 1

        for frame in range(1, target_samples):
            if self._viewport and self._viewport.should_close:
                break
            render_kernel(W, H, fov_tan, aspect, self._max_depth, use_sky, bg_color,
                          cam_pos, cam_fwd, cam_right, cam_up)
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

        self._image.pixels[:] = _pixels.to_numpy()[:H, :W]
        if self._viewport:
            self._viewport.update(self._image)
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    def __repr__(self):
        return f"TaichiRenderer(samples={self._samples}, max_depth={self._max_depth})"
