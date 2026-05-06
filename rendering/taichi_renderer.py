# ============================================
# Author: Seth
# Date: May 2026
# Version: 2.0
# Description: GPU-accelerated renderer using Taichi.
#              Delegates to rendering/taichi/ package for all GPU logic.
# ============================================

import math
import taichi as ti

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
        _frame_count[None] = 0  # reset accumulator

        target_samples = self._samples
        for frame in range(target_samples):
            if self._viewport and self._viewport.should_close:
                break
            
            render_kernel(W, H, fov_tan, aspect, self._max_depth, use_sky, bg_color,
                        cam_pos, cam_fwd, cam_right, cam_up)
            _frame_count[None] = frame + 1

            # copy to image and display every N frames
            if frame % 4 == 0 or frame == target_samples - 1:
                self._image.pixels[:] = _pixels.to_numpy()[:H, :W]
                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()

            print(f"\r  sample {frame+1}/{target_samples}", end='', flush=True)


        self._image.pixels[:] = _pixels.to_numpy()[:H, :W]

        if self._viewport:
            self._viewport.update(self._image)
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    def __repr__(self):
        return f"TaichiRenderer(samples={self._samples}, max_depth={self._max_depth})"
