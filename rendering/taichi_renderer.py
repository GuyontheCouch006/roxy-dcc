# ============================================
# Author: Seth
# Date: May 2026
# Version: 2.0
# Description: GPU-accelerated renderer using Taichi.
#              Delegates to rendering/taichi/ package for all GPU logic.
# ============================================

import math

from rendering.taichi import render_kernel, extract_scene
from rendering.taichi.fields import _pixels


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

        cam_pos   = list(cam.position)
        cam_fwd   = list(cam.forward)
        cam_right = list(cam.right)
        cam_up    = list(cam.up)

        world    = self._world
        use_sky  = int(world._use_sky)
        bg_color = list(world._background_color)

        extract_scene(world)

        render_kernel(
            W, H, fov_tan, aspect,
            self._samples, self._max_depth,
            use_sky, bg_color,
            cam_pos, cam_fwd, cam_right, cam_up,
        )

        self._image.pixels[:] = _pixels.to_numpy()[:H, :W]

        if self._viewport:
            self._viewport.update(self._image)
            while not self._viewport.should_close:
                self._viewport.poll_events()
                self._viewport.update(self._image)

    def __repr__(self):
        return f"TaichiRenderer(samples={self._samples}, max_depth={self._max_depth})"
