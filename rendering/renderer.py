# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Renderer drives the render loop — iterates pixels, fires rays
#              through the camera, and writes results to the image buffer.
# ============================================


class Renderer:
    """Orchestrates the per-pixel render loop."""

    def __init__(self, raytracer):
        self.raytracer = raytracer



import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rendering.image import Image
from rendering.gl_viewport import GLViewport

W, H = 800, 400
image = Image(W, H)
viewport = GLViewport(W, H, "Roxy — viewport test")

# Paint a test gradient — red left to right, green top to bottom
for y in range(H):
    for x in range(W):
        r = x / W
        g = y / H
        b = 0.3
        image._pixels[y, x] = (r, g, b)

while not viewport.should_close:
    viewport.poll_events()
    viewport.update(image)

viewport.close()