# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Image buffer storing per-pixel color data as a numpy float32 array.
#              Shape is (height, width, 3) to match moderngl texture format directly.
# ============================================

import numpy as np


class Image:
    """Fixed-size pixel buffer addressed by (x, y) screen coordinates.
    
    Pixels stored as float32 RGB in range 0.0-1.0.
    Shape is (height, width, 3) — matches moderngl texture upload directly.
    """

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._pixels = np.zeros((height, width, 3), dtype=np.float32)

    @property
    def width(self): return self._width

    @property
    def height(self): return self._height

    @property
    def aspect_ratio(self): return self._width / self._height

    @property
    def pixels(self): return self._pixels

    def write_pixel(self, x, y, color):
        """Write color to pixel (x, y). Silently ignores out-of-bounds."""
        if 0 <= x < self._width and 0 <= y < self._height:
            r = min(color.r, 1.0) ** 0.5
            g = min(color.g, 1.0) ** 0.5
            b = min(color.b, 1.0) ** 0.5
            self._pixels[y, x] = (r, g, b)
            
    def read_pixel(self, x, y):
        return self._pixels[y, x]

    def flush_scanline(self, y):
        """Hook for progressive display — notifies renderer to upload updated texture."""
        if self._on_scanline:
            self._on_scanline(y)

    def clear(self):
        """Reset all pixels to black."""
        self._pixels[:] = 0

    def __repr__(self):
        return f"Image({self._width}x{self._height})"