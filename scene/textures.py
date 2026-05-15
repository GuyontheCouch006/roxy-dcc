import numpy as np

from core import Color


class ImageTexture:
    """Simple RGB image texture sampled with repeating UVs and bilinear filtering."""

    def __init__(self, path=None, pixels=None, flip_v=True):
        if path is None and pixels is None:
            raise ValueError("ImageTexture needs either a path or pixel array")
        self.path = str(path) if path is not None else None
        self._pixels = None if pixels is None else self._coerce_pixels(pixels)
        self.flip_v = flip_v

    @classmethod
    def from_array(cls, pixels, flip_v=True):
        return cls(pixels=pixels, flip_v=flip_v)

    @staticmethod
    def _coerce_pixels(pixels):
        arr = np.asarray(pixels, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("texture pixels must be an HxWx3 array")
        arr = arr[:, :, :3]
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def _load_pixels(self):
        if self._pixels is not None:
            return self._pixels
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required to load image textures") from exc

        surface = pygame.image.load(self.path)
        pixels = pygame.surfarray.array3d(surface).astype(np.float32) / 255.0
        self._pixels = np.transpose(pixels[:, :, :3], (1, 0, 2))
        return self._pixels

    def load_pixels_u8(self, max_size=None):
        """Load texture pixels as uint8 HxWx3, optionally downsampling first.

        The CPU renderer samples float textures lazily through _load_pixels().
        The Taichi renderer needs a compact upload format; using uint8 here avoids
        expanding large JPEGs to multi-gigabyte float arrays before upload.
        """
        if self.path is None:
            pixels = self._load_pixels()
            return np.clip(pixels * 255.0 + 0.5, 0, 255).astype(np.uint8)

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required to load image textures") from exc

        surface = pygame.image.load(self.path)
        width, height = surface.get_size()
        if max_size is not None and max(width, height) > max_size:
            scale = float(max_size) / float(max(width, height))
            size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            surface = pygame.transform.smoothscale(surface, size)

        pixels = pygame.surfarray.array3d(surface)
        return np.transpose(pixels[:, :, :3], (1, 0, 2)).copy()

    def sample(self, uv):
        pixels = self._load_pixels()
        h, w = pixels.shape[:2]
        if h == 0 or w == 0:
            return Color(1, 1, 1)

        u = float(uv[0]) % 1.0
        v = float(uv[1]) % 1.0
        if self.flip_v:
            v = 1.0 - v

        x = u * (w - 1)
        y = v * (h - 1)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        tx, ty = x - x0, y - y0

        c00 = pixels[y0, x0]
        c10 = pixels[y0, x1]
        c01 = pixels[y1, x0]
        c11 = pixels[y1, x1]
        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        c = c0 * (1.0 - ty) + c1 * ty
        return Color(float(c[0]), float(c[1]), float(c[2]))

    def sample_many(self, uvs):
        pixels = self._load_pixels()
        h, w = pixels.shape[:2]
        uvs = np.asarray(uvs, dtype=np.float32).reshape((-1, 2))
        if h == 0 or w == 0:
            return np.ones((len(uvs), 3), dtype=np.float32)

        u = np.mod(uvs[:, 0], 1.0)
        v = np.mod(uvs[:, 1], 1.0)
        if self.flip_v:
            v = 1.0 - v

        x = u * float(w - 1)
        y = v * float(h - 1)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        tx = (x - x0).reshape((-1, 1))
        ty = (y - y0).reshape((-1, 1))

        c00 = pixels[y0, x0]
        c10 = pixels[y0, x1]
        c01 = pixels[y1, x0]
        c11 = pixels[y1, x1]
        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        return np.clip(c0 * (1.0 - ty) + c1 * ty, 0.0, 1.0).astype(np.float32)

    def to_dict(self):
        if self.path is None:
            return None
        return {"type": "image", "path": self.path, "flip_v": self.flip_v}


def create_texture_from_dict(data):
    if data is None:
        return None
    if data.get("type") == "image":
        return ImageTexture(data["path"], flip_v=data.get("flip_v", True))
    raise ValueError(f"Unknown texture type: {data.get('type')}")
