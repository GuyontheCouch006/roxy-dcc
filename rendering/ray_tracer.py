import time
from multiprocessing import Pool
import numpy as np
from core import Color

_worker_world  = None
_worker_camera = None


def _init_worker(world, camera):
    global _worker_world, _worker_camera
    _worker_world  = world
    _worker_camera = camera


def _trace(ray, world, max_depth, depth=0):
    if depth >= max_depth:
        return Color(0, 0, 0)
    hit = world.intersect(ray)
    if hit is None:
        return world.sky_color(ray)

    mat      = hit.material
    emission = mat.emitted() if hasattr(mat, 'emitted') else Color(0, 0, 0)
    result   = mat.scatter(ray, hit)
    if result is None:
        return emission

    scattered, attenuation = result

    if depth > 2:
        survival = max(attenuation[0], attenuation[1], attenuation[2])
        if survival < 0.1:
            return emission
        attenuation = attenuation / survival

    return emission + attenuation * _trace(scattered, world, max_depth, depth + 1)


def _trace_band_worker(args):
    """One sample per pixel, returns raw linear colors (no gamma)."""
    y_start, y_end, width, height, max_depth = args
    band = np.zeros((y_end - y_start, width, 3), dtype=np.float32)
    for y in range(y_start, y_end):
        for x in range(width):
            ray = _worker_camera.shoot(x, y, width, height)
            c   = _trace(ray, _worker_world, max_depth)
            band[y - y_start, x] = (c[0], c[1], c[2])
    return y_start, band


class RayTracer:
    def __init__(self, world, image, viewport, samples=64, max_depth=8, threaded=True):
        self._world     = world
        self._image     = image
        self._viewport  = viewport
        self._camera    = world.active_camera
        self._samples   = samples
        self._max_depth = max_depth
        self._threaded  = threaded

    def render(self):
        if self._threaded:
            self._render_threaded()
        else:
            self._render_single()

    def _render_single(self):
        W, H  = self._image.width, self._image.height
        accum = np.zeros((H, W, 3), dtype=np.float32)

        for frame in range(self._samples):
            for y in range(H):
                for x in range(W):
                    ray = self._camera.shoot(x, y, W, H)
                    c   = _trace(ray, self._world, self._max_depth)
                    accum[y, x] = (accum[y, x] * frame + [c[0], c[1], c[2]]) / (frame + 1)

            self._image.pixels[:] = np.sqrt(np.minimum(accum, 1.0))
            print(f"\r  sample {frame + 1}/{self._samples}", end='', flush=True)

            if self._viewport:
                self._viewport.update(self._image)
                self._viewport.poll_events()
                if self._viewport.should_close:
                    return

        print("\n  done.")

    def _render_threaded(self):
        W, H      = self._image.width, self._image.height
        band_size = 6
        accum     = np.zeros((H, W, 3), dtype=np.float32)

        tasks = [
            (y, min(y + band_size, H), W, H, self._max_depth)
            for y in range(0, H, band_size)
        ]

        with Pool(
            processes=10,
            initializer=_init_worker,
            initargs=(self._world, self._camera),
        ) as pool:
            for frame in range(self._samples):
                for y_start, band in pool.imap_unordered(_trace_band_worker, tasks):
                    y_end = y_start + len(band)
                    accum[y_start:y_end] = (accum[y_start:y_end] * frame + band) / (frame + 1)
                    self._image.pixels[y_start:y_end] = np.sqrt(
                        np.minimum(accum[y_start:y_end], 1.0)
                    )

                print(f"\r  sample {frame + 1}/{self._samples}", end='', flush=True)

                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()
                    if self._viewport.should_close:
                        pool.terminate()
                        return

        print("\n  done.")

    def __repr__(self):
        return f"RayTracer(samples={self._samples}, max_depth={self._max_depth})"
