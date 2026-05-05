# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Entry point and RayTracer orchestrator. Wires together the world,
#              camera, and image buffer to produce a rendered output.
# ============================================
import time
from multiprocessing import Pool
import numpy as np
from core import Color, Ray

# ── Module-level worker globals ───────────────────────────────────────────────
_worker_world = None
_worker_camera = None

def _init_worker(world, camera):
    """Called once per worker process at pool startup."""
    global _worker_world, _worker_camera
    _worker_world = world
    _worker_camera = camera

def _trace(ray, world, max_depth, depth=0):
    if depth >= max_depth:
        return Color(0, 0, 0)
    hit = world.intersect(ray)
    if hit is None:
        return world.sky_color(ray)

    mat = hit.material
    emission = mat.emitted() if hasattr(mat, 'emitted') else Color(0, 0, 0)

    result = mat.scatter(ray, hit)
    if result is None:
        return emission

    scattered, attenuation = result

    # Russian roulette — terminate dim rays early
    if depth > 2:
        survival = max(attenuation[0], attenuation[1], attenuation[2])
        if survival < 0.1:
            return emission
        attenuation = attenuation / survival

    return emission + attenuation * _trace(scattered, world, max_depth, depth + 1)
def _trace_band_worker(args):
    """Traces a band of rows. Uses worker-local world and camera globals."""
    y_start, y_end, width, height, samples, max_depth = args
    band = np.zeros((y_end - y_start, width, 3), dtype=np.float32)

    t0 = time.time()

    for y in range(y_start, y_end):
        for x in range(width):
            color = Color(0, 0, 0)
            for _ in range(samples):
                ray = _worker_camera.shoot(x, y, width, height)
                color = color + _trace(ray, _worker_world, max_depth)
            color = color / samples
            r = min(color[0], 1.0) ** 0.5
            g = min(color[1], 1.0) ** 0.5
            b = min(color[2], 1.0) ** 0.5
            band[y - y_start, x] = (r, g, b)

    elapsed = time.time() - t0
    return (y_start, band, elapsed)


class RayTracer:
    def __init__(self, world, image, viewport, samples=4, max_depth=8, threaded=True):
        self._world = world
        self._image = image
        self._viewport = viewport
        self._camera = world.active_camera
        self._samples = samples
        self._max_depth = max_depth
        self._threaded = threaded

    def render(self):
        if self._threaded:
            self._render_threaded()
        else:
            self._render_single()

    def _render_single(self):
        W, H = self._image.width, self._image.height

        for y in range(H):
            for x in range(W):
                color = Color(0, 0, 0)
                for _ in range(self._samples):
                    ray = self._camera.shoot(x, y, W, H)
                    color = color + _trace(ray, self._world, self._max_depth)
                color = color / self._samples
                r = min(color.r, 1.0) ** 0.5
                g = min(color.g, 1.0) ** 0.5
                b = min(color.b, 1.0) ** 0.5
                self._image._pixels[y, x] = (r, g, b)

            print(f"\r  row {y}/{H}", end='', flush=True)

            if self._viewport:
                self._viewport.update_scanline(self._image, y)
                self._viewport.poll_events()
                if self._viewport.should_close:
                    return

        print(f"\n  done.")

    def _render_threaded(self):
        W, H = self._image.width, self._image.height
        band_size = 6

        # Warm up the viewport before pool starts
        if self._viewport:
            for _ in range(10):
                self._viewport.poll_events()
                self._viewport.update(self._image)
                import pygame
                pygame.time.wait(50)

        tasks = [
            (y, min(y + band_size, H), W, H, self._samples, self._max_depth)
            for y in range(0, H, band_size)
        ]
        total = len(tasks)
        completed = 0

        with Pool(
            processes=10,
            initializer=_init_worker,
            initargs=(self._world, self._camera)
        ) as pool:
            for y_start, band, elapsed in pool.imap_unordered(_trace_band_worker, tasks):
                self._image.pixels[y_start:y_start + len(band)] = band
                completed += 1
                pct = completed / total * 100
                print(f"band {y_start:4d}  {elapsed:.2f}s  {completed}/{total}  ({pct:.0f}%)  \n",
                      end='', flush=True)
                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()
                    if self._viewport.should_close:
                        pool.terminate()
                        return

        print(f"\n  done.")

    def __repr__(self):
        return f"RayTracer(samples={self._samples}, max_depth={self._max_depth})"