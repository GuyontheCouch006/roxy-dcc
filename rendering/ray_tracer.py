import time
import random
from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
from core import Color, Point3, Ray
from rendering.denoise import edge_aware_denoise, linear_to_display
from rendering.render_stats import RenderStats
from rendering.sampling import clamp_color_sample, pixel_sample_offset

_worker_world  = None
_worker_camera = None
_worker_lights = None
_worker_direct_light_mode = None
_worker_sample_clamp = None


@dataclass(frozen=True)
class _SphereLight:
    center: Point3
    radius: float
    color: Color
    intensity: float


def _init_worker(world, camera, lights, direct_light_mode, sample_clamp):
    global _worker_world, _worker_camera, _worker_lights
    global _worker_direct_light_mode, _worker_sample_clamp
    _worker_world  = world
    _worker_camera = camera
    _worker_lights = lights
    _worker_direct_light_mode = direct_light_mode
    _worker_sample_clamp = sample_clamp


def _iter_objects(objects):
    for obj in objects:
        yield obj
        yield from _iter_objects(obj.children)


def _collect_emissive_sphere_lights(world):
    from scene.materials import Emissive
    from scene.primitives import Sphere

    lights = []
    for obj in _iter_objects(world.objects):
        if not obj.renderable or not isinstance(obj.shape, Sphere):
            continue
        mat = obj.material
        if not isinstance(mat, Emissive):
            continue
        exported = obj.taichi_export()
        if isinstance(exported, dict):
            lights.append(_SphereLight(
                center=Point3(*exported['center']),
                radius=exported['radius'],
                color=Color(*exported['albedo']),
                intensity=exported['emission'],
            ))
    return lights


def _direct_light_sample_one(hit, world, light, albedo, sample_weight):
    to_light = light.center - hit.point
    dist2 = max(to_light.length_sq(), 1e-6)
    dist = dist2 ** 0.5
    light_dir = to_light / dist
    ndotl = hit.normal.dot(light_dir)
    if ndotl <= 0:
        return Color(0, 0, 0), 0

    max_t = dist - light.radius * 1.001
    if max_t <= 0.001:
        return Color(0, 0, 0), 0

    shadow_ray = Ray(hit.point + hit.normal * 1e-4, light_dir)
    if world.occluded(shadow_ray, max_t):
        return Color(0, 0, 0), 1

    solid_angle_scale = light.radius * light.radius / max(dist2, light.radius * light.radius)
    contribution = (
        albedo *
        light.color *
        light.intensity *
        ndotl *
        solid_angle_scale *
        sample_weight
    )
    return contribution, 1


def _direct_light_sample(hit, world, lights, albedo, direct_light_mode):
    if not lights:
        return Color(0, 0, 0), 0

    if direct_light_mode == "all":
        direct = Color(0, 0, 0)
        rays_cast = 0
        for light in lights:
            contribution, shadow_rays = _direct_light_sample_one(hit, world, light, albedo, 1.0)
            direct += contribution
            rays_cast += shadow_rays
        return direct, rays_cast

    return _direct_light_sample_one(hit, world, random.choice(lights), albedo, len(lights))


def _trace(ray, world, max_depth, lights=None, direct_light_mode="one", depth=0):
    if depth >= max_depth:
        return Color(0, 0, 0), 0

    rays_cast = 1
    hit = world.intersect(ray)
    if hit is None:
        return world.sky_color(ray), rays_cast

    mat      = hit.material
    emission = mat.emitted() if hasattr(mat, 'emitted') else Color(0, 0, 0)
    result   = mat.scatter(ray, hit)
    if result is None:
        return emission, rays_cast

    scattered, attenuation = result
    direct = Color(0, 0, 0)
    if mat.taichi_type_id() == 0:
        direct, shadow_rays = _direct_light_sample(hit, world, lights or [], attenuation, direct_light_mode)
        rays_cast += shadow_rays
    elif mat.taichi_type_id() == 4:
        roughness = mat.taichi_params()[0]
        direct, shadow_rays = _direct_light_sample(hit, world, lights or [], attenuation, direct_light_mode)
        direct *= roughness
        rays_cast += shadow_rays

    if depth > 2:
        survival = max(attenuation[0], attenuation[1], attenuation[2])
        if survival < 0.1:
            return emission, rays_cast
        attenuation = attenuation / survival

    bounced, bounce_rays = _trace(scattered, world, max_depth, lights, direct_light_mode, depth + 1)
    return emission + direct + attenuation * bounced, rays_cast + bounce_rays


def _trace_band_worker(args):
    """One sample per pixel, returns raw linear colors (no gamma)."""
    y_start, y_end, width, height, max_depth, frame = args
    band = np.zeros((y_end - y_start, width, 3), dtype=np.float32)
    rays_cast = 0
    for y in range(y_start, y_end):
        for x in range(width):
            ray = _worker_camera.shoot(
                x, y, width, height,
                jitter=pixel_sample_offset(x, y, frame),
            )
            c, ray_count = _trace(
                ray, _worker_world, max_depth, _worker_lights, _worker_direct_light_mode)
            c = clamp_color_sample(c, _worker_sample_clamp)
            rays_cast += ray_count
            band[y - y_start, x] = (c[0], c[1], c[2])
    return y_start, band, rays_cast


class RayTracer:
    def __init__(self, world, image, viewport, samples=64, max_depth=8, threaded=True,
                 direct_light_mode="one", denoise=False,
                 denoise_radius=1, denoise_sigma=0.08, denoise_amount=0.8,
                 sample_clamp=10.0):
        self._world     = world
        self._image     = image
        self._viewport  = viewport
        self._camera    = world.active_camera
        self._samples   = samples
        self._max_depth = max_depth
        self._threaded  = threaded
        if direct_light_mode not in ("one", "random", "sample", "all", "final"):
            raise ValueError("direct_light_mode must be 'one' or 'all'")
        self._direct_light_mode = "all" if direct_light_mode in ("all", "final") else "one"
        self._denoise = denoise
        self._denoise_radius = denoise_radius
        self._denoise_sigma = denoise_sigma
        self._denoise_amount = denoise_amount
        self._sample_clamp = sample_clamp
        self._last_ray_count = 0
        self._last_stats = None
        self._lights = _collect_emissive_sphere_lights(world)

    def render(self):
        if self._threaded:
            self._render_threaded()
        else:
            self._render_single()

    def _render_single(self):
        W, H  = self._image.width, self._image.height
        accum = np.zeros((H, W, 3), dtype=np.float32)
        rays_cast = 0
        frames_rendered = 0
        render_start = time.perf_counter()

        for frame in range(self._samples):
            frames_rendered = frame + 1
            for y in range(H):
                for x in range(W):
                    ray = self._camera.shoot(
                        x, y, W, H,
                        jitter=pixel_sample_offset(x, y, frame),
                    )
                    c, ray_count = _trace(
                        ray, self._world, self._max_depth, self._lights,
                        self._direct_light_mode)
                    c = clamp_color_sample(c, self._sample_clamp)
                    rays_cast += ray_count
                    accum[y, x] = (accum[y, x] * frame + [c[0], c[1], c[2]]) / (frame + 1)

            self._image.pixels[:] = np.sqrt(np.minimum(accum, 1.0))
            print(f"\r  sample {frame + 1}/{self._samples}", end='', flush=True)

            if self._viewport:
                self._viewport.update(self._image)
                self._viewport.poll_events()
                if self._viewport.should_close:
                    self._apply_final_pixels(accum)
                    self._finish_stats(W, H, frames_rendered, rays_cast, render_start)
                    return

        self._apply_final_pixels(accum)
        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)

    def _render_threaded(self):
        W, H      = self._image.width, self._image.height
        band_size = 6
        accum     = np.zeros((H, W, 3), dtype=np.float32)
        rays_cast = 0
        frames_rendered = 0
        render_start = time.perf_counter()

        band_ranges = [
            (y, min(y + band_size, H), W, H, self._max_depth)
            for y in range(0, H, band_size)
        ]

        with Pool(
            processes=10,
            initializer=_init_worker,
            initargs=(
                self._world, self._camera, self._lights,
                self._direct_light_mode, self._sample_clamp,
            ),
        ) as pool:
            for frame in range(self._samples):
                frames_rendered = frame + 1
                tasks = [(*band, frame) for band in band_ranges]
                for y_start, band, band_rays in pool.imap_unordered(_trace_band_worker, tasks):
                    rays_cast += band_rays
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
                        self._apply_final_pixels(accum)
                        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)
                        return

        self._apply_final_pixels(accum)
        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)

    def _apply_final_pixels(self, accum):
        if self._denoise:
            accum = edge_aware_denoise(
                accum,
                radius=self._denoise_radius,
                sigma_color=self._denoise_sigma,
                amount=self._denoise_amount,
            )
        self._image.pixels[:] = linear_to_display(accum)

    def _finish_stats(self, width, height, frames_rendered, rays_cast, render_start):
        self._last_ray_count = rays_cast
        self._last_stats = RenderStats(
            width=width,
            height=height,
            samples_requested=self._samples,
            samples_rendered=frames_rendered,
            max_depth=self._max_depth,
            rays_cast=rays_cast,
            elapsed_seconds=time.perf_counter() - render_start,
        )
        print("\n" + self._last_stats.format_report())

    @property
    def last_ray_count(self):
        return self._last_ray_count

    @property
    def last_stats(self):
        return self._last_stats

    def __repr__(self):
        return (f"RayTracer(samples={self._samples}, max_depth={self._max_depth}, "
                f"direct_light_mode={self._direct_light_mode!r})")
