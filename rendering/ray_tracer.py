import time
import math
import random
from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
from core import Color, Point3, Ray
from rendering.denoise import edge_aware_denoise, linear_to_display
from rendering.intersector import WorldIntersector
from rendering.render_stats import RenderStats
from rendering.sampling import clamp_color_sample, pixel_sample_offset

_worker_world  = None
_worker_camera = None
_worker_intersector = None
_worker_lights = None
_worker_direct_light_mode = None
_worker_direct_light_max_depth = None
_worker_sample_clamp = None


@dataclass(frozen=True)
class _SphereLight:
    center: Point3
    radius: float
    color: Color
    intensity: float


@dataclass(frozen=True)
class _SphereLightSample:
    point: Point3
    normal: Point3
    direction: Point3
    distance: float
    geometry_term: float
    area_pdf: float
    solid_angle_pdf: float


_SHADOW_EPSILON = 1e-4
_PDF_EPSILON = 1e-12


def _init_worker(world, camera, intersector, lights, direct_light_mode,
                 direct_light_max_depth, sample_clamp):
    global _worker_world, _worker_camera, _worker_intersector, _worker_lights
    global _worker_direct_light_mode, _worker_direct_light_max_depth
    global _worker_sample_clamp
    _worker_world  = world
    _worker_camera = camera
    _worker_intersector = intersector
    _worker_lights = lights
    _worker_direct_light_mode = direct_light_mode
    _worker_direct_light_max_depth = direct_light_max_depth
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


def _orthonormal_basis(w):
    helper = Point3(0, 1, 0) if abs(w.x) > 0.9 else Point3(1, 0, 0)
    bitangent = w.cross(helper).normalize()
    tangent = bitangent.cross(w)
    return tangent, bitangent


def _sample_sphere_light_surface(hit_point, light):
    to_center = light.center - hit_point
    center_dist2 = max(to_center.length_sq(), _PDF_EPSILON)
    radius2 = light.radius * light.radius
    if center_dist2 <= radius2 + _PDF_EPSILON:
        return None

    center_dist = math.sqrt(center_dist2)
    center_dir = to_center / center_dist
    sin_theta_max2 = min(1.0, radius2 / center_dist2)
    cos_theta_max = math.sqrt(max(0.0, 1.0 - sin_theta_max2))
    solid_angle = 2.0 * math.pi * (1.0 - cos_theta_max)
    if solid_angle <= _PDF_EPSILON:
        return None

    u1 = random.random()
    u2 = random.random()
    cos_theta = 1.0 - u1 * (1.0 - cos_theta_max)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * u2

    tangent, bitangent = _orthonormal_basis(center_dir)
    direction = (
        tangent * (math.cos(phi) * sin_theta) +
        bitangent * (math.sin(phi) * sin_theta) +
        center_dir * cos_theta
    ).normalize()

    projection = to_center.dot(direction)
    closest_dist2 = center_dist2 - projection * projection
    hit_offset2 = radius2 - closest_dist2
    if hit_offset2 < -_PDF_EPSILON:
        return None

    distance = projection - math.sqrt(max(0.0, hit_offset2))
    if distance <= _SHADOW_EPSILON:
        return None

    point = hit_point + direction * distance
    normal = (point - light.center).normalize()
    light_cos = max(0.0, (-direction).dot(normal))
    if light_cos <= _PDF_EPSILON:
        return None

    geometry_term = light_cos / max(distance * distance, _PDF_EPSILON)
    solid_angle_pdf = 1.0 / solid_angle
    area_pdf = solid_angle_pdf * geometry_term
    if area_pdf <= _PDF_EPSILON:
        return None

    return _SphereLightSample(
        point=point,
        normal=normal,
        direction=direction,
        distance=distance,
        geometry_term=geometry_term,
        area_pdf=area_pdf,
        solid_angle_pdf=solid_angle_pdf,
    )


def _direct_light_sample_one(hit, intersector, light, albedo, sample_weight):
    sample = _sample_sphere_light_surface(hit.point, light)
    if sample is None:
        return Color(0, 0, 0), 0

    ndotl = hit.normal.dot(sample.direction)
    if ndotl <= 0:
        return Color(0, 0, 0), 0

    shadow_origin = hit.point + hit.normal * _SHADOW_EPSILON
    to_sample = sample.point - shadow_origin
    shadow_dist = to_sample.length()
    max_t = shadow_dist - _SHADOW_EPSILON
    if max_t <= 0.001:
        return Color(0, 0, 0), 0

    shadow_ray = Ray(shadow_origin, to_sample)
    if intersector.occluded(shadow_ray, max_t):
        return Color(0, 0, 0), 1

    light_pdf = sample.area_pdf
    contribution = (
        albedo *
        light.color *
        light.intensity *
        ndotl *
        sample.geometry_term *
        (1.0 / math.pi) *
        (1.0 / light_pdf) *
        sample_weight
    )
    return contribution, 1


def _direct_light_sample(hit, intersector, lights, albedo, direct_light_mode):
    if not lights:
        return Color(0, 0, 0), 0

    if direct_light_mode == "all":
        direct = Color(0, 0, 0)
        rays_cast = 0
        for light in lights:
            contribution, shadow_rays = _direct_light_sample_one(
                hit, intersector, light, albedo, 1.0)
            direct += contribution
            rays_cast += shadow_rays
        return direct, rays_cast

    return _direct_light_sample_one(
        hit, intersector, random.choice(lights), albedo, len(lights))


def _guide_tuple(color, rays_cast, collect_guides,
                 normal=None, albedo=None, depth_value=0.0):
    if collect_guides:
        normal = normal or Color(0, 0, 0)
        albedo = albedo or Color(0, 0, 0)
        return color, rays_cast, normal, albedo, depth_value
    return color, rays_cast


def _trace(ray, world, intersector, max_depth, lights=None,
           direct_light_mode="one", depth=0, collect_guides=False,
           direct_light_max_depth=1):
    if depth >= max_depth:
        return _guide_tuple(Color(0, 0, 0), 0, collect_guides)

    rays_cast = 1
    hit = intersector.intersect(ray)
    if hit is None:
        return _guide_tuple(world.sky_color(ray), rays_cast, collect_guides)

    mat      = hit.material
    emission = mat.emitted() if hasattr(mat, 'emitted') else Color(0, 0, 0)
    result   = mat.scatter(ray, hit)
    if result is None:
        return _guide_tuple(
            emission, rays_cast, collect_guides,
            normal=hit.normal, albedo=emission, depth_value=hit.t,
        )

    scattered, attenuation = result
    direct = Color(0, 0, 0)
    if mat.taichi_type_id() == 0 and depth < direct_light_max_depth:
        direct, shadow_rays = _direct_light_sample(
            hit, intersector, lights or [], attenuation, direct_light_mode)
        rays_cast += shadow_rays
    elif mat.taichi_type_id() == 4 and depth < direct_light_max_depth:
        roughness = mat.taichi_params()[0]
        direct, shadow_rays = _direct_light_sample(
            hit, intersector, lights or [], attenuation, direct_light_mode)
        direct *= roughness
        rays_cast += shadow_rays

    if depth > 2:
        survival = max(attenuation[0], attenuation[1], attenuation[2])
        if survival < 0.1:
            return _guide_tuple(
                emission, rays_cast, collect_guides,
                normal=hit.normal, albedo=attenuation, depth_value=hit.t,
            )
        attenuation = attenuation / survival

    bounced, bounce_rays = _trace(
        scattered, world, intersector, max_depth, lights,
        direct_light_mode, depth + 1,
        direct_light_max_depth=direct_light_max_depth,
    )
    color = emission + direct + attenuation * bounced
    return _guide_tuple(
        color, rays_cast + bounce_rays, collect_guides,
        normal=hit.normal, albedo=attenuation, depth_value=hit.t,
    )


def _trace_band_worker(args):
    """One sample per pixel, returns raw linear colors (no gamma)."""
    y_start, y_end, width, height, max_depth, frame = args
    band = np.zeros((y_end - y_start, width, 3), dtype=np.float32)
    normal_band = np.zeros_like(band)
    albedo_band = np.zeros_like(band)
    depth_band = np.zeros((y_end - y_start, width), dtype=np.float32)
    rays_cast = 0
    for y in range(y_start, y_end):
        for x in range(width):
            ray = _worker_camera.shoot(
                x, y, width, height,
                jitter=pixel_sample_offset(x, y, frame),
            )
            c, ray_count, normal, albedo, depth_value = _trace(
                ray, _worker_world, _worker_intersector, max_depth, _worker_lights,
                _worker_direct_light_mode, collect_guides=True,
                direct_light_max_depth=_worker_direct_light_max_depth)
            c = clamp_color_sample(c, _worker_sample_clamp)
            rays_cast += ray_count
            band[y - y_start, x] = (c[0], c[1], c[2])
            normal_band[y - y_start, x] = (normal[0], normal[1], normal[2])
            albedo_band[y - y_start, x] = (albedo[0], albedo[1], albedo[2])
            depth_band[y - y_start, x] = depth_value
    return y_start, band, normal_band, albedo_band, depth_band, rays_cast


class RayTracer:
    def __init__(self, world, image, viewport, samples=64, max_depth=8, threaded=True,
                 direct_light_mode="one", denoise=False,
                 denoise_radius=1, denoise_sigma=0.08, denoise_amount=0.8,
                 sample_clamp=10.0, adaptive_sampling=False,
                 adaptive_min_samples=4, adaptive_threshold=0.002,
                 adaptive_check_interval=1, direct_light_max_depth=1,
                 intersector=None, startup_progress=None):
        self._world     = world
        self._image     = image
        self._viewport  = viewport
        self._camera    = world.active_camera
        self._intersector = intersector or WorldIntersector(world)
        self._samples   = samples
        self._max_depth = max_depth
        self._threaded  = threaded
        if direct_light_mode not in ("one", "random", "sample", "all", "final"):
            raise ValueError("direct_light_mode must be 'one' or 'all'")
        self._direct_light_mode = "all" if direct_light_mode in ("all", "final") else "one"
        if direct_light_max_depth is None:
            direct_light_max_depth = max_depth
        self._direct_light_max_depth = max(0, int(direct_light_max_depth))
        self._denoise = denoise
        self._denoise_radius = denoise_radius
        self._denoise_sigma = denoise_sigma
        self._denoise_amount = denoise_amount
        self._sample_clamp = sample_clamp
        if adaptive_min_samples < 1:
            raise ValueError("adaptive_min_samples must be at least 1")
        if adaptive_threshold < 0:
            raise ValueError("adaptive_threshold must be non-negative")
        if adaptive_check_interval < 1:
            raise ValueError("adaptive_check_interval must be at least 1")
        self._adaptive_sampling = adaptive_sampling
        self._adaptive_min_samples = adaptive_min_samples
        self._adaptive_threshold = adaptive_threshold
        self._adaptive_check_interval = adaptive_check_interval
        self._adaptive_stopped = False
        self._last_adaptive_error = None
        self._last_ray_count = 0
        self._last_stats = None
        self._lights = _collect_emissive_sphere_lights(world)
        self._startup_progress = startup_progress

    def render(self):
        self._adaptive_stopped = False
        self._last_adaptive_error = None
        if self._threaded:
            self._render_threaded()
        else:
            self._render_single()

    def _render_single(self):
        W, H  = self._image.width, self._image.height
        accum = np.zeros((H, W, 3), dtype=np.float32)
        normal_accum = np.zeros_like(accum)
        albedo_accum = np.zeros_like(accum)
        depth_accum = np.zeros((H, W), dtype=np.float32)
        rays_cast = 0
        frames_rendered = 0
        render_start = time.perf_counter()
        previous_accum = None
        last_viewport_update = 0.0
        first_viewport_update = False
        if self._startup_progress:
            self._startup_progress.step(
                "Tracing first rows",
                "The viewport will update as soon as the first rows finish.",
            )

        for frame in range(self._samples):
            frames_rendered = frame + 1
            for y in range(H):
                for x in range(W):
                    ray = self._camera.shoot(
                        x, y, W, H,
                        jitter=pixel_sample_offset(x, y, frame),
                    )
                    c, ray_count, normal, albedo, depth_value = _trace(
                        ray, self._world, self._intersector,
                        self._max_depth, self._lights,
                        self._direct_light_mode, collect_guides=True,
                        direct_light_max_depth=self._direct_light_max_depth)
                    c = clamp_color_sample(c, self._sample_clamp)
                    rays_cast += ray_count
                    accum[y, x] = (accum[y, x] * frame + [c[0], c[1], c[2]]) / (frame + 1)
                    normal_accum[y, x] = (
                        normal_accum[y, x] * frame + [normal[0], normal[1], normal[2]]
                    ) / (frame + 1)
                    albedo_accum[y, x] = (
                        albedo_accum[y, x] * frame + [albedo[0], albedo[1], albedo[2]]
                    ) / (frame + 1)
                    depth_accum[y, x] = (
                        depth_accum[y, x] * frame + depth_value
                    ) / (frame + 1)

                self._image.pixels[y:y+1] = np.sqrt(np.minimum(accum[y:y+1], 1.0))
                now = time.perf_counter()
                if (
                    self._viewport
                    and (not first_viewport_update or now - last_viewport_update >= 0.25)
                ):
                    self._viewport.update(self._image)
                    self._viewport.poll_events()
                    if not first_viewport_update:
                        self._close_startup_progress()
                    first_viewport_update = True
                    last_viewport_update = now
                    if self._viewport.should_close:
                        self._apply_final_pixels(accum, normal_accum, albedo_accum, depth_accum)
                        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)
                        return

            self._image.pixels[:] = np.sqrt(np.minimum(accum, 1.0))
            should_stop = self._adaptive_should_stop(previous_accum, accum, frames_rendered)
            previous_accum = self._adaptive_snapshot(accum)
            print(f"\r  sample {frame + 1}/{self._samples}", end='', flush=True)

            if self._viewport:
                self._viewport.update(self._image)
                self._viewport.poll_events()
                if self._viewport.should_close:
                    self._apply_final_pixels(accum, normal_accum, albedo_accum, depth_accum)
                    self._finish_stats(W, H, frames_rendered, rays_cast, render_start)
                    return

            if should_stop:
                self._adaptive_stopped = True
                break

        self._apply_final_pixels(accum, normal_accum, albedo_accum, depth_accum)
        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)

    def _render_threaded(self):
        W, H      = self._image.width, self._image.height
        band_size = 6
        accum     = np.zeros((H, W, 3), dtype=np.float32)
        normal_accum = np.zeros_like(accum)
        albedo_accum = np.zeros_like(accum)
        depth_accum = np.zeros((H, W), dtype=np.float32)
        rays_cast = 0
        frames_rendered = 0
        render_start = time.perf_counter()
        previous_accum = None
        last_viewport_update = 0.0
        first_viewport_update = False
        if self._startup_progress:
            self._startup_progress.step(
                "Tracing first bands",
                "The viewport will update as soon as the first worker band finishes.",
            )

        band_ranges = [
            (y, min(y + band_size, H), W, H, self._max_depth)
            for y in range(0, H, band_size)
        ]

        with Pool(
            processes=10,
            initializer=_init_worker,
            initargs=(
                self._world, self._camera, self._intersector, self._lights,
                self._direct_light_mode, self._direct_light_max_depth,
                self._sample_clamp,
            ),
        ) as pool:
            for frame in range(self._samples):
                frames_rendered = frame + 1
                tasks = [(*band, frame) for band in band_ranges]
                for (
                    y_start, band, normal_band, albedo_band, depth_band, band_rays
                ) in pool.imap_unordered(_trace_band_worker, tasks):
                    rays_cast += band_rays
                    y_end = y_start + len(band)
                    accum[y_start:y_end] = (accum[y_start:y_end] * frame + band) / (frame + 1)
                    normal_accum[y_start:y_end] = (
                        normal_accum[y_start:y_end] * frame + normal_band
                    ) / (frame + 1)
                    albedo_accum[y_start:y_end] = (
                        albedo_accum[y_start:y_end] * frame + albedo_band
                    ) / (frame + 1)
                    depth_accum[y_start:y_end] = (
                        depth_accum[y_start:y_end] * frame + depth_band
                    ) / (frame + 1)
                    self._image.pixels[y_start:y_end] = np.sqrt(
                        np.minimum(accum[y_start:y_end], 1.0)
                    )
                    now = time.perf_counter()
                    if (
                        self._viewport
                        and (not first_viewport_update or now - last_viewport_update >= 0.25)
                    ):
                        self._viewport.update(self._image)
                        self._viewport.poll_events()
                        if not first_viewport_update:
                            self._close_startup_progress()
                        first_viewport_update = True
                        last_viewport_update = now
                        if self._viewport.should_close:
                            pool.terminate()
                            self._apply_final_pixels(
                                accum,
                                normal_accum,
                                albedo_accum,
                                depth_accum,
                            )
                            self._finish_stats(
                                W,
                                H,
                                frames_rendered,
                                rays_cast,
                                render_start,
                            )
                            return

                should_stop = self._adaptive_should_stop(previous_accum, accum, frames_rendered)
                previous_accum = self._adaptive_snapshot(accum)
                print(f"\r  sample {frame + 1}/{self._samples}", end='', flush=True)

                if self._viewport:
                    self._viewport.update(self._image)
                    self._viewport.poll_events()
                    if self._viewport.should_close:
                        pool.terminate()
                        self._apply_final_pixels(accum, normal_accum, albedo_accum, depth_accum)
                        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)
                        return

                if should_stop:
                    self._adaptive_stopped = True
                    break

        self._apply_final_pixels(accum, normal_accum, albedo_accum, depth_accum)
        self._finish_stats(W, H, frames_rendered, rays_cast, render_start)

    def _close_startup_progress(self):
        if self._startup_progress:
            self._startup_progress.close()
            self._startup_progress = None

    def _adaptive_snapshot(self, accum):
        if not self._adaptive_sampling:
            return None
        return accum.copy()

    def _adaptive_should_stop(self, previous_accum, accum, samples_rendered):
        if not self._adaptive_sampling or previous_accum is None:
            return False
        self._last_adaptive_error = float(np.mean(np.abs(accum - previous_accum)))
        if samples_rendered >= self._samples:
            return False
        if samples_rendered < self._adaptive_min_samples:
            return False
        if (samples_rendered - self._adaptive_min_samples) % self._adaptive_check_interval != 0:
            return False
        return self._last_adaptive_error <= self._adaptive_threshold

    def _apply_final_pixels(self, accum, normal=None, albedo=None, depth=None):
        if self._denoise:
            accum = edge_aware_denoise(
                accum,
                radius=self._denoise_radius,
                sigma_color=self._denoise_sigma,
                amount=self._denoise_amount,
                normal=normal,
                albedo=albedo,
                depth=depth,
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
            adaptive_sampling=self._adaptive_sampling,
            adaptive_stopped=self._adaptive_stopped,
            adaptive_threshold=self._adaptive_threshold,
            adaptive_error=self._last_adaptive_error,
            adaptive_min_samples=self._adaptive_min_samples,
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
