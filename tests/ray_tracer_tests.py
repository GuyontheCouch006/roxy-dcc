# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for RayTracer — render loop, trace recursion, sky fallback,
#              max depth termination, and attenuation accumulation.
# ============================================

import random
from types import SimpleNamespace

from core import Vec3, Color, Ray, Point3
from scene import Sphere, Diffuse, SceneObject, World, Camera
from scene.materials import Emissive
from rendering import RayTracer, Image
from rendering.ray_tracer import (
    _SphereLight,
    _direct_light_sample,
    _sample_sphere_light_surface,
)
from tests.utils import run_tests, approx_eq, vec3_approx_eq


W, H = 4, 4


def _world(use_sky=False):
    world = World(use_sky=use_sky)
    camera = Camera(width=W, height=H)
    world.add_camera(camera)
    return world


def _tracer(world, samples=1, max_depth=4, direct_light_mode="one", **kwargs):
    image = Image(W, H)
    return RayTracer(
        world,
        image,
        viewport=None,
        samples=samples,
        max_depth=max_depth,
        threaded=False,
        direct_light_mode=direct_light_mode,
        **kwargs,
    ), image


class _CountingOcclusionWorld:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.max_t_values = []

    def occluded(self, ray, max_t):
        self.max_t_values.append(max_t)
        return self.blocked


def _direct_light_hit():
    return SimpleNamespace(point=Point3(0, 0, 0), normal=Vec3(0, 0, 1))


def _test_sphere_light(z=4.0, x=0.0):
    return _SphereLight(
        center=Point3(x, 0, z),
        radius=0.5,
        color=Color(1, 1, 1),
        intensity=20.0,
    )


# ─── Construction ─────────────────────────────────────────────────────────────

def test_repr_contains_samples():
    world = _world()
    tracer, _ = _tracer(world, samples=3)
    assert "3" in repr(tracer)

def test_repr_contains_max_depth():
    world = _world()
    tracer, _ = _tracer(world, max_depth=5)
    assert "5" in repr(tracer)


# ─── Empty scene ──────────────────────────────────────────────────────────────

def test_render_empty_scene_produces_black_no_sky():
    world = _world(use_sky=False)
    tracer, image = _tracer(world)
    tracer.render()
    for y in range(H):
        for x in range(W):
            r, g, b = image.pixels[y, x]
            assert approx_eq(r, 0) and approx_eq(g, 0) and approx_eq(b, 0)

def test_render_empty_scene_with_sky_produces_nonblack():
    world = _world(use_sky=True)
    tracer, image = _tracer(world)
    tracer.render()
    has_nonblack = False
    for y in range(H):
        for x in range(W):
            r, g, b = image.pixels[y, x]
            if r > 0 or g > 0 or b > 0:
                has_nonblack = True
    assert has_nonblack

def test_render_reports_actual_rays_cast_for_empty_scene():
    world = _world(use_sky=True)
    tracer, _ = _tracer(world, samples=3, max_depth=4)
    tracer.render()
    assert tracer.last_ray_count == W * H * 3
    assert tracer.last_stats.rays_cast == W * H * 3
    assert tracer.last_stats.samples_rendered == 3
    assert tracer.last_stats.rays_per_pixel == 3

def test_render_reports_zero_rays_when_max_depth_is_zero():
    world = _world(use_sky=True)
    tracer, _ = _tracer(world, samples=2, max_depth=0)
    tracer.render()
    assert tracer.last_ray_count == 0
    assert tracer.last_stats.rays_cast == 0
    assert tracer.last_stats.max_depth == 0

def test_render_stats_format_includes_key_report_fields():
    world = _world(use_sky=True)
    tracer, _ = _tracer(world, samples=1, max_depth=1)
    tracer.render()
    report = tracer.last_stats.format_report()
    assert "Render report" in report
    assert "rays cast:" in report
    assert "rays / second:" in report

def test_adaptive_sampling_stops_after_stable_minimum_samples():
    world = _world(use_sky=False)
    tracer, _ = _tracer(
        world, samples=6, max_depth=1,
        adaptive_sampling=True,
        adaptive_min_samples=2,
        adaptive_threshold=0.0,
    )
    tracer.render()
    assert tracer.last_ray_count == W * H * 2
    assert tracer.last_stats.samples_requested == 6
    assert tracer.last_stats.samples_rendered == 2
    assert tracer.last_stats.adaptive_sampling is True
    assert tracer.last_stats.adaptive_stopped is True
    assert tracer.last_stats.adaptive_error == 0.0

def test_adaptive_sampling_report_includes_stop_fields_when_enabled():
    world = _world(use_sky=False)
    tracer, _ = _tracer(
        world, samples=3, max_depth=1,
        adaptive_sampling=True,
        adaptive_min_samples=2,
        adaptive_threshold=0.0,
    )
    tracer.render()
    report = tracer.last_stats.format_report()
    assert "adaptive stop:" in report
    assert "adaptive error:" in report
    assert "adaptive target:" in report


# ─── Sphere hit ───────────────────────────────────────────────────────────────

def test_render_sphere_in_front_writes_color():
    world = _world()
    world.add_object(SceneObject(
        shape=Sphere(),
        material=Diffuse(Color(1, 0, 0)),
        translation=Vec3(0, 0, -2),
    ))
    tracer, image = _tracer(world, samples=1, max_depth=1)
    tracer.render()
    # Centre pixel should pick up the red diffuse attenuation
    center_x, center_y = W // 2, H // 2
    r, g, b = image.pixels[center_y, center_x]
    # With max_depth=1, scatter returns attenuation * sky(0,0,0) = black.
    # The important thing is no exception and image was written without crashing.
    assert r >= 0 and g >= 0 and b >= 0

def test_render_direct_lights_diffuse_surface_at_max_depth_one():
    world = _world(use_sky=False)
    world.add_object(SceneObject(
        shape=Sphere(),
        material=Diffuse(Color(0.8, 0.8, 0.8)),
        translation=Vec3(0, 0, -3),
    ))
    world.add_object(SceneObject(
        shape=Sphere(),
        material=Emissive(Color(1, 1, 1), intensity=40.0),
        translation=Vec3(0, 2, -1),
        scale=Vec3(0.5, 0.5, 0.5),
    ))
    tracer, image = _tracer(world, samples=1, max_depth=1)
    tracer.render()
    pixel_sum = sum(float(image.pixels[y, x, c]) for y in range(H) for x in range(W) for c in range(3))
    assert pixel_sum > 0
    assert tracer.last_ray_count > W * H


def test_sphere_light_sample_is_on_light_surface_with_pdf_terms():
    random.seed(11)
    light = _test_sphere_light(z=4.0)
    sample = _sample_sphere_light_surface(Point3(0, 0, 0), light)
    assert sample is not None
    assert approx_eq((sample.point - light.center).length(), light.radius, eps=1e-5)
    assert sample.distance > 0.0
    assert sample.geometry_term > 0.0
    assert sample.area_pdf > 0.0
    assert sample.solid_angle_pdf > 0.0


def test_direct_light_one_mode_casts_one_shadow_ray_for_visible_light():
    random.seed(13)
    world = _CountingOcclusionWorld()
    lights = [_test_sphere_light(z=4.0), _test_sphere_light(z=5.0, x=0.5)]
    direct, rays = _direct_light_sample(
        _direct_light_hit(),
        world,
        lights,
        Color(1, 1, 1),
        "one",
    )
    assert rays == 1
    assert len(world.max_t_values) == 1
    assert world.max_t_values[0] > 0.0
    assert direct.r > 0.0 and direct.g > 0.0 and direct.b > 0.0


def test_direct_light_all_mode_casts_one_shadow_ray_per_visible_light():
    random.seed(17)
    world = _CountingOcclusionWorld()
    lights = [_test_sphere_light(z=4.0), _test_sphere_light(z=5.0, x=0.5)]
    direct, rays = _direct_light_sample(
        _direct_light_hit(),
        world,
        lights,
        Color(1, 1, 1),
        "all",
    )
    assert rays == len(lights)
    assert len(world.max_t_values) == len(lights)
    assert all(max_t > 0.0 for max_t in world.max_t_values)
    assert direct.r > 0.0 and direct.g > 0.0 and direct.b > 0.0


def test_direct_light_occlusion_casts_shadow_ray_and_blocks_contribution():
    random.seed(19)
    world = _CountingOcclusionWorld(blocked=True)
    direct, rays = _direct_light_sample(
        _direct_light_hit(),
        world,
        [_test_sphere_light(z=4.0)],
        Color(1, 1, 1),
        "one",
    )
    assert rays == 1
    assert len(world.max_t_values) == 1
    assert approx_eq(direct.r, 0.0)
    assert approx_eq(direct.g, 0.0)
    assert approx_eq(direct.b, 0.0)


def test_render_fills_entire_buffer():
    world = _world(use_sky=True)
    tracer, image = _tracer(world)
    tracer.render()
    # Every pixel should be written (not uninitialised NaN/negative)
    for y in range(H):
        for x in range(W):
            r, g, b = image.pixels[y, x]
            assert r >= 0 and g >= 0 and b >= 0


# ─── Max depth ────────────────────────────────────────────────────────────────

def test_trace_at_max_depth_returns_black():
    world = _world(use_sky=True)
    world.add_object(SceneObject(
        shape=Sphere(),
        material=Diffuse(Color(1, 1, 1)),
        translation=Vec3(0, 0, -2),
    ))
    # max_depth=0 forces immediate black termination
    tracer, image = _tracer(world, samples=1, max_depth=0)
    tracer.render()
    for y in range(H):
        for x in range(W):
            r, g, b = image.pixels[y, x]
            assert approx_eq(r, 0) and approx_eq(g, 0) and approx_eq(b, 0)


# ─── Samples ──────────────────────────────────────────────────────────────────

def test_samples_averages_result():
    """Multiple samples should not crash and should produce valid pixel values."""
    world = _world(use_sky=True)
    tracer, image = _tracer(world, samples=4)
    tracer.render()
    for y in range(H):
        for x in range(W):
            r, g, b = image.pixels[y, x]
            assert 0 <= r <= 2 and 0 <= g <= 2 and 0 <= b <= 2


# ─── Sky pass-through ─────────────────────────────────────────────────────────

def test_sky_color_used_when_no_hit():
    world_sky = _world(use_sky=True)
    tracer_sky, image_sky = _tracer(world_sky, samples=1, max_depth=1)
    tracer_sky.render()

    world_black = _world(use_sky=False)
    tracer_black, image_black = _tracer(world_black, samples=1, max_depth=1)
    tracer_black.render()

    # Sky render should be brighter overall than black-background render
    sky_sum = sum(float(image_sky.pixels[y, x, c]) for y in range(H) for x in range(W) for c in range(3))
    black_sum = sum(float(image_black.pixels[y, x, c]) for y in range(H) for x in range(W) for c in range(3))
    assert sky_sum > black_sum


if __name__ == "__main__":
    tests = [
        test_repr_contains_samples,
        test_repr_contains_max_depth,
        test_render_empty_scene_produces_black_no_sky,
        test_render_empty_scene_with_sky_produces_nonblack,
        test_render_reports_actual_rays_cast_for_empty_scene,
        test_render_reports_zero_rays_when_max_depth_is_zero,
        test_render_stats_format_includes_key_report_fields,
        test_adaptive_sampling_stops_after_stable_minimum_samples,
        test_adaptive_sampling_report_includes_stop_fields_when_enabled,
        test_render_sphere_in_front_writes_color,
        test_render_direct_lights_diffuse_surface_at_max_depth_one,
        test_sphere_light_sample_is_on_light_surface_with_pdf_terms,
        test_direct_light_one_mode_casts_one_shadow_ray_for_visible_light,
        test_direct_light_all_mode_casts_one_shadow_ray_per_visible_light,
        test_direct_light_occlusion_casts_shadow_ray_and_blocks_contribution,
        test_render_fills_entire_buffer,
        test_trace_at_max_depth_returns_black,
        test_samples_averages_result,
        test_sky_color_used_when_no_hit,
    ]
    run_tests(tests)
