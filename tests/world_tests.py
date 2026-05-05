# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for World — object/light/camera management, active camera,
#              closest-hit intersection, non-renderable skipping, and sky color.
# ============================================

from scene import Sphere, Diffuse, SceneObject, World
from core import Vec3, Ray, Color
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _sphere_obj(translation=None, renderable=True):
    obj = SceneObject(
        shape=Sphere(),
        material=Diffuse(Color(1, 1, 1)),
        translation=translation,
        renderable=renderable,
    )
    return obj


def _ray(ox, oy, oz, dx, dy, dz):
    return Ray(Vec3(ox, oy, oz), Vec3(dx, dy, dz))


# ─── Defaults ─────────────────────────────────────────────────────────────────

def test_default_empty_objects():
    w = World()
    assert w.objects == []

def test_default_empty_lights():
    w = World()
    assert w.lights == []

def test_default_empty_cameras():
    w = World()
    assert w.cameras == []

def test_default_background_is_black():
    w = World()
    assert vec3_approx_eq(w.background_color, Color(0, 0, 0))

def test_default_active_camera_is_none():
    w = World()
    assert w.active_camera is None


# ─── Add / remove objects ─────────────────────────────────────────────────────

def test_add_object_increases_count():
    w = World()
    w.add_object(_sphere_obj())
    assert len(w.objects) == 1

def test_add_multiple_objects():
    w = World()
    w.add_object(_sphere_obj())
    w.add_object(_sphere_obj())
    assert len(w.objects) == 2

def test_remove_object_decreases_count():
    w = World()
    obj = _sphere_obj()
    w.add_object(obj)
    w.remove_object(obj)
    assert len(w.objects) == 0


# ─── Add / remove cameras ─────────────────────────────────────────────────────

def test_first_camera_becomes_active():
    w = World()
    cam = object()  # Cameras are opaque to World — any object works here.
    w.add_camera(cam)
    assert w.active_camera is cam

def test_second_camera_does_not_replace_active():
    w = World()
    cam1, cam2 = object(), object()
    w.add_camera(cam1)
    w.add_camera(cam2)
    assert w.active_camera is cam1

def test_set_active_camera():
    w = World()
    cam1, cam2 = object(), object()
    w.add_camera(cam1)
    w.add_camera(cam2)
    w.active_camera = cam2
    assert w.active_camera is cam2

def test_set_active_camera_not_in_world_raises():
    w = World()
    cam = object()
    try:
        w.active_camera = cam
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_remove_active_camera_clears_to_next():
    w = World()
    cam1, cam2 = object(), object()
    w.add_camera(cam1)
    w.add_camera(cam2)
    w.remove_camera(cam1)
    assert w.active_camera is cam2

def test_remove_last_camera_sets_active_none():
    w = World()
    cam = object()
    w.add_camera(cam)
    w.remove_camera(cam)
    assert w.active_camera is None


# ─── Background color ─────────────────────────────────────────────────────────

def test_background_color_setter():
    w = World()
    w.background_color = Color(0.1, 0.2, 0.3)
    assert vec3_approx_eq(w.background_color, Color(0.1, 0.2, 0.3))


# ─── intersect ────────────────────────────────────────────────────────────────

def test_intersect_empty_world_returns_none():
    w = World()
    hit = w.intersect(_ray(0, 0, -5, 0, 0, 1))
    assert hit is None

def test_intersect_hits_object():
    w = World()
    w.add_object(_sphere_obj())
    hit = w.intersect(_ray(0, 0, -5, 0, 0, 1))
    assert hit is not None

def test_intersect_misses_returns_none():
    w = World()
    w.add_object(_sphere_obj())
    hit = w.intersect(_ray(0, 5, -5, 0, 0, 1))
    assert hit is None

def test_intersect_skips_non_renderable():
    w = World()
    w.add_object(_sphere_obj(renderable=False))
    hit = w.intersect(_ray(0, 0, -5, 0, 0, 1))
    assert hit is None

def test_intersect_returns_closest():
    w = World()
    w.add_object(_sphere_obj(translation=Vec3(0, 0, -2)))  # nearer
    w.add_object(_sphere_obj(translation=Vec3(0, 0,  2)))  # farther
    hit = w.intersect(_ray(0, 0, -8, 0, 0, 1))
    # Nearer sphere is at z=-2 so hit point should have z < 0.
    assert hit.point.z < 0


# ─── sky_color ────────────────────────────────────────────────────────────────

def test_sky_color_use_sky_false_returns_background():
    bg = Color(0.1, 0.2, 0.3)
    w = World(background_color=bg, use_sky=False)
    result = w.sky_color(_ray(0, 0, 0, 0, 1, 0))
    assert vec3_approx_eq(result, bg)

def test_sky_color_upward_ray_is_blue():
    w = World(use_sky=True)
    result = w.sky_color(_ray(0, 0, 0, 0, 1, 0))  # pointing straight up
    # At y=1, t=1.0 → pure blue (0.5, 0.7, 1.0)
    assert vec3_approx_eq(result, Color(0.5, 0.7, 1.0))

def test_sky_color_downward_ray_is_white():
    w = World(use_sky=True)
    result = w.sky_color(_ray(0, 0, 0, 0, -1, 0))  # pointing straight down
    # At y=-1, t=0.0 → pure white (1.0, 1.0, 1.0)
    assert vec3_approx_eq(result, Color(1.0, 1.0, 1.0))

def test_sky_color_horizon_ray_is_midpoint():
    w = World(use_sky=True)
    result = w.sky_color(_ray(0, 0, 0, 1, 0, 0))  # horizontal, y=0
    # t=0.5 → lerp halfway between white and blue
    expected = Color(0.75, 0.85, 1.0)
    assert vec3_approx_eq(result, expected)


# ─── repr ─────────────────────────────────────────────────────────────────────

def test_repr_shows_counts():
    w = World()
    w.add_object(_sphere_obj())
    r = repr(w)
    assert "World" in r and "1" in r


if __name__ == "__main__":
    tests = [
        test_default_empty_objects,
        test_default_empty_lights,
        test_default_empty_cameras,
        test_default_background_is_black,
        test_default_active_camera_is_none,
        test_add_object_increases_count,
        test_add_multiple_objects,
        test_remove_object_decreases_count,
        test_first_camera_becomes_active,
        test_second_camera_does_not_replace_active,
        test_set_active_camera,
        test_set_active_camera_not_in_world_raises,
        test_remove_active_camera_clears_to_next,
        test_remove_last_camera_sets_active_none,
        test_background_color_setter,
        test_intersect_empty_world_returns_none,
        test_intersect_hits_object,
        test_intersect_misses_returns_none,
        test_intersect_skips_non_renderable,
        test_intersect_returns_closest,
        test_sky_color_use_sky_false_returns_background,
        test_sky_color_upward_ray_is_blue,
        test_sky_color_downward_ray_is_white,
        test_sky_color_horizon_ray_is_midpoint,
        test_repr_shows_counts,
    ]
    run_tests(tests)
