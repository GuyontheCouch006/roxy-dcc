# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for all material types — Diffuse, Metal, Dielectric, Emissive.
# ============================================

from scene import Material, Diffuse
from scene.materials import Metal, Dielectric, Emissive
from core import Vec3, Ray, Color, HitRecord
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _make_hit(normal=None):
    """Create a simple front-face HitRecord for testing."""
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    outward = normal if normal is not None else Vec3(0, 0, -1)
    return HitRecord.from_ray(ray, 4.0, outward)


# ─── Material ABC ─────────────────────────────────────────────────────────────

def test_material_is_abstract():
    try:
        Material(Color(1, 1, 1))
        assert False, "Material should not be instantiable"
    except TypeError:
        pass


# ─── Diffuse — albedo ─────────────────────────────────────────────────────────

def test_diffuse_none_albedo_defaults_to_white():
    mat = Diffuse(None)
    _, attenuation = mat.scatter(None, _make_hit())
    assert vec3_approx_eq(attenuation, Color(1, 1, 1)), \
        f"Expected white attenuation, got {attenuation}"

def test_diffuse_custom_albedo_returned():
    albedo = Color(0.8, 0.2, 0.4)
    mat = Diffuse(albedo)
    _, attenuation = mat.scatter(None, _make_hit())
    assert vec3_approx_eq(attenuation, albedo), \
        f"Expected {albedo}, got {attenuation}"


# ─── Diffuse — scatter output ─────────────────────────────────────────────────

def test_scatter_returns_ray_and_attenuation():
    mat = Diffuse(Color(1, 1, 1))
    result = mat.scatter(None, _make_hit())
    assert len(result) == 2
    assert isinstance(result[0], Ray)

def test_scatter_ray_origin_is_hit_point():
    hit = _make_hit()
    mat = Diffuse(Color(1, 1, 1))
    scattered, _ = mat.scatter(None, hit)
    assert vec3_approx_eq(scattered.origin, hit.point), \
        f"Expected ray origin {hit.point}, got {scattered.origin}"

def test_scatter_ray_direction_is_unit_length():
    mat = Diffuse(Color(1, 1, 1))
    scattered, _ = mat.scatter(None, _make_hit())
    assert approx_eq(scattered.direction.length(), 1.0), \
        f"Expected unit direction, got length {scattered.direction.length()}"

def test_scatter_with_varied_normals():
    for normal in [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]:
        mat = Diffuse(Color(0.5, 0.5, 0.5))
        scattered, attenuation = mat.scatter(None, _make_hit(normal))
        assert isinstance(scattered, Ray)
        assert approx_eq(scattered.direction.length(), 1.0)


def _metal_hit():
    """Front-face hit on an upward surface, ray coming straight down."""
    ray = Ray(Vec3(0, 5, 0), Vec3(0, -1, 0))
    return ray, HitRecord.from_ray(ray, 4.0, Vec3(0, 1, 0))


def _dielectric_hit():
    """Front-face hit on a surface facing the ray."""
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    return ray, HitRecord.from_ray(ray, 4.0, Vec3(0, 0, -1))


# ─── Metal ────────────────────────────────────────────────────────────────────

def test_metal_scatter_returns_ray_and_attenuation():
    mat = Metal(Color(0.8, 0.8, 0.8), roughness=0.0)
    ray, hit = _metal_hit()
    result = mat.scatter(ray, hit)
    assert result is not None
    scattered, attenuation = result
    assert isinstance(scattered, Ray)

def test_metal_perfect_mirror_direction():
    mat = Metal(Color(1, 1, 1), roughness=0.0)
    ray, hit = _metal_hit()    # ray goes straight down, normal points up
    scattered, _ = mat.scatter(ray, hit)
    assert approx_eq(scattered.direction.y, 1.0, eps=1e-5), \
        "Reflecting straight down off upward surface should scatter straight up"

def test_metal_albedo_returned():
    albedo = Color(0.9, 0.3, 0.1)
    mat = Metal(albedo, roughness=0.0)
    ray, hit = _metal_hit()
    _, atten = mat.scatter(ray, hit)
    assert approx_eq(atten[0], albedo[0]) and approx_eq(atten[1], albedo[1])

def test_metal_scattered_direction_unit_length():
    mat = Metal(Color(0.8, 0.8, 0.8), roughness=0.0)
    ray, hit = _metal_hit()
    scattered, _ = mat.scatter(ray, hit)
    assert approx_eq(scattered.direction.length(), 1.0)


# ─── Dielectric ───────────────────────────────────────────────────────────────

def test_dielectric_scatter_not_none():
    mat = Dielectric(Color(1, 1, 1), ior=1.5)
    ray, hit = _dielectric_hit()
    assert mat.scatter(ray, hit) is not None

def test_dielectric_scattered_direction_unit_length():
    mat = Dielectric(Color(1, 1, 1), ior=1.5)
    ray, hit = _dielectric_hit()
    scattered, _ = mat.scatter(ray, hit)
    assert approx_eq(scattered.direction.length(), 1.0, eps=1e-5)

def test_dielectric_attenuation_is_albedo():
    albedo = Color(0.9, 0.95, 1.0)
    mat = Dielectric(albedo, ior=1.5)
    ray, hit = _dielectric_hit()
    _, atten = mat.scatter(ray, hit)
    assert approx_eq(atten[0], albedo[0]) and approx_eq(atten[1], albedo[1])


# ─── Emissive ─────────────────────────────────────────────────────────────────

def test_emissive_scatter_returns_none():
    mat = Emissive(Color(1, 1, 1), intensity=5.0)
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
    assert mat.scatter(ray, _make_hit()) is None

def test_emissive_emitted_scales_by_intensity():
    mat = Emissive(Color(1, 0.5, 0.25), intensity=4.0)
    e = mat.emitted()
    assert approx_eq(e[0], 4.0),   f"Expected r=4.0, got {e[0]}"
    assert approx_eq(e[1], 2.0),   f"Expected g=2.0, got {e[1]}"
    assert approx_eq(e[2], 1.0),   f"Expected b=1.0, got {e[2]}"

def test_emissive_zero_intensity():
    mat = Emissive(Color(1, 1, 1), intensity=0.0)
    e = mat.emitted()
    assert approx_eq(e[0], 0.0) and approx_eq(e[1], 0.0)

def test_emissive_taichi_type_id():
    mat = Emissive(Color(1, 1, 1), intensity=1.0)
    assert mat.taichi_type_id() == 3


if __name__ == "__main__":
    tests = [
        test_material_is_abstract,
        test_diffuse_none_albedo_defaults_to_white,
        test_diffuse_custom_albedo_returned,
        test_scatter_returns_ray_and_attenuation,
        test_scatter_ray_origin_is_hit_point,
        test_scatter_ray_direction_is_unit_length,
        test_scatter_with_varied_normals,
        test_metal_scatter_returns_ray_and_attenuation,
        test_metal_perfect_mirror_direction,
        test_metal_albedo_returned,
        test_metal_scattered_direction_unit_length,
        test_dielectric_scatter_not_none,
        test_dielectric_scattered_direction_unit_length,
        test_dielectric_attenuation_is_albedo,
        test_emissive_scatter_returns_none,
        test_emissive_emitted_scales_by_intensity,
        test_emissive_zero_intensity,
        test_emissive_taichi_type_id,
    ]
    run_tests(tests)
