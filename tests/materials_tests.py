# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Tests for Material and Diffuse — albedo defaults, scatter output,
#              ray origin/direction, and abstract base enforcement.
# ============================================

from scene import Material, Diffuse
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


if __name__ == "__main__":
    tests = [
        test_material_is_abstract,
        test_diffuse_none_albedo_defaults_to_white,
        test_diffuse_custom_albedo_returned,
        test_scatter_returns_ray_and_attenuation,
        test_scatter_ray_origin_is_hit_point,
        test_scatter_ray_direction_is_unit_length,
        test_scatter_with_varied_normals,
    ]
    run_tests(tests)
