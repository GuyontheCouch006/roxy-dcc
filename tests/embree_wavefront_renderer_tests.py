import importlib
import inspect

import numpy as np

from core import Color, Point3, Vec3
from rendering import Image
from rendering.intersector import TriangleArrayIntersector
from rendering.scene_arrays import flatten_world_triangles
from scene import Camera, Diffuse, Emissive, Mesh, SceneObject, Sphere, Triangle, World
from tests.utils import run_tests


W, H = 4, 4
PRIMARY_RAYS = W * H


class _BatchedTriangleIntersector:
    """Embree-shaped test double backed by the reference triangle arrays."""

    backend_name = "test-batched-triangle"

    def __init__(self, world):
        self._triangle_scene = flatten_world_triangles(world)
        self._reference = TriangleArrayIntersector(self._triangle_scene)

    @property
    def triangle_count(self):
        return self._triangle_scene.triangle_count

    @property
    def triangle_scene(self):
        return self._triangle_scene

    def intersect_raw_arrays(self, origins, directions, max_t=None):
        return self._reference.intersect_raw_arrays(origins, directions, max_t)

    def occluded_raw_arrays(self, origins, directions, max_t):
        return self._reference.occluded_raw_arrays(origins, directions, max_t)


def _wavefront_module():
    try:
        return importlib.import_module("rendering.embree_wavefront_renderer")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "Expected rendering.embree_wavefront_renderer to exist"
        ) from exc


def _renderer_class(module):
    for name in (
        "EmbreeWavefrontRenderer",
        "EmbreeBatchedRenderer",
        "WavefrontEmbreeRenderer",
    ):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise AssertionError("Expected an Embree wavefront renderer class")


def _make_renderer(cls, world, image, intersector):
    kwargs = {
        "intersector": intersector,
        "samples": 1,
        "max_depth": 1,
        "direct_light_mode": "one",
    }

    signature = inspect.signature(cls)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    accepted = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in signature.parameters
    }

    attempts = [
        lambda: cls(world, image, viewport=None, **accepted),
        lambda: cls(world, image, None, **accepted),
        lambda: cls(world, image, **accepted),
    ]
    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))

    raise AssertionError(
        "Could not construct Embree wavefront renderer with world/image/"
        f"viewport/intersector test API: {'; '.join(errors)}"
    )


def _world():
    world = World(use_sky=False)
    world.add_camera(Camera(width=W, height=H, fov=70))
    return world


def _triangle_object(material):
    triangle = Triangle(
        Point3(-0.9, -0.9, -2.0),
        Point3(0.9, -0.9, -2.0),
        Point3(0.0, 0.9, -2.0),
    )
    return SceneObject(shape=Mesh([triangle]), material=material)


def _render(world):
    module = _wavefront_module()
    cls = _renderer_class(module)
    image = Image(W, H)
    renderer = _make_renderer(cls, world, image, _BatchedTriangleIntersector(world))
    renderer.render()
    return renderer, image


def _pixel_sum(image):
    return float(np.sum(image.pixels))


def test_emissive_triangle_renders_nonzero_pixels_in_batched_wavefront_renderer():
    world = _world()
    world.add_object(_triangle_object(Emissive(Color(1.0, 0.8, 0.5), intensity=3.0)))

    renderer, image = _render(world)

    assert _pixel_sum(image) > 0.0
    assert renderer.last_stats.rays_cast >= PRIMARY_RAYS


def test_diffuse_triangle_direct_lighting_reports_shadow_rays():
    world = _world()
    world.add_object(_triangle_object(Diffuse(Color(0.8, 0.8, 0.8))))
    world.add_object(SceneObject(
        shape=Sphere(radius=0.1),
        material=Emissive(Color(1.0, 1.0, 1.0), intensity=40.0),
        translation=Vec3(0.0, 0.0, -0.5),
    ))

    renderer, image = _render(world)

    assert _pixel_sum(image) > 0.0
    assert renderer.last_stats.rays_cast > PRIMARY_RAYS


def test_wavefront_texture_sampling_matches_scalar_image_texture_if_exposed():
    module = _wavefront_module()
    sample_pixels = getattr(module, "_sample_pixels", None)
    if sample_pixels is None:
        return

    from core import Vec2
    from scene.textures import ImageTexture

    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    texture = ImageTexture.from_array(pixels, flip_v=False)
    uvs = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [1.25, -0.25],
    ], dtype=np.float32)

    batch = sample_pixels(pixels, uvs, flip_v=False)

    for i, (u, v) in enumerate(uvs):
        scalar = texture.sample(Vec2(float(u), float(v)))
        assert abs(float(batch[i, 0]) - scalar.r) < 1e-6
        assert abs(float(batch[i, 1]) - scalar.g) < 1e-6
        assert abs(float(batch[i, 2]) - scalar.b) < 1e-6


if __name__ == "__main__":
    run_tests([
        test_emissive_triangle_renders_nonzero_pixels_in_batched_wavefront_renderer,
        test_diffuse_triangle_direct_lighting_reports_shadow_rays,
        test_wavefront_texture_sampling_matches_scalar_image_texture_if_exposed,
    ])
