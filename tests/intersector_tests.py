from core import Color, Point3, Ray, Vec2, Vec3
import sys
import types

import numpy as np

from rendering.intersector import (
    EmbreeIntersector,
    EmbreeUnavailableError,
    TriangleArrayIntersector,
    WorldIntersector,
)
from rendering.ray_tracer import RayTracer
from rendering.image import Image
from rendering.scene_arrays import flatten_world_triangles
from scene import Camera, Diffuse, SceneObject, World
from scene.materials import Emissive
from scene.mesh import IndexedMesh
from scene.shape import Shape
from tests.utils import approx_eq, run_tests


W = H = 4


class _FakeEmbreeScene:
    def __init__(self):
        self.vertices = None
        self.indices = None

    def run(self, origins, directions, query="INTERSECT", output=True, dists=None):
        if query == "OCCLUDED":
            return {
                "primID": np.asarray([0, -1], dtype=np.int32),
                "tfar": np.asarray([2.0, np.inf], dtype=np.float32),
            }
        return {
            "primID": np.asarray([0, -1], dtype=np.int32),
            "geomID": np.asarray([0, -1], dtype=np.int32),
            "tfar": np.asarray([2.0, np.inf], dtype=np.float32),
            "u": np.asarray([0.25, 0.0], dtype=np.float32),
            "v": np.asarray([0.25, 0.0], dtype=np.float32),
        }


class _FakeTriangleMesh:
    def __init__(self, scene, vertices, indices):
        scene.vertices = vertices
        scene.indices = indices


def _install_fake_embree_modules():
    package = types.ModuleType("fake_embree")
    scene_module = types.ModuleType("fake_embree.rtcore_scene")
    mesh_module = types.ModuleType("fake_embree.mesh_construction")
    scene_module.EmbreeScene = _FakeEmbreeScene
    mesh_module.TriangleMesh = _FakeTriangleMesh
    sys.modules["fake_embree"] = package
    sys.modules["fake_embree.rtcore_scene"] = scene_module
    sys.modules["fake_embree.mesh_construction"] = mesh_module


def _mesh_world(material=None):
    material = material or Diffuse(Color(0.25, 0.5, 0.75))
    mesh = IndexedMesh(
        positions=[[-1, -1, -2], [1, -1, -2], [0, 1, -2]],
        tri_pos_idx=[[0, 1, 2]],
        normals=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        tri_normal_idx=[[0, 1, 2]],
        uvs=[[0, 0], [1, 0], [0.5, 1]],
        tri_uv_idx=[[0, 1, 2]],
        groups=["mat"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    world = World(use_sky=False)
    world.add_object(SceneObject(
        shapes=[Shape(mesh, {"mat": material})],
    ))
    world.add_camera(Camera(
        position=Point3(0, 0, 0),
        forward=Vec3(0, 0, -1),
        width=W,
        height=H,
    ))
    return world


def test_triangle_array_intersector_returns_hitrecord_with_material_and_uv():
    world = _mesh_world()
    scene = flatten_world_triangles(world)
    intersector = TriangleArrayIntersector(scene)

    hit = intersector.intersect(Ray(Point3(0, 0, 0), Vec3(0, 0, -1)))

    assert hit is not None
    assert approx_eq(hit.t, 2.0)
    assert hit.material is scene.materials[0]
    assert hit.uv is not None
    assert approx_eq(hit.uv.x, 0.5)
    assert approx_eq(hit.uv.y, 0.5)


def test_triangle_array_intersector_occluded_matches_hit():
    world = _mesh_world()
    intersector = TriangleArrayIntersector(flatten_world_triangles(world))

    ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))

    assert intersector.occluded(ray, 3.0)
    assert not intersector.occluded(ray, 1.0)


def test_triangle_array_intersector_raw_arrays_match_scalar_hits():
    world = _mesh_world()
    intersector = TriangleArrayIntersector(flatten_world_triangles(world))
    origins = np.asarray([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
    directions = np.asarray([[0, 0, -1], [0, 0, -1]], dtype=np.float32)

    raw = intersector.intersect_raw_arrays(origins, directions)
    blocked = intersector.occluded_raw_arrays(origins, directions, [3.0, 3.0])

    assert raw["hit"].tolist() == [True, False]
    assert approx_eq(raw["t"][0], 2.0)
    assert raw["tri_id"].tolist() == [0, -1]
    assert blocked.tolist() == [True, False]


def test_world_intersector_wraps_existing_world_queries():
    world = _mesh_world()
    intersector = WorldIntersector(world)

    hit = intersector.intersect(Ray(Point3(0, 0, 0), Vec3(0, 0, -1)))

    assert hit is not None
    assert approx_eq(hit.t, 2.0)
    assert intersector.occluded(Ray(Point3(0, 0, 0), Vec3(0, 0, -1)), 3.0)


def test_ray_tracer_accepts_custom_intersector():
    world = _mesh_world(Emissive(Color(0.25, 0.5, 0.75), intensity=1.0))
    image = Image(W, H)
    intersector = TriangleArrayIntersector(flatten_world_triangles(world))
    tracer = RayTracer(
        world,
        image,
        None,
        samples=1,
        max_depth=1,
        threaded=False,
        intersector=intersector,
    )

    tracer.render()

    assert tracer.last_ray_count == W * H
    assert image.pixels.sum() > 0.0


def test_embree_intersector_reports_missing_binding_cleanly():
    world = _mesh_world()

    try:
        EmbreeIntersector(
            world,
            binding_modules=(
                ("missing_embree_scene_module", "missing_embree_mesh_module"),
            ),
        )
    except EmbreeUnavailableError as exc:
        assert "No compatible Embree Python binding" in str(exc)
    else:
        raise AssertionError("Expected EmbreeUnavailableError")


def test_embree_intersector_maps_raw_results_to_hit_records():
    _install_fake_embree_modules()
    world = _mesh_world()
    intersector = EmbreeIntersector(
        world,
        binding_modules=(
            ("fake_embree.rtcore_scene", "fake_embree.mesh_construction"),
        ),
    )
    origins = np.asarray([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
    directions = np.asarray([[0, 0, -1], [0, 0, -1]], dtype=np.float32)

    raw = intersector.intersect_raw_arrays(origins, directions)
    hits = intersector.intersect_many([
        Ray(Point3(0, 0, 0), Vec3(0, 0, -1)),
        Ray(Point3(3, 0, 0), Vec3(0, 0, -1)),
    ])
    blocked = intersector.occluded_raw_arrays(origins, directions, [3.0, 3.0])

    assert raw["hit"].tolist() == [True, False]
    assert approx_eq(raw["t"][0], 2.0)
    assert raw["tri_id"].tolist() == [0, -1]
    assert hits[0] is not None
    assert hits[1] is None
    assert approx_eq(hits[0].uv.x, 0.375)
    assert approx_eq(hits[0].uv.y, 0.25)
    assert blocked.tolist() == [True, False]


if __name__ == "__main__":
    run_tests([
        test_triangle_array_intersector_returns_hitrecord_with_material_and_uv,
        test_triangle_array_intersector_occluded_matches_hit,
        test_triangle_array_intersector_raw_arrays_match_scalar_hits,
        test_world_intersector_wraps_existing_world_queries,
        test_ray_tracer_accepts_custom_intersector,
        test_embree_intersector_reports_missing_binding_cleanly,
        test_embree_intersector_maps_raw_results_to_hit_records,
    ])
