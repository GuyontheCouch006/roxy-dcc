from core import Color, Point3, Ray, Vec2, Vec3
from rendering.intersector import TriangleArrayIntersector, WorldIntersector
from rendering.ray_tracer import RayTracer
from rendering.image import Image
from rendering.scene_arrays import flatten_world_triangles
from scene import Camera, Diffuse, SceneObject, World
from scene.materials import Emissive
from scene.mesh import IndexedMesh
from scene.shape import Shape
from tests.utils import approx_eq, run_tests


W = H = 4


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


if __name__ == "__main__":
    run_tests([
        test_triangle_array_intersector_returns_hitrecord_with_material_and_uv,
        test_triangle_array_intersector_occluded_matches_hit,
        test_world_intersector_wraps_existing_world_queries,
        test_ray_tracer_accepts_custom_intersector,
    ])
