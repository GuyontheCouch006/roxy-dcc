from core import Color, Vec2, Vec3
from rendering.scene_arrays import flatten_world_triangles
from scene import Diffuse, SceneObject, World
from scene.mesh import IndexedMesh, Mesh, Triangle
from scene.shape import Shape
from tests.utils import approx_eq, run_tests


def test_flatten_world_triangles_preserves_indexed_mesh_sidecars():
    material = Diffuse(Color(0.2, 0.4, 0.6))
    mesh = IndexedMesh(
        positions=[[0, 0, -1], [1, 0, -1], [0, 1, -1]],
        tri_pos_idx=[[0, 1, 2]],
        normals=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        tri_normal_idx=[[0, 1, 2]],
        uvs=[[0, 0], [1, 0], [0, 1]],
        tri_uv_idx=[[0, 1, 2]],
        groups=["paint"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    world = World(use_sky=False)
    world.add_object(SceneObject(
        shapes=[Shape(mesh, {"paint": material}, name="indexed")],
        name="root",
    ))

    arrays = flatten_world_triangles(world)

    assert arrays.triangle_count == 1
    assert arrays.vertex_count == 3
    assert arrays.material_for_prim(0) is material
    assert arrays.has_uv[0]
    assert approx_eq(arrays.uv_at(0, 1.0, 0.0).x, 1.0)
    assert arrays.sources[0].object_path == "root"
    assert arrays.sources[0].shape_name == "indexed"
    assert arrays.sources[0].group == "paint"


def test_flatten_world_triangles_preserves_legacy_mesh_sidecars():
    material = Diffuse(Color(0.6, 0.4, 0.2))
    mesh = Mesh([
        Triangle(
            Vec3(0, 0, -1),
            Vec3(1, 0, -1),
            Vec3(0, 1, -1),
            uv0=Vec2(0, 0),
            uv1=Vec2(1, 0),
            uv2=Vec2(0, 1),
            group="legacy",
        )
    ])
    world = World(use_sky=False)
    world.add_object(SceneObject(
        shapes=[Shape(mesh, {"legacy": material}, name="mesh")],
    ))

    arrays = flatten_world_triangles(world)

    assert arrays.triangle_count == 1
    assert arrays.material_for_prim(0) is material
    assert arrays.has_uv[0]
    assert approx_eq(arrays.uv_at(0, 0.0, 1.0).y, 1.0)


def test_flatten_world_triangles_counts_skipped_primitives():
    from scene import Sphere

    world = World(use_sky=False)
    world.add_object(SceneObject(
        shape=Sphere(),
        material=Diffuse(Color(1, 1, 1)),
    ))

    arrays = flatten_world_triangles(world)

    assert arrays.triangle_count == 0
    assert arrays.skipped_primitives == 1


if __name__ == "__main__":
    run_tests([
        test_flatten_world_triangles_preserves_indexed_mesh_sidecars,
        test_flatten_world_triangles_preserves_legacy_mesh_sidecars,
        test_flatten_world_triangles_counts_skipped_primitives,
    ])
