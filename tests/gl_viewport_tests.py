import numpy as np

from core import Color, Vec3
from rendering.gl_viewport import (
    ViewportCamera,
    _build_gizmo_vertices,
    build_scene_viewport_buffers,
    pick_scene_object,
)
from scene import Diffuse, IndexedMesh, SceneObject, Shape, World
from tests.utils import approx_eq, run_tests, vec3_approx_eq


def test_viewport_camera_orbit_changes_eye_but_keeps_target_and_distance():
    camera = ViewportCamera(target=(0, 0, 0), distance=10)
    start_eye = camera.eye.copy()
    start_target = camera.target.copy()

    camera.orbit(20, -10)

    assert not np.allclose(camera.eye, start_eye)
    assert np.allclose(camera.target, start_target)
    assert approx_eq(float(np.linalg.norm(camera.eye - camera.target)), 10.0)


def test_viewport_camera_pan_moves_target_in_view_plane():
    camera = ViewportCamera(target=(0, 0, 0), distance=10)

    camera.pan(100, 50, width=800, height=400)

    assert not np.allclose(camera.target, np.zeros(3))
    assert abs(float(np.dot(camera.target, camera.forward))) < 1e-5


def test_viewport_camera_screen_center_ray_points_forward():
    camera = ViewportCamera(target=(0, 0, 0), distance=10, yaw=0, pitch=0)

    ray = camera.screen_ray(400, 300, 800, 600)

    assert vec3_approx_eq(ray.direction, Vec3(0, 0, -1))


def test_viewport_camera_frame_bounds_centers_camera_on_scene():
    world = _single_triangle_world()
    buffers = build_scene_viewport_buffers(world)
    camera = ViewportCamera(distance=1)

    camera.frame_bounds(buffers.bounds)

    assert np.allclose(camera.target, np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert camera.distance > 1.0
    assert camera.far > camera.near


def test_scene_viewport_buffers_extract_indexed_mesh_vertices_and_color():
    world = _single_triangle_world(translation=Vec3(2, 0, 0))
    obj = world.objects[0]

    buffers = build_scene_viewport_buffers(world)

    assert buffers.triangle_count == 1
    assert buffers.shape_count == 1
    assert buffers.vertices.shape == (3, 3)
    assert len(buffers.object_spans) == 1
    assert buffers.object_spans[0].scene_object is obj
    assert buffers.object_spans[0].count == 3
    assert np.allclose(buffers.vertices[0], np.array([2, 0, 0], dtype=np.float32))
    assert np.allclose(buffers.colors[0], np.array([1, 0, 0], dtype=np.float32))


def test_scene_viewport_buffers_resolve_group_materials_per_triangle():
    mesh = IndexedMesh(
        positions=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        tri_pos_idx=[
            [0, 1, 2],
            [1, 3, 2],
        ],
        groups=["red", "green"],
        tri_group_idx=[0, 1],
        build_bvh=False,
    )
    shape = Shape(
        mesh,
        {
            "red": Diffuse(Color(1, 0, 0)),
            "green": Diffuse(Color(0, 1, 0)),
        },
    )
    world = World(objects=[SceneObject(shapes=[shape])], use_sky=False)

    buffers = build_scene_viewport_buffers(world)

    assert buffers.triangle_count == 2
    assert np.allclose(buffers.colors[:3], np.array([[1, 0, 0]] * 3, dtype=np.float32))
    assert np.allclose(buffers.colors[3:6], np.array([[0, 1, 0]] * 3, dtype=np.float32))


def test_pick_scene_object_returns_nearest_object_under_cursor():
    near_obj = _pickable_triangle_object("near", z=0.0)
    far_obj = _pickable_triangle_object("far", z=-2.0)
    world = World(objects=[far_obj, near_obj], use_sky=False)
    camera = ViewportCamera(target=(0, 0, 0), distance=5, yaw=0, pitch=0)

    result = pick_scene_object(world, camera, 50, 50, 100, 100)

    assert result is not None
    assert result.scene_object is near_obj


def test_pick_scene_object_skips_non_selectable_objects():
    obj = _pickable_triangle_object("locked", z=0.0)
    obj.selectable = False
    world = World(objects=[obj], use_sky=False)
    camera = ViewportCamera(target=(0, 0, 0), distance=5, yaw=0, pitch=0)

    result = pick_scene_object(world, camera, 50, 50, 100, 100)

    assert result is None


def test_build_gizmo_vertices_returns_mode_specific_line_vertices():
    origin = np.array([0, 0, 0], dtype=np.float32)

    move = _build_gizmo_vertices(origin, 1.0, "move")
    rotate = _build_gizmo_vertices(origin, 1.0, "rotate")
    scale = _build_gizmo_vertices(origin, 1.0, "scale")

    assert move.shape[1] == 6
    assert rotate.shape[1] == 6
    assert scale.shape[1] == 6
    assert len(rotate) > len(move)
    assert len(scale) > len(move)


def _single_triangle_world(translation=None):
    mesh = IndexedMesh(
        positions=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        tri_pos_idx=[[0, 1, 2]],
        groups=["default"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    obj = SceneObject(
        shape=mesh,
        material=Diffuse(Color(1, 0, 0)),
        translation=translation,
    )
    return World(objects=[obj], use_sky=False)


def _pickable_triangle_object(name, z):
    mesh = IndexedMesh(
        positions=[
            [-1, -1, z],
            [1, -1, z],
            [0, 1, z],
        ],
        tri_pos_idx=[[0, 1, 2]],
        groups=["default"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    return SceneObject(
        shape=mesh,
        material=Diffuse(Color(1, 0, 0)),
        name=name,
    )


if __name__ == "__main__":
    run_tests([
        test_viewport_camera_orbit_changes_eye_but_keeps_target_and_distance,
        test_viewport_camera_pan_moves_target_in_view_plane,
        test_viewport_camera_screen_center_ray_points_forward,
        test_viewport_camera_frame_bounds_centers_camera_on_scene,
        test_scene_viewport_buffers_extract_indexed_mesh_vertices_and_color,
        test_scene_viewport_buffers_resolve_group_materials_per_triangle,
        test_pick_scene_object_returns_nearest_object_under_cursor,
        test_pick_scene_object_skips_non_selectable_objects,
        test_build_gizmo_vertices_returns_mode_specific_line_vertices,
    ])
