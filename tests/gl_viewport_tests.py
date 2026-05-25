import numpy as np

from core import Color, Vec3
from rendering.gl_viewport import ViewportCamera, build_scene_viewport_buffers
from scene import Diffuse, IndexedMesh, SceneObject, Shape, World
from tests.utils import approx_eq, run_tests


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

    buffers = build_scene_viewport_buffers(world)

    assert buffers.triangle_count == 1
    assert buffers.shape_count == 1
    assert buffers.vertices.shape == (3, 3)
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


if __name__ == "__main__":
    run_tests([
        test_viewport_camera_orbit_changes_eye_but_keeps_target_and_distance,
        test_viewport_camera_pan_moves_target_in_view_plane,
        test_viewport_camera_frame_bounds_centers_camera_on_scene,
        test_scene_viewport_buffers_extract_indexed_mesh_vertices_and_color,
        test_scene_viewport_buffers_resolve_group_materials_per_triangle,
    ])
