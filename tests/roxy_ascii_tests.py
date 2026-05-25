import os
import tempfile

from core import Color, Mat4x4, Vec3
from scene import Camera, Glossy, SceneObject, Sphere, World
from scene.io import load_scene, save_scene
from scene.mesh import IndexedMesh
from scene.io.roxy_ascii import (
    dumps_rxa,
    parse_rxa,
    rxa_scene_for_obj,
    rxa_scene_to_world,
    world_to_rxa_scene,
)
from tests.utils import approx_eq, run_tests, vec3_approx_eq


def _identity_values():
    return "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"


def _world():
    world = World(use_sky=False, background_color=Color(0.1, 0.2, 0.3))
    root = SceneObject(name="root", matrix=Mat4x4.translation(1, 2, 3))
    child = SceneObject(
        name="child",
        shape=Sphere(2.0),
        material=Glossy(Color(0.4, 0.5, 0.6), roughness=0.25),
        matrix=Mat4x4.translation(0, 1, 0),
    )
    root.add_child(child)
    world.add_object(root)
    world.add_camera(Camera(name="camera1", position=Vec3(0, 0, 5), forward=Vec3(0, 0, -1)))
    return world


def _write_obj(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_parse_create_set_and_connect_commands():
    scene = parse_rxa(f"""
        createNode transform -n "root";
        setAttr "root.matrix" -type matrix {_identity_values()};
        createNode sphereShape -n "rootShape" -p "root";
        setAttr "rootShape.radius" -type float 2.5;
        createNode diffuse -n "mat";
        setAttr "mat.albedo" -type color3 0.1 0.2 0.3;
        connectAttr "mat.outSurface" "rootShape.surfaceShader";
    """)

    assert scene.nodes["root"].type_name == "transform"
    assert scene.nodes["rootShape"].parent == "root"
    assert scene.nodes["rootShape"].attr("radius") == 2.5
    assert scene.nodes["mat"].attr("albedo") == [0.1, 0.2, 0.3]
    assert scene.connections[0].source == "mat.outSurface"


def test_dumps_round_trips_rxa_scene_commands():
    scene = parse_rxa(f"""
        createNode transform -n "root";
        setAttr "root.matrix" -type matrix {_identity_values()};
        createNode sphereShape -n "rootShape" -p "root";
        setAttr "rootShape.radius" -type float 1.0;
    """)

    restored = parse_rxa(dumps_rxa(scene))

    assert restored.nodes["root"].attr("matrix") == scene.nodes["root"].attr("matrix")
    assert restored.nodes["rootShape"].parent == "root"


def test_world_translates_to_rxa_nodes_connections_and_back():
    restored = rxa_scene_to_world(world_to_rxa_scene(_world()))

    root = restored.objects[0]
    child = root.children[0]
    material = child.shapes[0].material_for_group("default")

    assert root.name == "root"
    assert child.name == "child"
    assert child.matrix_mode
    assert vec3_approx_eq(child.local_matrix.transform_point(Vec3(0, 0, 0)), Vec3(0, 1, 0))
    assert isinstance(child.shape, Sphere)
    assert approx_eq(child.shape._radius, 2.0)
    assert isinstance(material, Glossy)
    assert approx_eq(material._roughness, 0.25)
    assert vec3_approx_eq(material._albedo, Color(0.4, 0.5, 0.6))
    assert restored.use_sky is False
    assert vec3_approx_eq(restored.background_color, Color(0.1, 0.2, 0.3))
    assert restored.active_camera.name == "camera1"


def test_load_save_scene_dispatches_rxa_extension():
    fd, path = tempfile.mkstemp(suffix=".rxa")
    os.close(fd)
    try:
        save_scene(_world(), path)
        restored = load_scene(path)

        assert restored.objects[0].children[0].name == "child"
        assert isinstance(restored.objects[0].children[0].shape, Sphere)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_obj_reference_rxa_scene_imports_external_obj_hierarchy():
    obj_path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        scene = rxa_scene_for_obj(obj_path, name="asset")
        world = rxa_scene_to_world(scene)

        assert world.objects[0].name == "asset"
        assert len(world.objects[0].children) == 1
        assert isinstance(world.objects[0].children[0].shape, IndexedMesh)
    finally:
        try:
            os.unlink(obj_path)
        except OSError:
            pass


if __name__ == "__main__":
    run_tests([
        test_parse_create_set_and_connect_commands,
        test_dumps_round_trips_rxa_scene_commands,
        test_world_translates_to_rxa_nodes_connections_and_back,
        test_load_save_scene_dispatches_rxa_extension,
        test_obj_reference_rxa_scene_imports_external_obj_hierarchy,
    ])
