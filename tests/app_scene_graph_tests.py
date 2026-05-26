from PySide6 import QtCore

from app.scene_graph import SceneGraphModel, SceneGraphRoles
from core import Color, Vec3
from scene import Camera, Diffuse, SceneObject, Sphere, World
from tests.utils import run_tests


def test_scene_graph_model_exposes_world_sections():
    world, _root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)

    world_index = model.index(0, 0, QtCore.QModelIndex())

    assert world_index.isValid()
    assert world_index.data() == "World"
    assert model.rowCount(world_index) == 3
    assert model.index(0, 0, world_index).data() == "Objects"
    assert model.index(1, 0, world_index).data() == "Cameras"
    assert model.index(2, 0, world_index).data() == "Lights"
    assert world_index.siblingAtColumn(2).data() == "1 objects, 1 cameras, 0 lights"


def test_scene_graph_model_preserves_object_child_and_shape_relationships():
    world, root, child = _world_with_hierarchy()
    model = SceneGraphModel(world)

    root_index = model.index_for_payload(root)
    child_index = model.index_for_payload(child)

    assert root_index.isValid()
    assert child_index.isValid()
    assert child_index.parent().data(SceneGraphRoles.PayloadRole) is root

    root_child_kinds = [
        model.index(row, 0, root_index).data(SceneGraphRoles.KindRole)
        for row in range(model.rowCount(root_index))
    ]
    child_kinds = [
        model.index(row, 0, child_index).data(SceneGraphRoles.KindRole)
        for row in range(model.rowCount(child_index))
    ]

    assert "object" in root_child_kinds
    assert "shape" in child_kinds


def test_scene_graph_model_marks_active_camera():
    world, _root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)
    camera_index = model.index_for_payload(world.active_camera)

    assert camera_index.isValid()
    assert camera_index.data() == "shotCam"
    assert "active" in camera_index.siblingAtColumn(2).data()


def test_scene_graph_model_visibility_checkbox_updates_world_object():
    world, root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)
    root_index = model.index_for_payload(root)

    assert root_index.data(QtCore.Qt.ItemDataRole.CheckStateRole) == QtCore.Qt.CheckState.Checked

    changed = model.setData(
        root_index,
        QtCore.Qt.CheckState.Unchecked,
        QtCore.Qt.ItemDataRole.CheckStateRole,
    )

    assert changed
    assert root.visible is False
    assert root_index.data(QtCore.Qt.ItemDataRole.CheckStateRole) == QtCore.Qt.CheckState.Unchecked


def test_scene_graph_model_iter_nodes_can_filter_by_kind():
    world, _root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)

    object_names = [node.name for node in model.iter_nodes(kind="object")]
    shape_names = [node.name for node in model.iter_nodes(kind="shape")]

    assert object_names == ["assembly", "wheel"]
    assert shape_names == ["wheel"]


def _world_with_hierarchy():
    world = World(use_sky=False)
    root = SceneObject(name="assembly")
    child = SceneObject(
        shape=Sphere(1.0),
        material=Diffuse(Color(0.2, 0.4, 0.8)),
        translation=Vec3(1, 0, 0),
        name="wheel",
    )
    root.add_child(child)
    world.add_object(root)
    world.add_camera(Camera(name="shotCam"))
    return world, root, child


if __name__ == "__main__":
    run_tests([
        test_scene_graph_model_exposes_world_sections,
        test_scene_graph_model_preserves_object_child_and_shape_relationships,
        test_scene_graph_model_marks_active_camera,
        test_scene_graph_model_visibility_checkbox_updates_world_object,
        test_scene_graph_model_iter_nodes_can_filter_by_kind,
    ])
