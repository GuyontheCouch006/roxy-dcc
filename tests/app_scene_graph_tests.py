import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from app.scene_graph import SceneGraphModel, SceneGraphRoles
from core import Color, Vec3
from scene import Camera, Diffuse, SceneObject, Sphere, World
from tests.utils import run_tests


def test_scene_graph_model_uses_world_as_hidden_root():
    world, _root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)

    hidden_root = model.node_from_index(QtCore.QModelIndex())

    assert hidden_root.name == "World"
    assert hidden_root.payload is world
    assert model.columnCount() == 1
    assert model.rowCount(QtCore.QModelIndex()) == 2
    assert model.index(0, 0, QtCore.QModelIndex()).data() == "assembly"
    assert model.index(1, 0, QtCore.QModelIndex()).data() == "shotCam"


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
    assert "active" in camera_index.data(QtCore.Qt.ItemDataRole.ToolTipRole)


def test_scene_graph_model_provides_outliner_icons():
    _ensure_qapp()
    world, root, _child = _world_with_hierarchy()
    model = SceneGraphModel(world)
    root_index = model.index_for_payload(root)

    icon = root_index.data(QtCore.Qt.ItemDataRole.DecorationRole)

    assert isinstance(icon, QtGui.QIcon)
    assert not icon.isNull()


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


def test_scene_graph_model_selection_includes_child_objects():
    world, root, child = _world_with_hierarchy()
    model = SceneGraphModel(world)
    root_index = model.index_for_payload(root)
    child_index = model.index_for_payload(child)
    camera_index = model.index_for_payload(world.active_camera)

    assert model.scene_objects_for_index(root_index) == (root, child)
    assert model.scene_objects_for_index(child_index) == (child,)
    assert model.scene_objects_for_index(camera_index) == ()


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


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


if __name__ == "__main__":
    run_tests([
        test_scene_graph_model_uses_world_as_hidden_root,
        test_scene_graph_model_preserves_object_child_and_shape_relationships,
        test_scene_graph_model_marks_active_camera,
        test_scene_graph_model_provides_outliner_icons,
        test_scene_graph_model_visibility_checkbox_updates_world_object,
        test_scene_graph_model_iter_nodes_can_filter_by_kind,
        test_scene_graph_model_selection_includes_child_objects,
    ])
