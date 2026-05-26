import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.main_window import RoxyMainWindow
from app.viewport import QtGLViewport
from core import Color, Vec3
from scene import Camera, Diffuse, SceneObject, Sphere, World
from tests.utils import run_tests


def test_qt_viewport_builds_scene_buffers_before_gl_initializes():
    _ensure_qapp()
    world, root = _single_object_world()

    viewport = QtGLViewport(world)

    assert viewport.world is world
    assert viewport.scene_buffers is not None
    assert viewport.scene_buffers.triangle_count > 0
    assert viewport.scene_buffers.span_for(root) is not None


def test_qt_viewport_selection_signal_is_optional():
    _ensure_qapp()
    world, root = _single_object_world()
    viewport = QtGLViewport(world)
    selected = []
    viewport.objectSelected.connect(selected.append)

    viewport.set_selected_object(root)
    viewport.set_selected_object(None, emit=True)

    assert viewport.selected_object is None
    assert selected == [None]


def test_main_window_hosts_scene_graph_and_qt_viewport():
    _ensure_qapp()
    world, _root = _single_object_world()

    window = RoxyMainWindow(world)

    assert window.scene_graph.model.world is world
    assert window.viewport.world is world
    assert window.centralWidget() is window.viewport


def test_main_window_scene_graph_selection_updates_viewport():
    _ensure_qapp()
    world, root = _single_object_world()
    window = RoxyMainWindow(world)

    root_index = window.scene_graph.model.index_for_payload(root)
    window.scene_graph.tree.setCurrentIndex(root_index)
    QtWidgets.QApplication.processEvents()

    assert window.viewport.selected_object is root


def test_main_window_parent_selection_updates_viewport_children():
    _ensure_qapp()
    world, root, child = _object_hierarchy_world()
    window = RoxyMainWindow(world)

    root_index = window.scene_graph.model.index_for_payload(root)
    window.scene_graph.tree.setCurrentIndex(root_index)
    QtWidgets.QApplication.processEvents()

    assert window.viewport.selected_object is root
    assert window.viewport.selected_objects == (root, child)


def test_main_window_viewport_selection_updates_scene_graph():
    _ensure_qapp()
    world, root = _single_object_world()
    window = RoxyMainWindow(world)

    window.viewport.set_selected_object(root, emit=True)
    QtWidgets.QApplication.processEvents()

    assert window.scene_graph.selected_payload() is root


def test_main_window_visibility_change_refreshes_viewport_buffers():
    _ensure_qapp()
    world, root = _single_object_world()
    window = RoxyMainWindow(world)
    root_index = window.scene_graph.model.index_for_payload(root)

    window.scene_graph.model.setData(
        root_index,
        QtCore.Qt.CheckState.Unchecked,
        QtCore.Qt.ItemDataRole.CheckStateRole,
    )
    QtWidgets.QApplication.processEvents()

    assert root.visible is False
    assert window.viewport.scene_buffers.triangle_count == 0


def _single_object_world():
    world = World(use_sky=False)
    obj = SceneObject(
        shape=Sphere(1.0),
        material=Diffuse(Color(0.8, 0.2, 0.2)),
        translation=Vec3(0, 0, 0),
        name="sphere",
    )
    world.add_object(obj)
    world.add_camera(Camera(name="camera1"))
    return world, obj


def _object_hierarchy_world():
    world = World(use_sky=False)
    root = SceneObject(name="assembly")
    child = SceneObject(
        shape=Sphere(1.0),
        material=Diffuse(Color(0.8, 0.2, 0.2)),
        translation=Vec3(0, 0, 0),
        name="wheel",
    )
    root.add_child(child)
    world.add_object(root)
    world.add_camera(Camera(name="camera1"))
    return world, root, child


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


if __name__ == "__main__":
    run_tests([
        test_qt_viewport_builds_scene_buffers_before_gl_initializes,
        test_qt_viewport_selection_signal_is_optional,
        test_main_window_hosts_scene_graph_and_qt_viewport,
        test_main_window_scene_graph_selection_updates_viewport,
        test_main_window_parent_selection_updates_viewport_children,
        test_main_window_viewport_selection_updates_scene_graph,
        test_main_window_visibility_change_refreshes_viewport_buffers,
    ])
