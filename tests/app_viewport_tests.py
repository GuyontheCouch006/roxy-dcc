import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets
from PySide6 import QtGui

from app.main_window import RoxyMainWindow
from app.viewport import QtGLViewport
from core import Color, Vec3
from rendering.gl_viewport import (
    _object_gizmo_axes,
    _object_gizmo_origin,
    _object_gizmo_size,
)
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
    assert window.viewport.session is window.session
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
    assert window.viewport.selected_objects == (root,)
    assert window.viewport.highlighted_objects == (root, child)


def test_main_window_viewport_selection_updates_scene_graph():
    _ensure_qapp()
    world, root = _single_object_world()
    window = RoxyMainWindow(world)

    window.viewport.set_selected_object(root, emit=True)
    QtWidgets.QApplication.processEvents()

    assert window.scene_graph.selected_payload() is root


def test_viewport_pick_toggle_updates_shared_session_selection():
    _ensure_qapp()
    world, left, right = _two_object_world()
    window = RoxyMainWindow(world)
    window.viewport.resize(800, 600)

    _pick_object_center(window.viewport, left)
    _pick_object_center(window.viewport, right, toggle=True)
    _pick_object_center(window.viewport, left, toggle=True)
    QtWidgets.QApplication.processEvents()

    assert window.session.selected_scene_objects() == (right,)
    assert window.session.active_scene_object() is right
    assert window.scene_graph.selected_payload() is right


def test_move_gizmo_drag_uses_session_and_records_one_undo_item():
    _ensure_qapp()
    world, root = _single_object_world()
    window = RoxyMainWindow(world)
    viewport = window.viewport
    viewport.resize(800, 600)
    window.session.replace_selection(root)
    viewport.set_gizmo_mode("move")
    undo_before = window.session.undo_count

    origin = _object_gizmo_origin(root)
    size = _object_gizmo_size(root)
    start = viewport.camera.project_point(origin + [size * 0.5, 0, 0], 800, 600)[:2]
    end = viewport.camera.project_point(origin + [size, 0, 0], 800, 600)[:2]

    assert viewport.begin_transform_gizmo_drag(float(start[0]), float(start[1]))
    assert viewport.drag_transform_gizmo(float(end[0]), float(end[1]))
    viewport.end_transform_gizmo_drag()

    assert root.world_matrix.transform_point(Vec3(0, 0, 0)).x > 0.0
    assert window.session.undo_count == undo_before + 1
    window.session.undo()
    assert root.world_matrix.transform_point(Vec3(0, 0, 0)) == Vec3(0, 0, 0)


def test_move_gizmo_drag_defaults_to_active_object_local_axis():
    _ensure_qapp()
    world = World(use_sky=False)
    root = SceneObject(
        shape=Sphere(1.0),
        material=Diffuse(Color(0.8, 0.2, 0.2)),
        rotation=Vec3(0, 0, 90),
        name="rotated",
    )
    world.add_object(root)
    world.add_camera(Camera(name="camera1"))
    window = RoxyMainWindow(world)
    viewport = window.viewport
    viewport.resize(800, 600)
    window.session.replace_selection(root)
    viewport.set_gizmo_mode("move")

    origin = _object_gizmo_origin(root)
    size = _object_gizmo_size(root)
    axis = _object_gizmo_axes(root)["x"]
    start = viewport.camera.project_point(origin + axis * size * 0.5, 800, 600)[:2]
    end = viewport.camera.project_point(origin + axis * size, 800, 600)[:2]

    assert viewport.begin_transform_gizmo_drag(float(start[0]), float(start[1]))
    assert viewport.drag_transform_gizmo(float(end[0]), float(end[1]))
    viewport.end_transform_gizmo_drag()

    moved = root.world_matrix.transform_point(Vec3(0, 0, 0))
    assert moved.y > 0.0
    assert abs(moved.x) < 1e-5


def test_viewport_hotkeys_switch_gizmo_modes():
    _ensure_qapp()
    world, _root = _single_object_world()
    viewport = QtGLViewport(world)

    for key, mode in (
        (QtCore.Qt.Key.Key_W, "move"),
        (QtCore.Qt.Key.Key_E, "rotate"),
        (QtCore.Qt.Key.Key_R, "scale"),
        (QtCore.Qt.Key.Key_Q, "select"),
    ):
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            key,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        viewport.keyPressEvent(event)
        assert viewport.gizmo_mode == mode


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


def _two_object_world():
    world = World(use_sky=False)
    left = SceneObject(
        shape=Sphere(0.45),
        material=Diffuse(Color(0.8, 0.2, 0.2)),
        translation=Vec3(-1, 0, 0),
        name="left",
    )
    right = SceneObject(
        shape=Sphere(0.45),
        material=Diffuse(Color(0.2, 0.4, 0.8)),
        translation=Vec3(1, 0, 0),
        name="right",
    )
    world.add_object(left)
    world.add_object(right)
    world.add_camera(Camera(name="camera1"))
    return world, left, right


def _pick_object_center(viewport, scene_object, toggle=False):
    point = scene_object.world_matrix.transform_point(Vec3(0, 0, 0))
    screen = viewport.camera.project_point(point, viewport.width(), viewport.height())
    return viewport.pick_object(float(screen[0]), float(screen[1]), toggle=toggle)


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
        test_viewport_pick_toggle_updates_shared_session_selection,
        test_move_gizmo_drag_uses_session_and_records_one_undo_item,
        test_move_gizmo_drag_defaults_to_active_object_local_axis,
        test_viewport_hotkeys_switch_gizmo_modes,
        test_main_window_visibility_change_refreshes_viewport_buffers,
    ])
