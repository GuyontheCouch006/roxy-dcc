import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.main_window import RoxyMainWindow
from app.node_network import NodeNetworkPanel, SceneNodeGraphModel
from core import Color
from scene import Diffuse, SceneObject, Sphere, World
from tests.utils import run_tests, vec3_approx_eq


def test_main_window_hosts_details_dock_and_leaves_node_network_hidden():
    _ensure_qapp()
    window = RoxyMainWindow(_world_with_two_materials()[0])

    assert window.details.session is window.session
    assert window.findChild(QtWidgets.QDockWidget, "detailsDock") is not None
    assert window.findChild(QtWidgets.QDockWidget, "nodeNetworkDock") is None


def test_details_material_edit_goes_through_session_undo():
    _ensure_qapp()
    world, _left, _right, red, _blue = _world_with_two_materials()
    window = RoxyMainWindow(world)
    window.session.set_selected_payload(red)
    QtWidgets.QApplication.processEvents()

    spin = window.details.findChild(QtWidgets.QDoubleSpinBox, "attr_albedo_r")
    assert spin is not None

    spin.setValue(0.9)
    QtWidgets.QApplication.processEvents()

    assert red._albedo.r == 0.9
    assert window.session.undo()
    assert vec3_approx_eq(red._albedo, Color(0.8, 0.1, 0.1))


def test_node_graph_contains_history_and_shader_edges():
    world, left, _right, red, _blue = _world_with_two_materials()
    session = _session(world)
    model = SceneNodeGraphModel(session)
    kinds = {node.kind for node in model.nodes}
    edge_kinds = {edge.kind for edge in model.edges}

    assert {"object", "shape", "material", "history"}.issubset(kinds)
    assert {"ownership", "geometry", "shader"}.issubset(edge_kinds)
    assert model.node_for_payload(left) is not None
    assert model.node_for_payload(red) is not None


def test_node_network_scaffold_shader_rewire_uses_api_and_is_undoable():
    _ensure_qapp()
    world, left, _right, red, blue = _world_with_two_materials()
    session = _session(world)
    panel = NodeNetworkPanel(session=session)
    shape = left.shapes[0]

    panel.connect_shader(blue, shape)

    assert shape.material_for_group("default") is blue
    assert session.undo()
    assert shape.material_for_group("default") is red


def test_selecting_shape_in_outliner_updates_details_without_viewport_selection():
    _ensure_qapp()
    world, left, _right, _red, _blue = _world_with_two_materials()
    window = RoxyMainWindow(world)
    shape = left.shapes[0]
    shape_index = window.scene_graph.model.index_for_payload(shape)

    window.scene_graph.tree.setCurrentIndex(shape_index)
    QtWidgets.QApplication.processEvents()

    assert window.details.selected_payload is shape
    assert window.viewport.selected_object is None


def _world_with_two_materials():
    red = Diffuse(Color(0.8, 0.1, 0.1), name="redShader")
    blue = Diffuse(Color(0.1, 0.2, 0.8), name="blueShader")
    left = SceneObject(shape=Sphere(1.0), material=red, name="left")
    right = SceneObject(shape=Sphere(1.0), material=blue, name="right")
    world = World(objects=[left, right], use_sky=False)
    return world, left, right, red, blue


def _session(world):
    from scene import SceneSession
    return SceneSession(world)


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


if __name__ == "__main__":
    run_tests([
        test_main_window_hosts_details_dock_and_leaves_node_network_hidden,
        test_details_material_edit_goes_through_session_undo,
        test_node_graph_contains_history_and_shader_edges,
        test_node_network_scaffold_shader_rewire_uses_api_and_is_undoable,
        test_selecting_shape_in_outliner_updates_details_without_viewport_selection,
    ])
