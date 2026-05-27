import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from app.main_window import RoxyMainWindow, versioned_scene_path
from app.scripts import scene_commands
from core import Vec3
from scene import Emissive, Sphere, World
from tests.utils import run_tests


def test_main_window_builds_file_and_create_menus():
    _ensure_qapp()
    window = RoxyMainWindow(World(use_sky=False))

    menu_titles = [action.text().replace("&", "") for action in window.menuBar().actions()]

    assert "File" in menu_titles
    assert "Create" in menu_titles
    assert window.findChild(QtGui.QAction, "fileOpenAction") is not None
    assert window.findChild(QtGui.QAction, "fileSaveAsAction") is not None
    assert window.findChild(QtGui.QAction, "fileVersionUpAction") is not None
    assert window.findChild(QtGui.QAction, "fileReferenceAction") is not None
    assert window.findChild(QtGui.QAction, "fileImportAction") is not None
    assert window.findChild(QtGui.QAction, "fileExportAction") is not None
    assert window.findChild(QtGui.QAction, "createSphereAction") is not None
    assert window.findChild(QtGui.QAction, "createAreaLightAction") is not None


def test_menu_script_create_primitive_uses_session_and_is_undoable():
    _ensure_qapp()
    window = RoxyMainWindow(World(use_sky=False))

    handle = scene_commands.create_primitive(
        window.session,
        "sphere",
        name="menuSphere",
        radius=2.5,
        x=1.0,
    )
    QtWidgets.QApplication.processEvents()

    assert handle.raw in window.session.world.objects
    assert handle.raw.name == "menuSphere"
    assert isinstance(handle.raw.shape, Sphere)
    assert handle.raw.translation == Vec3(1, 0, 0)
    assert window.session.active_scene_object() is handle.raw
    assert window.viewport.selected_object is handle.raw

    assert window.undo()
    QtWidgets.QApplication.processEvents()

    assert handle.raw not in window.session.world.objects
    assert window.viewport.selected_object is None


def test_menu_script_create_light_adds_emissive_object():
    _ensure_qapp()
    window = RoxyMainWindow(World(use_sky=False))

    handle = scene_commands.create_light(
        window.session,
        "sphere",
        name="keyLight",
        intensity=40.0,
    )

    assert handle.raw.name == "keyLight"
    assert isinstance(handle.raw.material, Emissive)
    assert handle.raw in window.session.world.objects


def test_versioned_scene_path_increments_maya_style_suffix(tmp_path):
    base = tmp_path / "shot_v003.rxa"
    next_path = versioned_scene_path(base)
    assert next_path == tmp_path / "shot_v004.rxa"

    plain = tmp_path / "shot.rxa"
    assert versioned_scene_path(plain) == tmp_path / "shot_v001.rxa"

    (tmp_path / "shot_v001.rxa").write_text("", encoding="utf-8")
    assert versioned_scene_path(plain) == tmp_path / "shot_v002.rxa"


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


if __name__ == "__main__":
    def _versioned_path_smoke():
        with tempfile.TemporaryDirectory() as tmp:
            test_versioned_scene_path_increments_maya_style_suffix(Path(tmp))

    run_tests([
        test_main_window_builds_file_and_create_menus,
        test_menu_script_create_primitive_uses_session_and_is_undoable,
        test_menu_script_create_light_adds_emissive_object,
        _versioned_path_smoke,
    ])
