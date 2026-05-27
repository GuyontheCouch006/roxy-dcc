from __future__ import annotations

import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6 import QtCore, QtGui, QtWidgets

from app.scene_graph import SceneGraphModel, SceneGraphRoles
from app.scripts import scene_commands
from app.viewport import QtGLViewport
from scene import SceneObject, SceneSession
from scene.io import load_scene, save_scene


SCENE_FILE_FILTER = "Roxy Scene (*.rxa);;JSON Scene (*.json);;All Files (*)"
IMPORT_FILE_FILTER = "Scene or Asset (*.rxa *.json *.obj);;All Files (*)"
MAX_RECENT_FILES = 10
SETTINGS_RECENT_FILES_KEY = "recentFiles"


class SceneGraphPanel(QtWidgets.QWidget):
    nodeSelected = QtCore.Signal(object)

    def __init__(self, world=None, parent=None, session=None):
        super().__init__(parent)
        self._session = None
        self._applying_session_selection = False
        self._model = SceneGraphModel(world, self)

        self._tree = QtWidgets.QTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.setIconSize(QtCore.QSize(18, 18))
        self._tree.expandToDepth(1)
        self._tree.header().setHidden(True)
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self._tree.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)
        if session is not None:
            self.set_session(session)

    @property
    def model(self):
        return self._model

    @property
    def tree(self):
        return self._tree

    def set_world(self, world):
        self._model.set_world(world)
        self._tree.expandToDepth(1)

    def set_session(self, session):
        if self._session is session:
            return
        if self._session is not None:
            self._session.remove_selection_listener(self._on_session_selection_changed)
            self._session.remove_world_listener(self._on_session_world_changed)
        self._session = session
        self._model.set_session(session)
        if session is not None:
            session.add_selection_listener(self._on_session_selection_changed)
            session.add_world_listener(self._on_session_world_changed)
            self._sync_selection_from_session()
        self._tree.expandToDepth(1)

    def select_payload(self, payload):
        index = self._model.index_for_payload(payload)
        if not index.isValid():
            if self._session is not None:
                self._session.clear_selection()
            else:
                self._clear_tree_selection()
            return
        if self._session is not None and isinstance(payload, SceneObject):
            self._session.replace_selection(payload)
        else:
            self._select_indexes((index,))

    def selected_payload(self):
        index = self._tree.currentIndex()
        return index.data(SceneGraphRoles.PayloadRole) if index.isValid() else None

    def selected_scene_objects(self):
        if self._session is not None:
            return self._session.selected_scene_objects()
        indexes = self._selected_object_indexes()
        return tuple(self._model.scene_object_for_index(index) for index in indexes)

    def _on_selection_changed(self, selected, deselected):
        del selected, deselected
        if self._applying_session_selection:
            return
        scene_objects = [
            self._model.scene_object_for_index(index)
            for index in self._selected_object_indexes()
        ]
        scene_objects = [obj for obj in scene_objects if obj is not None]
        active = self._model.scene_object_for_index(self._tree.currentIndex())
        if active not in scene_objects:
            active = scene_objects[-1] if scene_objects else None
        if self._session is not None:
            self._session.set_selection(scene_objects, active=active)
        self.nodeSelected.emit(tuple(scene_objects))

    def _on_session_selection_changed(self, session):
        del session
        self._sync_selection_from_session()

    def _on_session_world_changed(self, session):
        self.set_world(session.world)
        self._sync_selection_from_session()

    def _sync_selection_from_session(self):
        if self._session is None:
            return
        indexes = [
            self._model.index_for_payload(obj)
            for obj in self._session.selected_scene_objects()
        ]
        self._select_indexes(tuple(index for index in indexes if index.isValid()))
        active = self._session.active_scene_object()
        if active is not None:
            active_index = self._model.index_for_payload(active)
            if active_index.isValid():
                self._tree.setCurrentIndex(active_index)

    def _selected_object_indexes(self):
        indexes = self._tree.selectionModel().selectedRows(0)
        return tuple(
            index
            for index in indexes
            if self._model.scene_object_for_index(index) is not None
        )

    def _select_indexes(self, indexes):
        self._applying_session_selection = True
        try:
            selection_model = self._tree.selectionModel()
            blocker = QtCore.QSignalBlocker(selection_model)
            try:
                selection_model.clearSelection()
                for index in indexes:
                    selection_model.select(
                        index,
                        QtCore.QItemSelectionModel.SelectionFlag.Select
                        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )
                if indexes:
                    self._tree.setCurrentIndex(indexes[-1])
            finally:
                del blocker
        finally:
            self._applying_session_selection = False

    def _clear_tree_selection(self):
        self._select_indexes(())


class CommandParameterDialog(QtWidgets.QDialog):
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._widgets = {}

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        for field in fields:
            key = field["key"]
            label = field.get("label", key)
            if field.get("type") == "text":
                widget = QtWidgets.QLineEdit(str(field.get("default", "")))
            else:
                widget = QtWidgets.QDoubleSpinBox()
                widget.setDecimals(int(field.get("decimals", 3)))
                widget.setRange(
                    float(field.get("minimum", -1_000_000.0)),
                    float(field.get("maximum", 1_000_000.0)),
                )
                widget.setSingleStep(float(field.get("step", 0.1)))
                widget.setValue(float(field.get("default", 0.0)))
            self._widgets[key] = widget
            form.addRow(label, widget)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self):
        values = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                values[key] = widget.text().strip() or None
            else:
                values[key] = widget.value()
        return values


_VERSION_RE = re.compile(r"^(?P<base>.*)_v(?P<version>\d+)$", re.IGNORECASE)


def versioned_scene_path(path):
    path = Path(path)
    match = _VERSION_RE.match(path.stem)
    if match:
        base = match.group("base")
        version = int(match.group("version"))
        width = len(match.group("version"))
    else:
        base = path.stem
        version = 0
        width = 3

    while True:
        version += 1
        candidate = path.with_name(f"{base}_v{version:0{width}d}{path.suffix}")
        if not candidate.exists():
            return candidate


class RoxyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Roxy")
        self.resize(1280, 820)

        world = world or scene_commands.new_scene()
        self._session = SceneSession(world)
        self._current_path = None
        self._settings = QtCore.QSettings("Roxy", "Roxy")
        self._recent_files = self._load_recent_files()
        self._recent_files_menu = None
        self._scene_graph = SceneGraphPanel(parent=self, session=self._session)
        self._viewport = QtGLViewport(parent=self, session=self._session)
        self._build_menu_bar()
        self._update_window_title()

        scene_graph_dock = QtWidgets.QDockWidget("Scene Graph", self)
        scene_graph_dock.setObjectName("sceneGraphDock")
        scene_graph_dock.setWidget(self._scene_graph)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, scene_graph_dock)

        self.setCentralWidget(self._viewport)
        self._undo_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Undo,
            self,
        )
        self._undo_shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self._undo_shortcut.activated.connect(self.undo)

    @property
    def scene_graph(self):
        return self._scene_graph

    @property
    def viewport(self):
        return self._viewport

    @property
    def session(self):
        return self._session

    @property
    def current_path(self):
        return self._current_path

    def set_world(self, world):
        self._session.set_world(world)

    def undo(self):
        return self._session.undo()

    def new_scene(self):
        self.set_world(scene_commands.new_scene())
        self._current_path = None
        self._update_window_title()

    def open_scene(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Scene",
            "",
            SCENE_FILE_FILTER,
        )
        if path:
            self.open_scene_path(path)

    def open_scene_path(self, path):
        try:
            world = load_scene(path)
        except Exception as exc:
            self._show_error("Open Scene Failed", str(exc))
            return False
        self.set_world(world)
        self._current_path = Path(path)
        self._add_recent_file(self._current_path)
        self._update_window_title()
        return True

    def save_scene(self):
        if self._current_path is None:
            return self.save_scene_as()
        return self.save_scene_path(self._current_path)

    def save_scene_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scene As",
            str(self._current_path or Path.cwd() / "untitled.rxa"),
            SCENE_FILE_FILTER,
        )
        if not path:
            return False
        return self.save_scene_path(path)

    def save_scene_path(self, path):
        path = Path(path)
        try:
            save_scene(self._session.world, path)
        except Exception as exc:
            self._show_error("Save Scene Failed", str(exc))
            return False
        self._current_path = path
        self._add_recent_file(path)
        self._update_window_title()
        return True

    def version_up_scene(self):
        if self._current_path is None:
            return self.save_scene_as()
        return self.save_scene_path(versioned_scene_path(self._current_path))

    def import_scene(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Scene or Asset",
            "",
            IMPORT_FILE_FILTER,
        )
        if path:
            self._run_script_command(scene_commands.import_scene, self._session, path)

    def export_scene(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Scene",
            str(self._current_path or Path.cwd() / "untitled.rxa"),
            SCENE_FILE_FILTER,
        )
        if path:
            return self.save_scene_path(path)
        return False

    def create_primitive(self, primitive_type):
        fields = self._primitive_fields(primitive_type)
        values = self._command_values(f"Create {primitive_type.title()}", fields)
        if values is None:
            return None
        return self._run_script_command(
            scene_commands.create_primitive,
            self._session,
            primitive_type,
            **values,
        )

    def create_light(self, light_type):
        title = "Create Sphere Light" if light_type == "sphere" else "Create Area Light"
        fields = self._light_fields(light_type)
        values = self._command_values(title, fields)
        if values is None:
            return None
        return self._run_script_command(
            scene_commands.create_light,
            self._session,
            light_type,
            **values,
        )

    def _build_menu_bar(self):
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("&File")
        file_menu.setObjectName("fileMenu")
        self._add_action(
            file_menu,
            "New",
            self.new_scene,
            shortcut=QtGui.QKeySequence.StandardKey.New,
            object_name="fileNewAction",
        )
        self._add_action(
            file_menu,
            "Open...",
            self.open_scene,
            shortcut=QtGui.QKeySequence.StandardKey.Open,
            object_name="fileOpenAction",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Save",
            self.save_scene,
            shortcut=QtGui.QKeySequence.StandardKey.Save,
            object_name="fileSaveAction",
        )
        self._add_action(
            file_menu,
            "Save As...",
            self.save_scene_as,
            shortcut=QtGui.QKeySequence.StandardKey.SaveAs,
            object_name="fileSaveAsAction",
        )
        self._add_action(
            file_menu,
            "Version Up",
            self.version_up_scene,
            object_name="fileVersionUpAction",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Import...",
            self.import_scene,
            object_name="fileImportAction",
        )
        self._add_action(
            file_menu,
            "Export...",
            self.export_scene,
            object_name="fileExportAction",
        )
        file_menu.addSeparator()
        self._recent_files_menu = file_menu.addMenu("Recent Files")
        self._recent_files_menu.setObjectName("recentFilesMenu")
        self._update_recent_files_menu()

        create_menu = menu_bar.addMenu("&Create")
        create_menu.setObjectName("createMenu")
        primitive_menu = create_menu.addMenu("Primitives")
        primitive_menu.setObjectName("createPrimitivesMenu")
        self._add_action(
            primitive_menu,
            "Sphere...",
            lambda: self.create_primitive("sphere"),
            object_name="createSphereAction",
        )
        self._add_action(
            primitive_menu,
            "Cube...",
            lambda: self.create_primitive("cube"),
            object_name="createCubeAction",
        )
        self._add_action(
            primitive_menu,
            "Plane...",
            lambda: self.create_primitive("plane"),
            object_name="createPlaneAction",
        )
        torus_action = self._add_action(
            primitive_menu,
            "Torus...",
            lambda: self.create_primitive("torus"),
            object_name="createTorusAction",
        )
        torus_action.setToolTip(
            "Creates a viewport-only torus; ray intersection is not implemented yet."
        )

        light_menu = create_menu.addMenu("Lights")
        light_menu.setObjectName("createLightsMenu")
        self._add_action(
            light_menu,
            "Sphere Light...",
            lambda: self.create_light("sphere"),
            object_name="createSphereLightAction",
        )
        self._add_action(
            light_menu,
            "Area Light...",
            lambda: self.create_light("area"),
            object_name="createAreaLightAction",
        )

    def _add_action(self, menu, text, callback, shortcut=None, object_name=None):
        action = QtGui.QAction(text, self)
        if object_name:
            action.setObjectName(object_name)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda checked=False: callback())
        menu.addAction(action)
        return action

    def _command_values(self, title, fields):
        dialog = CommandParameterDialog(title, fields, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.values()

    def _primitive_fields(self, primitive_type):
        fields = self._base_creation_fields()
        if primitive_type == "sphere":
            fields.insert(1, self._float_field("radius", "Radius", 1.0, minimum=0.001))
        elif primitive_type == "cube":
            fields.insert(1, self._float_field("side_length", "Side Length", 1.0, minimum=0.001))
        elif primitive_type == "plane":
            fields.insert(1, self._float_field("distance", "Distance", 0.0))
        elif primitive_type == "torus":
            fields.insert(1, self._float_field("major_radius", "Major Radius", 1.0, minimum=0.001))
            fields.insert(2, self._float_field("minor_radius", "Minor Radius", 0.25, minimum=0.001))
        return fields

    def _light_fields(self, light_type):
        fields = self._base_creation_fields(y=4.0)
        fields.insert(1, self._float_field("intensity", "Intensity", 20.0, minimum=0.0))
        if light_type == "sphere":
            fields.insert(1, self._float_field("radius", "Radius", 0.5, minimum=0.001))
        else:
            fields.insert(1, self._float_field("side_length", "Side Length", 1.0, minimum=0.001))
        return fields

    def _base_creation_fields(self, *, y=0.0):
        return [
            {"key": "name", "label": "Name", "type": "text", "default": ""},
            self._float_field("x", "Translate X", 0.0),
            self._float_field("y", "Translate Y", y),
            self._float_field("z", "Translate Z", 0.0),
        ]

    @staticmethod
    def _float_field(
        key,
        label,
        default,
        *,
        minimum=-1_000_000.0,
        maximum=1_000_000.0,
        decimals=3,
        step=0.1,
    ):
        return {
            "key": key,
            "label": label,
            "default": default,
            "minimum": minimum,
            "maximum": maximum,
            "decimals": decimals,
            "step": step,
        }

    def _run_script_command(self, command, *args, **kwargs):
        try:
            return command(*args, **kwargs)
        except Exception as exc:
            self._show_error("Command Failed", str(exc))
            return None

    def _load_recent_files(self):
        value = self._settings.value(SETTINGS_RECENT_FILES_KEY, [])
        if isinstance(value, str):
            value = [value]
        return [Path(path) for path in value if path]

    def _add_recent_file(self, path):
        path = Path(path)
        self._recent_files = [
            existing for existing in self._recent_files if existing != path
        ]
        self._recent_files.insert(0, path)
        del self._recent_files[MAX_RECENT_FILES:]
        self._settings.setValue(
            SETTINGS_RECENT_FILES_KEY,
            [str(path) for path in self._recent_files],
        )
        self._update_recent_files_menu()

    def _update_recent_files_menu(self):
        if self._recent_files_menu is None:
            return
        self._recent_files_menu.clear()
        if not self._recent_files:
            action = self._recent_files_menu.addAction("No Recent Files")
            action.setEnabled(False)
            return
        for path in self._recent_files:
            action = self._recent_files_menu.addAction(str(path))
            action.triggered.connect(lambda checked=False, p=path: self._open_recent_file(p))

    def _open_recent_file(self, path):
        path = Path(path)
        if not path.exists():
            self._recent_files = [
                existing for existing in self._recent_files if existing != path
            ]
            self._update_recent_files_menu()
            self._show_error("Missing Recent File", f"{path} no longer exists.")
            return False
        return self.open_scene_path(path)

    def _update_window_title(self):
        suffix = f" - {self._current_path.name}" if self._current_path else ""
        self.setWindowTitle(f"Roxy{suffix}")

    def _show_error(self, title, message):
        QtWidgets.QMessageBox.warning(self, title, message)


def run(world=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = RoxyMainWindow(world)
    window.show()
    return app.exec()


def _demo_world():
    from core import Color, Vec3
    from scene import Camera, Diffuse, SceneObject, Sphere, World

    world = World(use_sky=False)
    root = SceneObject(name="root")
    root.add_child(
        SceneObject(
            shape=Sphere(1.0),
            material=Diffuse(Color(0.8, 0.2, 0.2)),
            translation=Vec3(0, 1, 0),
            name="sphere",
        )
    )
    world.add_object(root)
    world.add_camera(Camera(name="camera1"))
    return world


if __name__ == "__main__":
    run(_demo_world())
