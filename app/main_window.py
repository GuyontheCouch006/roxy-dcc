from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6 import QtCore, QtGui, QtWidgets

from app.scene_graph import SceneGraphModel, SceneGraphRoles
from app.viewport import QtGLViewport
from scene import SceneObject, SceneSession


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


class RoxyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Roxy")
        self.resize(1280, 820)

        self._session = SceneSession(world)
        self._scene_graph = SceneGraphPanel(parent=self, session=self._session)
        self._viewport = QtGLViewport(parent=self, session=self._session)

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

    def set_world(self, world):
        self._session.set_world(world)

    def undo(self):
        return self._session.undo()


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
