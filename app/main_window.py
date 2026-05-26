from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.scene_graph import SceneGraphModel, SceneGraphRoles


class SceneGraphPanel(QtWidgets.QWidget):
    nodeSelected = QtCore.Signal(object)

    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self._model = SceneGraphModel(world, self)

        self._tree = QtWidgets.QTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.setIconSize(QtCore.QSize(18, 18))
        self._tree.expandToDepth(1)
        self._tree.header().setHidden(True)
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._tree.selectionModel().currentChanged.connect(self._on_current_changed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    @property
    def model(self):
        return self._model

    @property
    def tree(self):
        return self._tree

    def set_world(self, world):
        self._model.set_world(world)
        self._tree.expandToDepth(1)

    def selected_payload(self):
        index = self._tree.currentIndex()
        return index.data(SceneGraphRoles.PayloadRole) if index.isValid() else None

    def _on_current_changed(self, current, previous):
        del previous
        self.nodeSelected.emit(current.data(SceneGraphRoles.PayloadRole))


class RoxyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Roxy")
        self.resize(1280, 820)

        self._scene_graph = SceneGraphPanel(world, self)
        scene_graph_dock = QtWidgets.QDockWidget("Scene Graph", self)
        scene_graph_dock.setObjectName("sceneGraphDock")
        scene_graph_dock.setWidget(self._scene_graph)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, scene_graph_dock)

        self._central = QtWidgets.QFrame(self)
        self._central.setObjectName("viewportHost")
        self._central.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setCentralWidget(self._central)

    @property
    def scene_graph(self):
        return self._scene_graph

    def set_world(self, world):
        self._scene_graph.set_world(world)


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
