from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from scene import Camera, SceneObject
from scene.history import GeometrySourceNode
from scene.materials import Material
from scene.shape import Shape


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    label: str
    payload: object
    x: float
    y: float


@dataclass(frozen=True)
class GraphEdge:
    source: str
    destination: str
    kind: str
    label: str = ""


class SceneNodeGraphModel:
    def __init__(self, session):
        self.session = session
        self.nodes = []
        self.edges = []
        self._build()

    def _build(self):
        y = 0.0
        material_nodes = {}
        history_nodes = {node.shape: node for node in self.session.history_nodes(raw=True)}
        for obj in self.session.world.objects:
            y = self._add_object_tree(
                obj,
                depth=0,
                y=y,
                material_nodes=material_nodes,
                history_nodes=history_nodes,
            )

        camera_x = 0.0
        for row, camera in enumerate(self.session.world.cameras):
            node_id = _node_id("camera", camera)
            self.nodes.append(GraphNode(node_id, "camera", camera.name, camera, camera_x, y + row * 90.0))

    def _add_object_tree(self, obj, depth, y, material_nodes, history_nodes):
        object_id = _node_id("object", obj)
        self.nodes.append(GraphNode(object_id, "object", obj.name or "transform", obj, depth * 220.0, y))
        shape_y = y
        for shape in obj.shapes:
            source = history_nodes[shape]
            source_id = _node_id("history", source)
            shape_id = _node_id("shape", shape)
            self.nodes.append(GraphNode(source_id, "history", source.name, source, (depth + 1) * 220.0, shape_y - 35.0))
            self.nodes.append(GraphNode(shape_id, "shape", shape.name or type(shape.geometry).__name__, shape, (depth + 2) * 220.0, shape_y))
            self.edges.append(GraphEdge(object_id, shape_id, "ownership", "shape"))
            self.edges.append(GraphEdge(source_id, shape_id, "geometry", "inGeometry"))
            for group, material in shape.material_groups.items():
                material_id = material_nodes.get(id(material))
                if material_id is None:
                    material_id = _node_id("material", material)
                    material_nodes[id(material)] = material_id
                    self.nodes.append(GraphNode(material_id, "material", _material_name(material), material, (depth + 3) * 220.0, shape_y))
                self.edges.append(GraphEdge(material_id, shape_id, "shader", group))
            shape_y += 95.0
        child_y = max(y + 95.0, shape_y)
        for child in obj.children:
            child_id = _node_id("object", child)
            child_y = self._add_object_tree(
                child,
                depth + 1,
                child_y,
                material_nodes,
                history_nodes,
            )
            self.edges.append(GraphEdge(object_id, child_id, "hierarchy", "parent"))
        return max(child_y, shape_y, y + 95.0)

    def node_for_payload(self, payload):
        for node in self.nodes:
            if node.payload is payload:
                return node
        return None


class NodeNetworkPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, session=None):
        super().__init__(parent)
        self._session = None
        self._model = None
        self._items = {}
        self._applying_selection = False

        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._view.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._scene.selectionChanged.connect(self._on_graph_selection_changed)

        if session is not None:
            self.set_session(session)

    @property
    def model(self):
        return self._model

    @property
    def view(self):
        return self._view

    def set_session(self, session):
        if self._session is session:
            return
        if self._session is not None:
            self._session.remove_payload_listener(self._on_payload_changed)
            self._session.remove_scene_listener(self._on_scene_changed)
            self._session.remove_world_listener(self._on_world_changed)
        self._session = session
        if session is not None:
            session.add_payload_listener(self._on_payload_changed)
            session.add_scene_listener(self._on_scene_changed)
            session.add_world_listener(self._on_world_changed)
        self.rebuild()

    def rebuild(self):
        self._scene.clear()
        self._items.clear()
        if self._session is None:
            self._model = None
            return
        self._model = SceneNodeGraphModel(self._session)
        for edge in self._model.edges:
            self._add_edge(edge)
        for node in self._model.nodes:
            self._add_node(node)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-80, -80, 80, 80))

    def connect_shader(self, material, shape, group="default"):
        if self._session is None:
            return
        self._session.shape(shape).connect_shader(material, group=group)

    def _add_node(self, node):
        item = _GraphNodeItem(node)
        self._scene.addItem(item)
        self._items[node.node_id] = item

    def _add_edge(self, edge):
        if self._model is None:
            return
        source = next((node for node in self._model.nodes if node.node_id == edge.source), None)
        destination = next((node for node in self._model.nodes if node.node_id == edge.destination), None)
        if source is None or destination is None:
            return
        line = QtWidgets.QGraphicsLineItem(
            source.x + 150,
            source.y + 25,
            destination.x,
            destination.y + 25,
        )
        pen = QtGui.QPen(_edge_color(edge.kind), 1.4)
        line.setPen(pen)
        line.setZValue(-1)
        self._scene.addItem(line)

    def _on_payload_changed(self, session):
        payload = session.selected_raw_payload()
        self._applying_selection = True
        try:
            self._scene.clearSelection()
            if self._model is None:
                return
            node = self._model.node_for_payload(payload)
            if node is not None and node.node_id in self._items:
                self._items[node.node_id].setSelected(True)
        finally:
            self._applying_selection = False

    def _on_scene_changed(self, session):
        del session
        self.rebuild()

    def _on_world_changed(self, session):
        del session
        self.rebuild()

    def _on_graph_selection_changed(self):
        if self._applying_selection or self._session is None:
            return
        selected = self._scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        payload = item.data(0)
        if isinstance(payload, SceneObject):
            self._session.replace_selection(payload)
        elif payload is not None:
            self._session.set_selected_payload(payload)


class _GraphNodeItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, node):
        super().__init__(0, 0, 150, 50)
        self.setPos(node.x, node.y)
        self.setBrush(QtGui.QBrush(_node_color(node.kind)))
        self.setPen(QtGui.QPen(QtGui.QColor("#20242a"), 1.5))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setData(0, node.payload)

        text = QtWidgets.QGraphicsTextItem(node.label, self)
        text.setDefaultTextColor(QtGui.QColor("#f1f4f8"))
        text.setTextWidth(136)
        text.setPos(7, 7)


def _node_id(kind, payload):
    return f"{kind}:{id(payload)}"


def _material_name(material):
    return getattr(material, "name", "") or type(material).__name__


def _node_color(kind):
    colors = {
        "object": QtGui.QColor("#b77f2d"),
        "shape": QtGui.QColor("#3b8f52"),
        "material": QtGui.QColor("#9f4266"),
        "camera": QtGui.QColor("#7654ad"),
        "history": QtGui.QColor("#41698f"),
    }
    return colors.get(kind, QtGui.QColor("#4f5965"))


def _edge_color(kind):
    colors = {
        "shader": QtGui.QColor("#ee7aa8"),
        "geometry": QtGui.QColor("#5aa7ff"),
        "hierarchy": QtGui.QColor("#a0a9b4"),
        "ownership": QtGui.QColor("#7fd48b"),
    }
    return colors.get(kind, QtGui.QColor("#7b8490"))
