from __future__ import annotations

from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui


class SceneGraphRoles:
    NodeRole = QtCore.Qt.ItemDataRole.UserRole + 1
    PayloadRole = QtCore.Qt.ItemDataRole.UserRole + 2
    KindRole = QtCore.Qt.ItemDataRole.UserRole + 3
    PathRole = QtCore.Qt.ItemDataRole.UserRole + 4


@dataclass
class SceneGraphNode:
    name: str
    kind: str
    payload: object = None
    detail: str = ""
    parent: "SceneGraphNode | None" = None
    children: list["SceneGraphNode"] = field(default_factory=list)

    def append(self, child: "SceneGraphNode"):
        child.parent = self
        self.children.append(child)
        return child

    def child(self, row):
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self):
        if self.parent is None:
            return 0
        return self.parent.children.index(self)

    @property
    def path(self):
        parts = []
        node = self
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))


class SceneGraphModel(QtCore.QAbstractItemModel):
    """Qt item model that presents a scene.World as an outliner tree."""

    COLUMNS = ("Name",)

    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self._world = world
        self._session = None
        self._root = self._build_tree(world)

    @property
    def world(self):
        return self._world

    def set_world(self, world):
        self.beginResetModel()
        self._world = world
        self._root = self._build_tree(world)
        self.endResetModel()

    def set_session(self, session):
        self._session = session
        self.set_world(session.world if session is not None else None)

    def refresh(self):
        self.set_world(self._world)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        parent_node = self.node_from_index(parent)
        child_node = parent_node.child(row)
        if child_node is None:
            return QtCore.QModelIndex()
        return self.createIndex(row, column, child_node)

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()

        node = index.internalPointer()
        parent_node = node.parent
        if parent_node is None or parent_node is self._root:
            return QtCore.QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid() and parent.column() > 0:
            return 0
        return len(self.node_from_index(parent).children)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        node = index.internalPointer()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return node.name
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            return _icon_for_kind(node.kind)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return _node_tooltip(node)
        if role == SceneGraphRoles.NodeRole:
            return node
        if role == SceneGraphRoles.PayloadRole:
            return node.payload
        if role == SceneGraphRoles.KindRole:
            return node.kind
        if role == SceneGraphRoles.PathRole:
            return node.path
        if (
            role == QtCore.Qt.ItemDataRole.CheckStateRole
            and index.column() == 0
            and node.kind == "object"
        ):
            return (
                QtCore.Qt.CheckState.Checked
                if getattr(node.payload, "visible", True)
                else QtCore.Qt.CheckState.Unchecked
            )
        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if not index.isValid():
            return False

        node = index.internalPointer()
        if (
            role == QtCore.Qt.ItemDataRole.CheckStateRole
            and index.column() == 0
            and node.kind == "object"
        ):
            visible = _check_state_value(value) == _check_state_value(
                QtCore.Qt.CheckState.Checked
            )
            if self._session is not None:
                self._session.set_object_visible(node.payload, visible)
            else:
                node.payload.visible = visible
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags

        flags = (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        node = index.internalPointer()
        if index.column() == 0 and node.kind == "object":
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(
        self,
        section,
        orientation,
        role=QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
            and 0 <= section < len(self.COLUMNS)
        ):
            return self.COLUMNS[section]
        return None

    def node_from_index(self, index):
        if index.isValid():
            return index.internalPointer()
        return self._root

    def index_for_payload(self, payload, column=0):
        node = self._find_node(lambda item: item.payload is payload)
        if node is None or node is self._root:
            return QtCore.QModelIndex()
        return self.createIndex(node.row(), column, node)

    def scene_objects_for_index(self, index):
        if not index.isValid():
            return ()
        return tuple(
            node.payload
            for node in self._iter_subtree(self.node_from_index(index))
            if node.kind == "object" and node.payload is not None
        )

    def scene_object_for_index(self, index):
        if not index.isValid():
            return None
        node = self.node_from_index(index)
        if node.kind == "object":
            return node.payload
        return None

    def iter_nodes(self, kind=None):
        def walk(node):
            for child in node.children:
                if kind is None or child.kind == kind:
                    yield child
                yield from walk(child)

        return walk(self._root)

    def _find_node(self, predicate):
        for node in self.iter_nodes():
            if predicate(node):
                return node
        return None

    def _iter_subtree(self, node):
        if node is None:
            return
        yield node
        for child in node.children:
            yield from self._iter_subtree(child)

    def _build_tree(self, world):
        if world is None:
            return SceneGraphNode("World", "world", detail="empty")

        root = SceneGraphNode("World", "world", world, _world_detail(world))

        for row, obj in enumerate(world.objects):
            root.append(_object_node(obj, row))

        for row, camera in enumerate(world.cameras):
            root.append(_camera_node(camera, row, camera is world.active_camera))

        for row, light in enumerate(world.lights):
            root.append(_light_node(light, row))

        return root


def _object_node(obj, row):
    shapes = getattr(obj, "shapes", ())
    children = getattr(obj, "children", ())
    node = SceneGraphNode(
        _node_name(obj, f"object{row + 1}"),
        "object",
        obj,
        _object_detail(obj),
    )

    for child_row, child in enumerate(children):
        node.append(_object_node(child, child_row))

    for shape_row, shape in enumerate(shapes):
        node.append(_shape_node(shape, shape_row))

    return node


def _shape_node(shape, row):
    geometry = getattr(shape, "geometry", None)
    geometry_type = type(geometry).__name__ if geometry is not None else "None"
    return SceneGraphNode(
        _node_name(shape, f"shape{row + 1}", fallback_attr="name") or geometry_type,
        "shape",
        shape,
        _shape_detail(shape, geometry_type),
    )


def _camera_node(camera, row, active):
    parts = []
    if active:
        parts.append("active")
    if hasattr(camera, "fov"):
        parts.append(f"fov {camera.fov:g}")
    if hasattr(camera, "aspect_ratio"):
        parts.append(f"aspect {camera.aspect_ratio:.3f}")
    return SceneGraphNode(
        _node_name(camera, f"camera{row + 1}"),
        "camera",
        camera,
        ", ".join(parts),
    )


def _light_node(light, row):
    return SceneGraphNode(
        _node_name(light, f"light{row + 1}"),
        "light",
        light,
        type(light).__name__,
    )


def _world_detail(world):
    return (
        f"{len(world.objects)} objects, "
        f"{len(world.cameras)} cameras, "
        f"{len(world.lights)} lights"
    )


def _object_detail(obj):
    shape_count = len(getattr(obj, "shapes", ()))
    child_count = len(getattr(obj, "children", ()))
    flags = []
    if not getattr(obj, "visible", True):
        flags.append("hidden")
    if not getattr(obj, "renderable", True):
        flags.append("not renderable")
    if not getattr(obj, "selectable", True):
        flags.append("not selectable")

    detail = f"{_count_label(shape_count, 'shape')}, {_count_label(child_count, 'child')}"
    if flags:
        detail = f"{detail}; {', '.join(flags)}"
    return detail


def _shape_detail(shape, geometry_type):
    groups = getattr(shape, "material_groups", {})
    if not groups:
        return geometry_type
    names = ", ".join(str(name) for name in groups.keys())
    return f"{geometry_type}; groups: {names}"


def _count_detail(items, label):
    return _count_label(len(items), label)


def _count_label(count, label):
    suffix = "" if count == 1 else "s"
    return f"{count} {label}{suffix}"


def _node_name(item, fallback, fallback_attr="name"):
    name = getattr(item, fallback_attr, "")
    return str(name) if name else fallback


def _display_kind(kind):
    if kind == "section":
        return ""
    return kind.title()


def _check_state_value(value):
    return getattr(value, "value", value)


_ICON_CACHE = {}


def _icon_for_kind(kind):
    if QtGui.QGuiApplication.instance() is None:
        return None
    if kind not in _ICON_CACHE:
        _ICON_CACHE[kind] = _build_icon(kind)
    return _ICON_CACHE[kind]


def _build_icon(kind):
    size = 18
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    color = _icon_color(kind)
    pen = QtGui.QPen(color.darker(135), 1.4)
    painter.setPen(pen)
    painter.setBrush(QtGui.QBrush(color))

    if kind == "world":
        painter.drawEllipse(QtCore.QRectF(3.0, 3.0, 12.0, 12.0))
        painter.setPen(QtGui.QPen(color.lighter(160), 1.0))
        painter.drawArc(QtCore.QRectF(5.0, 3.0, 8.0, 12.0), 0, 360 * 16)
        painter.drawLine(QtCore.QPointF(3.5, 9.0), QtCore.QPointF(14.5, 9.0))
    elif kind == "section":
        painter.drawRoundedRect(QtCore.QRectF(2.5, 5.0, 13.0, 9.5), 2.0, 2.0)
        painter.drawRect(QtCore.QRectF(3.5, 3.5, 5.0, 3.0))
    elif kind == "object":
        points = [
            QtCore.QPointF(9.0, 2.5),
            QtCore.QPointF(14.5, 5.8),
            QtCore.QPointF(14.5, 12.2),
            QtCore.QPointF(9.0, 15.5),
            QtCore.QPointF(3.5, 12.2),
            QtCore.QPointF(3.5, 5.8),
        ]
        painter.drawPolygon(QtGui.QPolygonF(points))
        painter.setPen(QtGui.QPen(color.darker(155), 1.0))
        painter.drawLine(QtCore.QPointF(9.0, 2.5), QtCore.QPointF(9.0, 15.5))
        painter.drawLine(QtCore.QPointF(3.5, 5.8), QtCore.QPointF(14.5, 12.2))
    elif kind == "shape":
        painter.drawEllipse(QtCore.QRectF(3.0, 4.0, 12.0, 10.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QtCore.QRectF(5.5, 5.5, 7.0, 7.0))
    elif kind == "camera":
        painter.drawRoundedRect(QtCore.QRectF(2.5, 5.5, 9.5, 7.0), 1.5, 1.5)
        painter.drawPolygon(
            QtGui.QPolygonF([
                QtCore.QPointF(12.0, 7.0),
                QtCore.QPointF(16.0, 5.5),
                QtCore.QPointF(16.0, 12.5),
                QtCore.QPointF(12.0, 11.0),
            ])
        )
    elif kind == "light":
        painter.drawEllipse(QtCore.QRectF(5.2, 4.0, 7.6, 7.6))
        painter.drawLine(QtCore.QPointF(9.0, 1.8), QtCore.QPointF(9.0, 3.2))
        painter.drawLine(QtCore.QPointF(9.0, 12.8), QtCore.QPointF(9.0, 16.2))
        painter.drawLine(QtCore.QPointF(2.0, 8.0), QtCore.QPointF(4.0, 8.0))
        painter.drawLine(QtCore.QPointF(14.0, 8.0), QtCore.QPointF(16.0, 8.0))
    else:
        painter.drawRoundedRect(QtCore.QRectF(4.0, 4.0, 10.0, 10.0), 2.0, 2.0)

    painter.end()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


def _icon_color(kind):
    colors = {
        "world": QtGui.QColor("#5aa7ff"),
        "section": QtGui.QColor("#88939f"),
        "object": QtGui.QColor("#f2b84b"),
        "shape": QtGui.QColor("#7fd48b"),
        "camera": QtGui.QColor("#b48cff"),
        "light": QtGui.QColor("#ffd95a"),
    }
    return colors.get(kind, QtGui.QColor("#b8c0cc"))


def _node_tooltip(node):
    parts = [_display_kind(node.kind) or node.name]
    if node.detail:
        parts.append(node.detail)
    if node.path:
        parts.append(node.path)
    return "\n".join(parts)
