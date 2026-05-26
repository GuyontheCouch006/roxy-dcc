from __future__ import annotations

from dataclasses import dataclass, field

from PySide6 import QtCore


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
        while node is not None and node.kind != "root":
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))


class SceneGraphModel(QtCore.QAbstractItemModel):
    """Qt item model that presents a scene.World as an outliner tree."""

    COLUMNS = ("Name", "Type", "Details")

    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self._world = world
        self._root = self._build_tree(world)

    @property
    def world(self):
        return self._world

    def set_world(self, world):
        self.beginResetModel()
        self._world = world
        self._root = self._build_tree(world)
        self.endResetModel()

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
        column = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return node.name
            if column == 1:
                return _display_kind(node.kind)
            if column == 2:
                return node.detail
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return node.path
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
            and column == 0
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
            node.payload.visible = _check_state_value(value) == _check_state_value(
                QtCore.Qt.CheckState.Checked
            )
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

    def _build_tree(self, world):
        root = SceneGraphNode("Root", "root")
        if world is None:
            root.append(SceneGraphNode("World", "world", detail="empty"))
            return root

        world_node = root.append(
            SceneGraphNode("World", "world", world, _world_detail(world))
        )

        objects = world_node.append(
            SceneGraphNode("Objects", "section", detail=_count_detail(world.objects, "object"))
        )
        for row, obj in enumerate(world.objects):
            objects.append(_object_node(obj, row))

        cameras = world_node.append(
            SceneGraphNode("Cameras", "section", detail=_count_detail(world.cameras, "camera"))
        )
        for row, camera in enumerate(world.cameras):
            cameras.append(_camera_node(camera, row, camera is world.active_camera))

        lights = world_node.append(
            SceneGraphNode("Lights", "section", detail=_count_detail(world.lights, "light"))
        )
        for row, light in enumerate(world.lights):
            lights.append(_light_node(light, row))

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
