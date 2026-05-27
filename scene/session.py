from __future__ import annotations

from core import AABB, Vec3
from scene.commands import (
    AddObjectsCommand,
    AssignMaterialCommand,
    AttributeCommand,
    DisconnectMaterialCommand,
    PayloadSelectionCommand,
    SelectionCommand,
    SessionSettings,
    TransformObjectsCommand,
    UndoStack,
    VisibilityCommand,
)
from scene.camera import Camera
from scene.history import GeometrySourceNode
from scene.materials import Material
from scene.node_handles import CameraHandle, HistoryNodeHandle, MaterialHandle, ShapeHandle
from scene.object_handle import ObjectHandle
from scene.scene_object import SceneObject
from scene.shape import Shape
from scene.transform_ops import (
    apply_absolute_matrix,
    apply_relative_matrix,
    expanded_with_descendants,
    parent_linear_is_identity,
    rotation_matrix,
    rootmost,
    scale_matrix,
    translation_matrix,
    walk_scene_objects,
    world_origin,
    world_vector_to_parent_local,
)
from scene.transform_state import capture_transforms, restore_transforms
from scene.world import World


class SceneSession:
    """Scriptable scene context shared by UI tools and Python scripts."""

    def __init__(self, world, undo_enabled=True, undo_depth=10):
        self.world = world
        self.settings = SessionSettings()
        self.settings.undo.enabled = bool(undo_enabled)
        self.settings.undo.depth = int(undo_depth)
        self._undo_stack = UndoStack(self.settings.undo)
        self._handles = {}
        self._selected = ()
        self._active = None
        self._selected_payload = None
        self._selection_listeners = []
        self._payload_listeners = []
        self._scene_listeners = []
        self._preview = None
        self._world_listeners = []
        self._history_nodes = {}

    @property
    def undo_count(self):
        return self._undo_stack.undo_count

    @property
    def redo_count(self):
        return self._undo_stack.redo_count

    @property
    def can_undo(self):
        return self.undo_count > 0

    @property
    def can_redo(self):
        return self.redo_count > 0

    def set_world(self, world):
        if world is self.world:
            return
        self.world = world
        self._handles.clear()
        self._history_nodes.clear()
        self._selected = ()
        self._active = None
        self._selected_payload = None
        self._undo_stack.clear()
        self._notify_world_changed()
        self._notify_selection_changed()
        self._notify_payload_changed()

    def add_object(self, scene_object, select=True):
        self.add_objects((scene_object,), select=select)
        return self._handle_for(scene_object)

    def add_objects(self, scene_objects, select=True, active=None):
        scene_objects = tuple(scene_objects)
        if not scene_objects:
            return ()
        command = AddObjectsCommand(self, scene_objects, select=select, active=active)
        self._execute(command)
        return tuple(self._handle_for(scene_object) for scene_object in scene_objects)

    def notify_world_changed(self):
        self._notify_world_changed()
        self._notify_selection_changed()
        self._notify_scene_changed()

    def object(self, name_or_scene_object):
        if isinstance(name_or_scene_object, ObjectHandle):
            return name_or_scene_object
        if isinstance(name_or_scene_object, SceneObject):
            if name_or_scene_object not in tuple(walk_scene_objects(self.world)):
                raise ValueError("SceneObject is not in this session's world")
            return self._handle_for(name_or_scene_object)
        for scene_object in walk_scene_objects(self.world):
            if scene_object.name == name_or_scene_object:
                return self._handle_for(scene_object)
        raise KeyError(f"No scene object named {name_or_scene_object!r}")

    def objects(self):
        return tuple(
            self._handle_for(scene_object)
            for scene_object in walk_scene_objects(self.world)
        )

    def shape(self, name_or_shape):
        if isinstance(name_or_shape, ShapeHandle):
            return name_or_shape
        if isinstance(name_or_shape, Shape):
            if name_or_shape not in tuple(self._walk_shapes()):
                raise ValueError("Shape is not in this session's world")
            return self._shape_handle_for(name_or_shape)
        for shape in self._walk_shapes():
            if shape.name == name_or_shape:
                return self._shape_handle_for(shape)
        raise KeyError(f"No shape named {name_or_shape!r}")

    def shapes(self):
        return tuple(self._shape_handle_for(shape) for shape in self._walk_shapes())

    def material(self, name_or_material):
        if isinstance(name_or_material, MaterialHandle):
            return name_or_material
        if isinstance(name_or_material, Material):
            if name_or_material not in tuple(self._walk_materials()):
                raise ValueError("Material is not in this session's world")
            return self._material_handle_for(name_or_material)
        for material in self._walk_materials():
            if _material_display_name(material) == name_or_material:
                return self._material_handle_for(material)
        raise KeyError(f"No material named {name_or_material!r}")

    def materials(self):
        seen = set()
        handles = []
        for material in self._walk_materials():
            key = id(material)
            if key in seen:
                continue
            seen.add(key)
            handles.append(self._material_handle_for(material))
        return tuple(handles)

    def camera(self, name_or_camera):
        if isinstance(name_or_camera, CameraHandle):
            return name_or_camera
        if isinstance(name_or_camera, Camera):
            if name_or_camera not in self.world.cameras:
                raise ValueError("Camera is not in this session's world")
            return self._camera_handle_for(name_or_camera)
        for camera in self.world.cameras:
            if camera.name == name_or_camera:
                return self._camera_handle_for(camera)
        raise KeyError(f"No camera named {name_or_camera!r}")

    def cameras(self):
        return tuple(self._camera_handle_for(camera) for camera in self.world.cameras)

    def history_node(self, name_or_node):
        if isinstance(name_or_node, HistoryNodeHandle):
            return name_or_node
        if isinstance(name_or_node, GeometrySourceNode):
            node = self._history_node_for_shape(name_or_node.shape)
            if node is not name_or_node:
                raise ValueError("History node is not in this session's world")
            return self._history_handle_for(node)
        for node in self.history_nodes(raw=True):
            if node.name == name_or_node:
                return self._history_handle_for(node)
        raise KeyError(f"No history node named {name_or_node!r}")

    def history_nodes(self, raw=False):
        nodes = tuple(self._history_node_for_shape(shape) for shape in self._walk_shapes())
        if raw:
            return nodes
        return tuple(self._history_handle_for(node) for node in nodes)

    def selected(self):
        return tuple(self._handle_for(scene_object) for scene_object in self._selected)

    def selected_objects(self):
        return self.selected()

    def selected_scene_objects(self):
        return self._selected

    def selected_payload(self):
        return self._payload_handle_for(self._selected_payload)

    def selected_raw_payload(self):
        return self._selected_payload

    def set_selected_payload(self, payload):
        payload = self._coerce_payload(payload)
        if payload is self._selected_payload:
            return
        self._execute(PayloadSelectionCommand(self, self._selected_payload, payload))

    def active_object(self):
        return self._handle_for(self._active) if self._active is not None else None

    def active_scene_object(self):
        return self._active

    def highlighted_objects(self):
        return tuple(
            self._handle_for(scene_object)
            for scene_object in self.highlighted_scene_objects()
        )

    def highlighted_scene_objects(self):
        return expanded_with_descendants(self._selected)

    def transform_targets(self):
        return tuple(
            self._handle_for(scene_object)
            for scene_object in self.transform_target_objects()
        )

    def rootmost_selection(self):
        return self.transform_targets()

    def transform_target_objects(self):
        return rootmost(self._selected)

    def set_selection(self, scene_objects, active=None):
        if isinstance(scene_objects, (SceneObject, ObjectHandle)):
            scene_objects = (scene_objects,)
        selected = self._coerce_scene_objects(scene_objects)
        active = self._coerce_active(active, selected)
        if selected == self._selected and active is self._active:
            return
        command = SelectionCommand(
            self,
            self._selected,
            self._active,
            selected,
            active,
        )
        self._execute(command)

    def select(self, scene_objects, active=None):
        if isinstance(scene_objects, (SceneObject, ObjectHandle)):
            self.replace_selection(scene_objects)
            return
        self.set_selection(scene_objects, active=active)

    def replace_selection(self, scene_object):
        if scene_object is None:
            self.clear_selection()
            return
        scene_object = self._coerce_scene_object(scene_object)
        self.set_selection((scene_object,), active=scene_object)

    def toggle_selection(self, scene_object):
        scene_object = self._coerce_scene_object(scene_object)
        selected = list(self._selected)
        if scene_object in selected:
            selected.remove(scene_object)
            active = selected[-1] if selected else None
        else:
            selected.append(scene_object)
            active = scene_object
        self.set_selection(selected, active=active)

    def clear_selection(self):
        self.set_selection((), active=None)

    def set_attr(self, target, attr_name, value):
        target = self._coerce_payload(target)
        before = self._read_attr_raw(target, attr_name)
        after = self._coerce_attr_value(target, attr_name, value)
        if before == after:
            return
        self._execute(
            AttributeCommand(
                self,
                target,
                attr_name,
                before,
                after,
                label=f"Set {attr_name}",
            )
        )

    def assign_material(self, shape, material, group="default"):
        shape = self._coerce_shape(shape)
        material = self._coerce_material(material)
        group = str(group or "default")
        before = shape.material_groups.get(group)
        if before is material:
            return
        self._execute(AssignMaterialCommand(shape, group, before, material))

    def disconnect_material(self, shape, group="default"):
        shape = self._coerce_shape(shape)
        group = str(group or "default")
        before = shape.material_groups.get(group)
        if before is None:
            return
        self._execute(DisconnectMaterialCommand(shape, group, before))

    def connect_attr(self, source, destination):
        material = self._material_from_plug_source(source)
        shape, group = self._shape_group_from_plug_destination(destination)
        self.assign_material(shape, material, group=group)
        return self.shape(shape)

    def disconnect_attr(self, source_or_destination, destination=None):
        destination = source_or_destination if destination is None else destination
        shape, group = self._shape_group_from_plug_destination(destination)
        self.disconnect_material(shape, group=group)
        return self.shape(shape)

    def transform_objects(
        self,
        scene_objects,
        *,
        matrix,
        local=False,
        relative=False,
        pivot=None,
        label="Transform",
    ):
        targets = self._coerce_scene_objects(scene_objects)
        if not targets:
            return
        before = capture_transforms(targets)
        self._apply_matrix(targets, matrix, local=local, relative=relative, pivot=pivot)
        after = capture_transforms(targets)
        self._execute(TransformObjectsCommand(before, after, label=label))

    def set_local_trs_objects(
        self,
        scene_objects,
        *,
        translate,
        rotate,
        scale,
        label="Transform",
    ):
        targets = self._coerce_scene_objects(scene_objects)
        if not targets:
            return

        def mutate(scene_object):
            scene_object.translation = translate
            scene_object.rotation = rotate
            scene_object.scale = scale
            scene_object.shear = Vec3(0, 0, 0)
            scene_object.pivot = Vec3(0, 0, 0)

        self._edit_transforms(targets, mutate, label)

    def move_objects(
        self,
        scene_objects,
        *,
        x=0.0,
        y=0.0,
        z=0.0,
        local=False,
        label="Move",
    ):
        targets = self._coerce_scene_objects(scene_objects)
        if not targets:
            return

        def mutate(scene_object):
            if scene_object.matrix_mode:
                apply_relative_matrix(
                    scene_object,
                    translation_matrix(x, y, z),
                    local=local,
                )
                return
            local_delta = Vec3(x, y, z) if local else world_vector_to_parent_local(
                scene_object,
                (x, y, z),
            )
            scene_object.translation = scene_object.translation + local_delta

        self._edit_transforms(targets, mutate, label)

    def rotate_objects(
        self,
        scene_objects,
        *,
        x=0.0,
        y=0.0,
        z=0.0,
        local=False,
        pivot=None,
        label="Rotate",
    ):
        targets = self._coerce_scene_objects(scene_objects)
        if not targets:
            return
        matrix = rotation_matrix(x, y, z)

        def mutate(scene_object):
            if (
                not scene_object.matrix_mode
                and pivot is None
                and (local or parent_linear_is_identity(scene_object))
            ):
                scene_object.rotation = (
                    scene_object.rotation
                    + scene_object.rotation.__class__(x, y, z)
                )
                return
            apply_relative_matrix(
                scene_object,
                matrix,
                local=local,
                pivot=pivot if pivot is not None else world_origin(scene_object),
            )

        self._edit_transforms(targets, mutate, label)

    def scale_objects(
        self,
        scene_objects,
        *,
        x=1.0,
        y=1.0,
        z=1.0,
        local=False,
        pivot=None,
        label="Scale",
    ):
        targets = self._coerce_scene_objects(scene_objects)
        if not targets:
            return
        matrix = scale_matrix(x, y, z)

        def mutate(scene_object):
            if (
                not scene_object.matrix_mode
                and pivot is None
                and (local or parent_linear_is_identity(scene_object))
            ):
                scene_object.scale = (
                    scene_object.scale
                    * scene_object.scale.__class__(x, y, z)
                )
                return
            apply_relative_matrix(
                scene_object,
                matrix,
                local=local,
                pivot=pivot if pivot is not None else world_origin(scene_object),
            )

        self._edit_transforms(targets, mutate, label)

    def begin_transform(self, label="Transform", scene_objects=None):
        if self._preview is not None:
            self.cancel_transform()
        targets = (
            rootmost(self._coerce_scene_objects(scene_objects))
            if scene_objects is not None
            else self.transform_target_objects()
        )
        if not targets:
            return False
        self._preview = {
            "label": label,
            "objects": targets,
            "before": capture_transforms(targets),
        }
        return True

    def preview_transform(self, matrix, *, local=False, pivot=None):
        if self._preview is None:
            return False
        before = self._preview["before"]
        targets = self._preview["objects"]
        restore_transforms(before)
        self._apply_matrix(targets, matrix, local=local, relative=True, pivot=pivot)
        self._notify_scene_changed()
        return True

    def finish_transform(self):
        if self._preview is None:
            return False
        preview = self._preview
        self._preview = None
        command = TransformObjectsCommand(
            preview["before"],
            capture_transforms(preview["objects"]),
            label=preview["label"],
        )
        self._execute(command)
        return True

    def cancel_transform(self):
        if self._preview is None:
            return False
        restore_transforms(self._preview["before"])
        self._preview = None
        self._notify_scene_changed()
        return True

    def set_object_visible(self, scene_object, visible):
        scene_object = self._coerce_scene_object(scene_object)
        visible = bool(visible)
        if scene_object.visible == visible:
            return
        self._execute(VisibilityCommand(scene_object, scene_object.visible, visible))

    def undo(self):
        command = self._undo_stack.undo()
        if command is None:
            return False
        self._notify_for_command(command)
        return True

    def redo(self):
        command = self._undo_stack.redo()
        if command is None:
            return False
        self._notify_for_command(command)
        return True

    def add_selection_listener(self, callback):
        self._selection_listeners.append(callback)
        return callback

    def remove_selection_listener(self, callback):
        if callback in self._selection_listeners:
            self._selection_listeners.remove(callback)

    def add_payload_listener(self, callback):
        self._payload_listeners.append(callback)
        return callback

    def remove_payload_listener(self, callback):
        if callback in self._payload_listeners:
            self._payload_listeners.remove(callback)

    def add_scene_listener(self, callback):
        self._scene_listeners.append(callback)
        return callback

    def remove_scene_listener(self, callback):
        if callback in self._scene_listeners:
            self._scene_listeners.remove(callback)

    def add_world_listener(self, callback):
        self._world_listeners.append(callback)
        return callback

    def remove_world_listener(self, callback):
        if callback in self._world_listeners:
            self._world_listeners.remove(callback)

    def _set_selection_raw(self, selected, active):
        self._selected = tuple(selected)
        self._active = active if active in self._selected else None
        self._selected_payload = self._active

    def _set_payload_selection_raw(self, payload):
        self._selected_payload = payload

    def _execute(self, command):
        applied = self._undo_stack.execute(command)
        self._notify_for_command(applied)

    def _edit_transforms(self, scene_objects, mutate, label):
        before = capture_transforms(scene_objects)
        for scene_object in scene_objects:
            mutate(scene_object)
        after = capture_transforms(scene_objects)
        self._execute(TransformObjectsCommand(before, after, label=label))

    def _notify_for_command(self, command):
        if command.affects_world:
            self._notify_world_changed()
        if command.affects_selection:
            self._notify_selection_changed()
            self._notify_payload_changed()
        if getattr(command, "affects_payload", False):
            self._notify_payload_changed()
        if command.affects_scene:
            self._notify_scene_changed()

    def _notify_selection_changed(self):
        for callback in tuple(self._selection_listeners):
            callback(self)

    def _notify_payload_changed(self):
        for callback in tuple(self._payload_listeners):
            callback(self)

    def _notify_scene_changed(self):
        for callback in tuple(self._scene_listeners):
            callback(self)

    def _notify_world_changed(self):
        for callback in tuple(self._world_listeners):
            callback(self)

    def _handle_for(self, scene_object):
        key = id(scene_object)
        handle = self._handles.get(key)
        if handle is None or handle.raw is not scene_object:
            handle = ObjectHandle(self, scene_object)
            self._handles[key] = handle
        return handle

    def _shape_handle_for(self, shape):
        return self._generic_handle_for(shape, ShapeHandle)

    def _material_handle_for(self, material):
        return self._generic_handle_for(material, MaterialHandle)

    def _camera_handle_for(self, camera):
        return self._generic_handle_for(camera, CameraHandle)

    def _history_handle_for(self, node):
        return self._generic_handle_for(node, HistoryNodeHandle)

    def _generic_handle_for(self, raw, handle_type):
        key = (handle_type, id(raw))
        handle = self._handles.get(key)
        if handle is None or handle.raw is not raw:
            handle = handle_type(self, raw)
            self._handles[key] = handle
        return handle

    def _payload_handle_for(self, payload):
        if payload is None:
            return None
        if isinstance(payload, SceneObject):
            return self._handle_for(payload)
        if isinstance(payload, Shape):
            return self._shape_handle_for(payload)
        if isinstance(payload, Material):
            return self._material_handle_for(payload)
        if isinstance(payload, Camera):
            return self._camera_handle_for(payload)
        if isinstance(payload, GeometrySourceNode):
            return self._history_handle_for(payload)
        return payload

    def _coerce_scene_objects(self, scene_objects):
        selected = []
        seen = set()
        for item in scene_objects or ():
            scene_object = self._coerce_scene_object(item)
            key = id(scene_object)
            if key in seen:
                continue
            seen.add(key)
            selected.append(scene_object)
        return tuple(selected)

    def _coerce_scene_object(self, item):
        if isinstance(item, ObjectHandle):
            item = item.raw
        if not isinstance(item, SceneObject):
            raise TypeError("Expected a SceneObject or ObjectHandle")
        if item not in tuple(walk_scene_objects(self.world)):
            raise ValueError("SceneObject is not in this session's world")
        return item

    def _coerce_shape(self, item):
        if isinstance(item, ShapeHandle):
            item = item.raw
        if not isinstance(item, Shape):
            raise TypeError("Expected a Shape or ShapeHandle")
        if item not in tuple(self._walk_shapes()):
            raise ValueError("Shape is not in this session's world")
        return item

    def _coerce_material(self, item):
        if isinstance(item, MaterialHandle):
            item = item.raw
        if not isinstance(item, Material):
            raise TypeError("Expected a Material or MaterialHandle")
        if item not in tuple(self._walk_materials()):
            raise ValueError("Material is not in this session's world")
        return item

    def _coerce_payload(self, item):
        if isinstance(item, ObjectHandle):
            item = item.raw
        elif isinstance(item, (ShapeHandle, MaterialHandle, CameraHandle, HistoryNodeHandle)):
            item = item.raw
        if item is None:
            return None
        if isinstance(item, SceneObject):
            return self._coerce_scene_object(item)
        if isinstance(item, Shape):
            return self._coerce_shape(item)
        if isinstance(item, Material):
            return self._coerce_material(item)
        if isinstance(item, Camera):
            if item not in self.world.cameras:
                raise ValueError("Camera is not in this session's world")
            return item
        if isinstance(item, GeometrySourceNode):
            return self._history_node_for_shape(item.shape)
        if isinstance(item, World) and item is self.world:
            return item
        raise TypeError("Unsupported scene payload")

    def _coerce_active(self, active, selected):
        if active is None:
            return selected[0] if len(selected) == 1 else None
        active = self._coerce_scene_object(active)
        return active if active in selected else None

    def _apply_matrix(
        self,
        scene_objects,
        matrix,
        *,
        local=False,
        relative=False,
        pivot=None,
    ):
        for scene_object in scene_objects:
            if relative:
                apply_relative_matrix(scene_object, matrix, local=local, pivot=pivot)
            else:
                apply_absolute_matrix(scene_object, matrix, local=local)

    def _walk_shapes(self):
        for scene_object in walk_scene_objects(self.world):
            for shape in scene_object.shapes:
                yield shape

    def _walk_materials(self):
        for shape in self._walk_shapes():
            for material in shape.material_groups.values():
                if material is not None:
                    yield material

    def _history_node_for_shape(self, shape):
        shape = self._coerce_shape(shape)
        geometry = shape.geometry
        key = id(shape)
        node = self._history_nodes.get(key)
        type_name = type(geometry).__name__ if geometry is not None else "None"
        name = f"{shape.name or type_name}Source"
        attrs = _geometry_attrs(geometry)
        if node is None:
            node = GeometrySourceNode(shape=shape, name=name, type_name=type_name, attrs=attrs)
            self._history_nodes[key] = node
        else:
            node.name = name
            node.type_name = type_name
            node.attrs = attrs
        return node

    def _read_attr_raw(self, target, attr_name):
        attr_name = str(attr_name)
        if isinstance(target, SceneObject):
            return getattr(target, attr_name)
        if isinstance(target, Shape):
            if attr_name == "name":
                return target.name
            return _read_geometry_attr(target.geometry, attr_name)
        if isinstance(target, Material):
            if attr_name == "name":
                return target.name
            if attr_name == "albedo":
                return target._albedo
            if attr_name in ("roughness", "ior", "intensity"):
                return getattr(target, f"_{attr_name}")
        if isinstance(target, Camera):
            return getattr(target, attr_name)
        if isinstance(target, GeometrySourceNode):
            if attr_name in ("name", "type_name"):
                return getattr(target, attr_name)
            return target.attrs.get(attr_name)
        if isinstance(target, World):
            if attr_name == "use_sky":
                return target.use_sky
            if attr_name == "background_color":
                return target.background_color
        raise ValueError(f"Unsupported attribute {attr_name!r}")

    def _coerce_attr_value(self, target, attr_name, value):
        del target, attr_name
        return value

    def _apply_attr_raw(self, target, attr_name, value):
        attr_name = str(attr_name)
        if isinstance(target, SceneObject):
            setattr(target, attr_name, value)
            return
        if isinstance(target, Shape):
            if attr_name == "name":
                target.name = value
                return
            _write_geometry_attr(target.geometry, attr_name, value)
            return
        if isinstance(target, Material):
            if attr_name == "name":
                target.name = value
            elif attr_name == "albedo":
                target._albedo = value
            elif attr_name in ("roughness", "ior", "intensity"):
                setattr(target, f"_{attr_name}", value)
            else:
                raise ValueError(f"Unsupported material attribute {attr_name!r}")
            return
        if isinstance(target, Camera):
            setattr(target, attr_name, value)
            return
        if isinstance(target, GeometrySourceNode):
            if attr_name in ("name", "type_name"):
                setattr(target, attr_name, value)
            else:
                target.attrs[attr_name] = value
            return
        if isinstance(target, World):
            if attr_name == "use_sky":
                target.use_sky = bool(value)
                return
            if attr_name == "background_color":
                target.background_color = value
                return
        raise ValueError(f"Unsupported attribute target {target!r}")

    def _material_from_plug_source(self, source):
        if isinstance(source, (Material, MaterialHandle)):
            return self._coerce_material(source)
        if isinstance(source, str):
            node_name, attr = _split_plug(source)
            if attr != "outSurface":
                raise ValueError("Only shader.outSurface connections are supported")
            return self.material(node_name).raw
        raise TypeError("Expected material handle/object or plug path")

    def _shape_group_from_plug_destination(self, destination):
        if isinstance(destination, (Shape, ShapeHandle)):
            return self._coerce_shape(destination), "default"
        if isinstance(destination, str):
            node_name, attr = _split_plug(destination)
            shape = self.shape(node_name).raw
            if attr == "surfaceShader":
                return shape, "default"
            prefix = "materialGroups."
            if attr.startswith(prefix):
                return shape, attr[len(prefix):] or "default"
            raise ValueError("Only shape.surfaceShader/materialGroups connections are supported")
        raise TypeError("Expected shape handle/object or plug path")


def _split_plug(path):
    if "." not in path:
        raise ValueError(f"Attribute path must be node.attr: {path}")
    return path.split(".", 1)


def _material_display_name(material):
    return material.name or f"{type(material).__name__}_{id(material):x}"


def _geometry_attrs(geometry):
    attrs = {}
    for public, private in (
        ("radius", "_radius"),
        ("side_length", "_side_length"),
        ("distance", "_distance"),
        ("normal", "_normal"),
        ("major_radius", "_major_radius"),
        ("minor_radius", "_minor_radius"),
    ):
        if hasattr(geometry, private):
            attrs[public] = getattr(geometry, private)
    return attrs


def _read_geometry_attr(geometry, attr_name):
    attrs = _geometry_attrs(geometry)
    if attr_name in attrs:
        return attrs[attr_name]
    raise ValueError(f"Unsupported geometry attribute {attr_name!r}")


def _write_geometry_attr(geometry, attr_name, value):
    mapping = {
        "radius": "_radius",
        "side_length": "_side_length",
        "distance": "_distance",
        "normal": "_normal",
        "major_radius": "_major_radius",
        "minor_radius": "_minor_radius",
    }
    private = mapping.get(attr_name)
    if private is None or not hasattr(geometry, private):
        raise ValueError(f"Unsupported geometry attribute {attr_name!r}")
    setattr(geometry, private, value)
    if attr_name == "side_length" and hasattr(geometry, "_bounds"):
        half = float(value) / 2.0
        geometry._bounds = AABB(Vec3(-half, -half, -half), Vec3(half, half, half))
