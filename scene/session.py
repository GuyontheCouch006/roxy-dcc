from __future__ import annotations

from core import Vec3
from scene.commands import (
    AddObjectsCommand,
    SelectionCommand,
    SessionSettings,
    TransformObjectsCommand,
    UndoStack,
    VisibilityCommand,
)
from scene.object_handle import ObjectHandle
from scene.scene_object import SceneObject
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
        self._selection_listeners = []
        self._scene_listeners = []
        self._preview = None
        self._world_listeners = []

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
        self._selected = ()
        self._active = None
        self._undo_stack.clear()
        self._notify_world_changed()
        self._notify_selection_changed()

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

    def selected(self):
        return tuple(self._handle_for(scene_object) for scene_object in self._selected)

    def selected_objects(self):
        return self.selected()

    def selected_scene_objects(self):
        return self._selected

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
        if command.affects_scene:
            self._notify_scene_changed()

    def _notify_selection_changed(self):
        for callback in tuple(self._selection_listeners):
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
