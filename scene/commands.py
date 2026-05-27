from __future__ import annotations

from dataclasses import dataclass, field

from scene.transform_state import restore_transforms


@dataclass
class UndoSettings:
    enabled: bool = True
    depth: int = 10


@dataclass
class SessionSettings:
    undo: UndoSettings = field(default_factory=UndoSettings)


class Command:
    affects_scene = False
    affects_selection = False
    affects_payload = False
    affects_world = False

    def execute(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class UndoStack:
    def __init__(self, settings):
        self._settings = settings
        self._undo = []
        self._redo = []

    @property
    def undo_count(self):
        return len(self._undo)

    @property
    def redo_count(self):
        return len(self._redo)

    def execute(self, command):
        command.execute()
        self._redo.clear()
        if not self._settings.enabled:
            return command

        depth = max(0, int(self._settings.depth))
        if depth == 0:
            self._undo.clear()
            return command

        self._undo.append(command)
        if len(self._undo) > depth:
            del self._undo[:len(self._undo) - depth]
        return command

    def undo(self):
        if not self._undo:
            return None
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        return command

    def redo(self):
        if not self._redo:
            return None
        command = self._redo.pop()
        command.execute()
        self._undo.append(command)
        depth = max(0, int(self._settings.depth))
        if depth == 0:
            self._undo.clear()
        elif len(self._undo) > depth:
            del self._undo[:len(self._undo) - depth]
        return command

    def clear(self):
        self._undo.clear()
        self._redo.clear()


class TransformObjectsCommand(Command):
    affects_scene = True

    def __init__(self, before, after, label="Transform"):
        self.before = dict(before)
        self.after = dict(after)
        self.label = label

    def execute(self):
        restore_transforms(self.after)

    def undo(self):
        restore_transforms(self.before)


class SelectionCommand(Command):
    affects_selection = True

    def __init__(self, session, before, before_active, after, after_active):
        self._session = session
        self.before = tuple(before)
        self.before_active = before_active
        self.after = tuple(after)
        self.after_active = after_active

    def execute(self):
        self._session._set_selection_raw(self.after, self.after_active)

    def undo(self):
        self._session._set_selection_raw(self.before, self.before_active)


class PayloadSelectionCommand(Command):
    affects_payload = True

    def __init__(self, session, before, after):
        self._session = session
        self.before = before
        self.after = after

    def execute(self):
        self._session._set_payload_selection_raw(self.after)

    def undo(self):
        self._session._set_payload_selection_raw(self.before)


class AttributeCommand(Command):
    affects_scene = True
    affects_world = True

    def __init__(self, session, target, attr_name, before, after, label="Set Attribute"):
        self._session = session
        self.target = target
        self.attr_name = attr_name
        self.before = before
        self.after = after
        self.label = label

    def execute(self):
        self._session._apply_attr_raw(self.target, self.attr_name, self.after)

    def undo(self):
        self._session._apply_attr_raw(self.target, self.attr_name, self.before)


class AssignMaterialCommand(Command):
    affects_scene = True
    affects_world = True

    def __init__(self, shape, group, before, after):
        self.shape = shape
        self.group = group
        self.before = before
        self.after = after

    def execute(self):
        self.shape._material_groups[self.group] = self.after

    def undo(self):
        if self.before is None:
            self.shape._material_groups.pop(self.group, None)
        else:
            self.shape._material_groups[self.group] = self.before


class DisconnectMaterialCommand(Command):
    affects_scene = True
    affects_world = True

    def __init__(self, shape, group, before):
        self.shape = shape
        self.group = group
        self.before = before

    def execute(self):
        self.shape._material_groups.pop(self.group, None)

    def undo(self):
        if self.before is not None:
            self.shape._material_groups[self.group] = self.before


class AddObjectsCommand(Command):
    affects_scene = True
    affects_selection = True
    affects_world = True

    def __init__(self, session, scene_objects, select=True, active=None):
        self._session = session
        self.scene_objects = tuple(scene_objects)
        self.select = bool(select)
        self.active = active
        self.before = tuple(session.selected_scene_objects())
        self.before_active = session.active_scene_object()

    def execute(self):
        for scene_object in self.scene_objects:
            if scene_object not in self._session.world.objects:
                self._session.world.add_object(scene_object)
        if self.select:
            active = self.active or (self.scene_objects[-1] if self.scene_objects else None)
            self._session._set_selection_raw(self.scene_objects, active)
        else:
            self._session._set_selection_raw(self.before, self.before_active)

    def undo(self):
        for scene_object in reversed(self.scene_objects):
            if scene_object in self._session.world.objects:
                self._session.world.remove_object(scene_object)
        self._session._set_selection_raw(self.before, self.before_active)


class VisibilityCommand(Command):
    affects_scene = True

    def __init__(self, scene_object, before, after):
        self.scene_object = scene_object
        self.before = bool(before)
        self.after = bool(after)

    def execute(self):
        self.scene_object.visible = self.after

    def undo(self):
        self.scene_object.visible = self.before
