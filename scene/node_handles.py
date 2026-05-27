from __future__ import annotations


class ScenePayloadHandle:
    """Script-facing wrapper around a non-transform scene payload."""

    kind = "payload"

    def __init__(self, session, raw):
        self._session = session
        self._raw = raw

    @property
    def raw(self):
        return self._raw

    @property
    def name(self):
        return getattr(self._raw, "name", "")

    def set_attr(self, name, value):
        self._session.set_attr(self, name, value)
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self._raw!r})"


class ShapeHandle(ScenePayloadHandle):
    kind = "shape"

    @property
    def shape(self):
        return self._raw

    def assign_material(self, material, group="default"):
        self._session.assign_material(self, material, group=group)
        return self

    def connect_shader(self, material, group="default"):
        return self.assign_material(material, group=group)

    def disconnect_shader(self, group="default"):
        self._session.disconnect_material(self, group=group)
        return self


class MaterialHandle(ScenePayloadHandle):
    kind = "material"

    @property
    def material(self):
        return self._raw


class CameraHandle(ScenePayloadHandle):
    kind = "camera"

    @property
    def camera(self):
        return self._raw


class HistoryNodeHandle(ScenePayloadHandle):
    kind = "history"

    @property
    def history_node(self):
        return self._raw


class WorldHandle(ScenePayloadHandle):
    kind = "world"

    @property
    def world(self):
        return self._raw
