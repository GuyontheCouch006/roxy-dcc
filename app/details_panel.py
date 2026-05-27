from __future__ import annotations

from PySide6 import QtWidgets

from core import Color, Vec3
from scene import Camera, SceneObject
from scene.history import GeometrySourceNode
from scene.materials import Dielectric, Emissive, Glossy, Material, Metal
from scene.shape import Shape
from scene.world import World


class DetailsPanel(QtWidgets.QWidget):
    """Editable attribute view driven by SceneSession handles."""

    def __init__(self, parent=None, session=None):
        super().__init__(parent)
        self._session = None
        self._payload = None
        self._widgets = {}
        self._building = False

        self._title = QtWidgets.QLabel("No Selection")
        self._title.setObjectName("detailsTitle")
        self._form_widget = QtWidgets.QWidget()
        self._form = QtWidgets.QFormLayout(self._form_widget)
        self._form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._title)
        layout.addWidget(scroll)

        if session is not None:
            self.set_session(session)

    @property
    def session(self):
        return self._session

    @property
    def selected_payload(self):
        return self._payload

    def widget_for_attr(self, attr_name):
        return self._widgets.get(attr_name)

    def set_session(self, session):
        if self._session is session:
            return
        if self._session is not None:
            self._session.remove_payload_listener(self._on_payload_changed)
            self._session.remove_scene_listener(self._on_scene_changed)
        self._session = session
        if session is not None:
            session.add_payload_listener(self._on_payload_changed)
            session.add_scene_listener(self._on_scene_changed)
            self.set_payload(session.selected_raw_payload())

    def set_payload(self, payload):
        self._payload = payload
        self._rebuild()

    def _on_payload_changed(self, session):
        self.set_payload(session.selected_raw_payload())

    def _on_scene_changed(self, session):
        del session
        self._rebuild()

    def _rebuild(self):
        self._building = True
        try:
            self._widgets.clear()
            while self._form.rowCount():
                self._form.removeRow(0)

            payload = self._payload
            self._title.setText(_payload_title(payload))
            if payload is None or self._session is None:
                return

            if isinstance(payload, SceneObject):
                self._add_text("name", payload.name)
                self._add_bool("visible", payload.visible)
                self._add_bool("renderable", payload.renderable)
                self._add_bool("selectable", payload.selectable)
                self._add_vec3("translation", payload.translation)
                self._add_vec3("rotation", payload.rotation)
                self._add_vec3("scale", payload.scale)
            elif isinstance(payload, Shape):
                self._add_text("name", payload.name)
                for name, value in _geometry_attrs(payload.geometry).items():
                    if isinstance(value, Vec3):
                        self._add_vec3(name, value)
                    else:
                        self._add_float(name, value, minimum=-1_000_000.0)
            elif isinstance(payload, Material):
                self._add_text("name", payload.name)
                self._add_vec3("albedo", payload._albedo, color=True)
                if isinstance(payload, (Metal, Glossy)):
                    self._add_float("roughness", payload._roughness, minimum=0.0, maximum=1.0)
                if isinstance(payload, Dielectric):
                    self._add_float("ior", payload._ior, minimum=1.0)
                if isinstance(payload, Emissive):
                    self._add_float("intensity", payload._intensity, minimum=0.0)
            elif isinstance(payload, Camera):
                self._add_text("name", payload.name)
                self._add_vec3("position", payload.position)
                self._add_vec3("forward", payload.forward)
                self._add_vec3("up", payload.up)
                self._add_float("fov", payload.fov, minimum=1.0, maximum=179.0)
            elif isinstance(payload, GeometrySourceNode):
                self._add_readonly("name", payload.name)
                self._add_readonly("type", payload.type_name)
            elif isinstance(payload, World):
                self._add_bool("use_sky", payload.use_sky)
                self._add_vec3("background_color", payload.background_color, color=True)
        finally:
            self._building = False

    def _add_text(self, attr_name, value):
        widget = QtWidgets.QLineEdit(str(value or ""))
        widget.setObjectName(f"attr_{attr_name}")
        widget.editingFinished.connect(
            lambda name=attr_name, w=widget: self._set_attr(name, w.text())
        )
        self._widgets[attr_name] = widget
        self._form.addRow(_label(attr_name), widget)

    def _add_readonly(self, attr_name, value):
        widget = QtWidgets.QLineEdit(str(value or ""))
        widget.setReadOnly(True)
        widget.setObjectName(f"attr_{attr_name}")
        self._widgets[attr_name] = widget
        self._form.addRow(_label(attr_name), widget)

    def _add_bool(self, attr_name, value):
        widget = QtWidgets.QCheckBox()
        widget.setObjectName(f"attr_{attr_name}")
        widget.setChecked(bool(value))
        widget.toggled.connect(lambda checked, name=attr_name: self._set_attr(name, bool(checked)))
        self._widgets[attr_name] = widget
        self._form.addRow(_label(attr_name), widget)

    def _add_float(self, attr_name, value, minimum=-1_000_000.0, maximum=1_000_000.0):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setObjectName(f"attr_{attr_name}")
        widget.setDecimals(4)
        widget.setRange(float(minimum), float(maximum))
        widget.setValue(float(value))
        widget.valueChanged.connect(lambda new, name=attr_name: self._set_attr(name, float(new)))
        self._widgets[attr_name] = widget
        self._form.addRow(_label(attr_name), widget)

    def _add_vec3(self, attr_name, value, color=False):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        components = ("r", "g", "b") if color else ("x", "y", "z")
        spinboxes = []
        for component in components:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.0 if color else -1_000_000.0, 1.0 if color else 1_000_000.0)
            spin.setValue(float(getattr(value, component)))
            spin.setObjectName(f"attr_{attr_name}_{component}")
            layout.addWidget(spin)
            spinboxes.append(spin)
        for spin in spinboxes:
            spin.valueChanged.connect(
                lambda _new, name=attr_name, boxes=spinboxes, is_color=color:
                    self._set_attr(name, _value_from_spinboxes(boxes, is_color))
            )
        self._widgets[attr_name] = row
        self._form.addRow(_label(attr_name), row)

    def _set_attr(self, attr_name, value):
        if self._building or self._session is None or self._payload is None:
            return
        self._session.set_attr(self._payload, attr_name, value)


def _payload_title(payload):
    if payload is None:
        return "No Selection"
    return f"{type(payload).__name__}: {getattr(payload, 'name', '') or type(payload).__name__}"


def _label(attr_name):
    return str(attr_name).replace("_", " ").title()


def _value_from_spinboxes(spinboxes, is_color):
    values = [box.value() for box in spinboxes]
    if is_color:
        return Color(*values)
    return Vec3(*values)


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
