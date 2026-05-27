from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from core import Color, Vec3
from scene import Camera, SceneObject
from scene.history import GeometrySourceNode
from scene.materials import Dielectric, Emissive, Glossy, Material, Metal
from scene.shape import Shape
from scene.world import World


class ColorSwatchButton(QtWidgets.QPushButton):
    """Clickable color field that edits a Color through QColorDialog."""

    colorChanged = QtCore.Signal(object)

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self._color = _coerce_color(color)
        self.setMinimumHeight(24)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._choose_color)
        self._refresh()

    @property
    def color(self):
        return self._color

    def set_color(self, color, emit=False):
        self._color = _coerce_color(color)
        self._refresh()
        if emit:
            self.colorChanged.emit(self._color)

    def _choose_color(self):
        selected = QtWidgets.QColorDialog.getColor(
            _to_qcolor(self._color),
            self,
            "Choose Color",
        )
        if selected.isValid():
            self.set_color(_from_qcolor(selected), emit=True)

    def _refresh(self):
        hex_color = _color_hex(self._color)
        self.setText(hex_color)
        text_color = "#111111" if _luminance(self._color) > 0.55 else "#ffffff"
        self.setStyleSheet(
            "QPushButton {"
            f" background-color: {hex_color};"
            f" color: {text_color};"
            " border: 1px solid rgba(255, 255, 255, 80);"
            " border-radius: 3px;"
            " padding: 3px 8px;"
            " text-align: center;"
            "}"
        )


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
                for spec in _material_parameter_specs(payload):
                    attr_name = spec["attr"]
                    value = spec["value"]
                    label = spec.get("label")
                    if spec["type"] == "color":
                        self._add_color(attr_name, value, label=label)
                    elif spec["type"] == "float":
                        self._add_float(
                            attr_name,
                            value,
                            minimum=spec.get("minimum", -1_000_000.0),
                            maximum=spec.get("maximum", 1_000_000.0),
                            label=label,
                        )
                    elif spec["type"] == "bool":
                        self._add_bool(attr_name, value, label=label)
                    elif spec["type"] == "texture":
                        self._add_text(attr_name, _texture_path(value), label=label)
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
                self._add_text("name", payload.name)
                self._add_bool("use_sky", payload.use_sky)
                self._add_color("background_color", payload.background_color)
        finally:
            self._building = False

    def _add_text(self, attr_name, value, label=None):
        widget = QtWidgets.QLineEdit(str(value or ""))
        widget.setObjectName(f"attr_{attr_name}")
        widget.editingFinished.connect(
            lambda name=attr_name, w=widget: self._set_attr(name, w.text())
        )
        self._widgets[attr_name] = widget
        self._form.addRow(label or _label(attr_name), widget)

    def _add_readonly(self, attr_name, value):
        widget = QtWidgets.QLineEdit(str(value or ""))
        widget.setReadOnly(True)
        widget.setObjectName(f"attr_{attr_name}")
        self._widgets[attr_name] = widget
        self._form.addRow(_label(attr_name), widget)

    def _add_bool(self, attr_name, value, label=None):
        widget = QtWidgets.QCheckBox()
        widget.setObjectName(f"attr_{attr_name}")
        widget.setChecked(bool(value))
        widget.toggled.connect(lambda checked, name=attr_name: self._set_attr(name, bool(checked)))
        self._widgets[attr_name] = widget
        self._form.addRow(label or _label(attr_name), widget)

    def _add_float(
        self,
        attr_name,
        value,
        minimum=-1_000_000.0,
        maximum=1_000_000.0,
        label=None,
    ):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setObjectName(f"attr_{attr_name}")
        widget.setDecimals(4)
        widget.setRange(float(minimum), float(maximum))
        widget.setValue(float(value))
        widget.valueChanged.connect(lambda new, name=attr_name: self._set_attr(name, float(new)))
        self._widgets[attr_name] = widget
        self._form.addRow(label or _label(attr_name), widget)

    def _add_color(self, attr_name, value, label=None):
        widget = ColorSwatchButton(value)
        widget.setObjectName(f"attr_{attr_name}")
        widget.colorChanged.connect(lambda color, name=attr_name: self._set_attr(name, color))
        self._widgets[attr_name] = widget
        self._form.addRow(label or _label(attr_name), widget)

    def _add_vec3(self, attr_name, value, color=False):
        if color:
            self._add_color(attr_name, value)
            return

        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        components = ("x", "y", "z")
        spinboxes = []
        for component in components:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-1_000_000.0, 1_000_000.0)
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


def _material_parameter_specs(material):
    specs = [
        {
            "attr": "albedo",
            "label": "Color" if isinstance(material, Emissive) else "Albedo",
            "type": "color",
            "value": material._albedo,
        },
        {
            "attr": "albedo_texture",
            "label": "Albedo Texture",
            "type": "texture",
            "value": material._albedo_texture,
        },
    ]

    if material._albedo_texture is not None:
        specs.append({
            "attr": "albedo_texture_flip_v",
            "label": "Flip Texture V",
            "type": "bool",
            "value": material._albedo_texture.flip_v,
        })

    if isinstance(material, (Metal, Glossy)):
        specs.append({
            "attr": "roughness",
            "type": "float",
            "value": material._roughness,
            "minimum": 0.0,
            "maximum": 1.0,
        })
    if isinstance(material, Dielectric):
        specs.append({
            "attr": "ior",
            "label": "IOR",
            "type": "float",
            "value": material._ior,
            "minimum": 1.0,
        })
    if isinstance(material, Emissive):
        specs.append({
            "attr": "intensity",
            "type": "float",
            "value": material._intensity,
            "minimum": 0.0,
        })
    return specs


def _texture_path(texture):
    return "" if texture is None else getattr(texture, "path", "") or ""


def _coerce_color(color):
    return Color(float(color.r), float(color.g), float(color.b))


def _to_qcolor(color):
    return QtGui.QColor.fromRgbF(
        _clamp01(color.r),
        _clamp01(color.g),
        _clamp01(color.b),
    )


def _from_qcolor(color):
    return Color(color.redF(), color.greenF(), color.blueF())


def _color_hex(color):
    return _to_qcolor(color).name(QtGui.QColor.NameFormat.HexRgb)


def _luminance(color):
    return (
        0.2126 * _clamp01(color.r)
        + 0.7152 * _clamp01(color.g)
        + 0.0722 * _clamp01(color.b)
    )


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


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
