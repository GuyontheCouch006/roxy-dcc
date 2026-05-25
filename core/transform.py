# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Transform encapsulates a scene object's TRS (translation, rotation,
#              scale), shear, and pivot. Lazily rebuilds its world matrix and inverse
#              when any component changes.
# ============================================

from core.vectors import Vec3
from core.matrix import Mat4x4
from core.aabb import AABB
from core.enums import RotationOrder


class Transform:
    """Scene-object transform with lazy matrix rebuild on property change."""

    def __init__(
        self,
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        pivot=None,
        matrix=None,
        inverse_matrix=None,
        rotation_order=RotationOrder.XYZ,
        shape=None,
    ):
        self._translation = translation if translation is not None else Vec3(0, 0, 0)
        self._rotation = rotation if rotation is not None else Vec3(0, 0, 0)
        self._scale = scale if scale is not None else Vec3(1, 1, 1)
        self._shear = shear if shear is not None else Vec3(0, 0, 0)
        self._pivot = pivot if pivot is not None else Vec3(0, 0, 0)
        self._rotation_order = rotation_order
        self._shape = shape
        self._matrix_mode = matrix is not None

        if self._matrix_mode:
            self._matrix = self._coerce_matrix(matrix)
            self._inverse_matrix = (
                self._coerce_matrix(inverse_matrix)
                if inverse_matrix is not None
                else self._matrix.inverse()
            )
        else:
            self._matrix = Mat4x4.from_trs(
                self._translation, self._rotation, self._scale,
                self._shear, self._pivot, self._rotation_order,
            )
            self._inverse_matrix = Mat4x4.inverse_trs(
                self._translation, self._rotation, self._scale,
                self._shear, self._pivot, self._rotation_order,
            )
        self.dirty = False  # Tracks whether matrices need to be rebuilt.

        self._update_world_aabb()

    @staticmethod
    def _coerce_matrix(value):
        if isinstance(value, Mat4x4):
            rows = value.rows
        else:
            rows = value
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError("Matrix transforms must be 4x4")
        return Mat4x4([[float(v) for v in row] for row in rows])

    def _update_world_aabb(self):
        if not self._shape:
            # Default to a unit cube; replaced when a shape is assigned.
            self._world_aabb = AABB(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5))
        elif self._shape.is_infinite:
            # Infinite shapes have no bounds; use an empty AABB or skip intersection tests.
            self._world_aabb = AABB(Vec3(0, 0, 0), Vec3(0, 0, 0))
        else:
            self._world_aabb = self._shape.local_bounds().transform(self._matrix)

    def _rebuild(self):
        if not self._matrix_mode:
            self._matrix = Mat4x4.from_trs(
                self._translation, self._rotation, self._scale,
                self._shear, self._pivot, self._rotation_order,
            )
            self._inverse_matrix = Mat4x4.inverse_trs(
                self._translation, self._rotation, self._scale,
                self._shear, self._pivot, self._rotation_order,
            )
        self._update_world_aabb()
        self.dirty = False

    def _mark_components_dirty(self):
        self._matrix_mode = False
        self.dirty = True

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = value
        self._mark_components_dirty()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self._mark_components_dirty()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self._mark_components_dirty()

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, value):
        self._shear = value
        self._mark_components_dirty()

    @property
    def pivot(self):
        return self._pivot

    @pivot.setter
    def pivot(self, value):
        self._pivot = value
        self._mark_components_dirty()

    @property
    def rotation_order(self):
        return self._rotation_order

    @rotation_order.setter
    def rotation_order(self, value):
        self._rotation_order = value
        self._mark_components_dirty()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        if self._matrix_mode:
            self._update_world_aabb()
            self.dirty = False
        else:
            self.dirty = True

    @property
    def matrix_mode(self):
        return self._matrix_mode

    @property
    def matrix(self):
        if self.dirty:
            self._rebuild()
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self.set_matrix(value)

    def set_matrix(self, matrix, inverse_matrix=None):
        self._matrix = self._coerce_matrix(matrix)
        self._inverse_matrix = (
            self._coerce_matrix(inverse_matrix)
            if inverse_matrix is not None
            else self._matrix.inverse()
        )
        self._matrix_mode = True
        self._update_world_aabb()
        self.dirty = False

    # ─── Computed outputs (trigger rebuild if dirty) ───────────────────────────

    @property
    def world_matrix(self):
        if self.dirty:
            self._rebuild()
        return self._matrix

    @property
    def world_inverse_matrix(self):
        if self.dirty:
            self._rebuild()
        return self._inverse_matrix

    @property
    def world_aabb(self):
        if self.dirty:
            self._rebuild()
        return self._world_aabb

    def __repr__(self):
        return (f"Transform(t={self._translation}, r={self._rotation}, "
                f"s={self._scale}, order={self._rotation_order})")

    def to_dict(self):
        data = {
            "mode": "matrix" if self._matrix_mode else "components",
            "translation": self._translation.to_dict(),
            "rotation": self._rotation.to_dict(),
            "scale": self._scale.to_dict(),
            "shear": self._shear.to_dict(),
            "pivot": self._pivot.to_dict(),
            "rotation_order": self._rotation_order.value,  # enum to int
        }
        if self._matrix_mode:
            data["matrix"] = [row[:] for row in self._matrix.rows]
            data["inverse_matrix"] = [row[:] for row in self._inverse_matrix.rows]
        return data

    @classmethod
    def from_dict(cls, data, shape=None):
        if data.get("mode") == "matrix" or "matrix" in data:
            return cls(
                matrix=data["matrix"],
                inverse_matrix=data.get("inverse_matrix"),
                translation=Vec3.from_dict(data.get("translation", [0, 0, 0])),
                rotation=Vec3.from_dict(data.get("rotation", [0, 0, 0])),
                scale=Vec3.from_dict(data.get("scale", [1, 1, 1])),
                shear=Vec3.from_dict(data.get("shear", [0, 0, 0])),
                pivot=Vec3.from_dict(data.get("pivot", [0, 0, 0])),
                rotation_order=RotationOrder(data.get("rotation_order", RotationOrder.XYZ.value)),
                shape=shape,
            )
        return cls(
            translation=Vec3.from_dict(data["translation"]),
            rotation=Vec3.from_dict(data["rotation"]),
            scale=Vec3.from_dict(data["scale"]),
            shear=Vec3.from_dict(data["shear"]),
            pivot=Vec3.from_dict(data["pivot"]),
            rotation_order=RotationOrder(data["rotation_order"]),
            shape=shape,
        )
