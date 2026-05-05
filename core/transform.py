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

        self._matrix = Mat4x4.from_trs(
            self._translation, self._rotation, self._scale,
            self._shear, self._pivot, self._rotation_order,
        )
        self._inverse_matrix = Mat4x4.inverse_trs(
            self._translation, self._rotation, self._scale,
            self._shear, self._pivot, self._rotation_order,
        )
        self.dirty = False  # Tracks whether matrices need to be rebuilt.

        if not shape:
            # Default to a unit cube; replaced when a shape is assigned.
            self._world_aabb = AABB(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5))
        elif shape.is_infinite:
            # Infinite shapes have no bounds; use an empty AABB or skip intersection tests.
            self._world_aabb = AABB(Vec3(0, 0, 0), Vec3(0, 0, 0))
        else:
            self._world_aabb = shape.local_bounds().transform(self._matrix)

    def _rebuild(self):
        self._matrix = Mat4x4.from_trs(
            self._translation, self._rotation, self._scale,
            self._shear, self._pivot, self._rotation_order,
        )
        self._inverse_matrix = Mat4x4.inverse_trs(
            self._translation, self._rotation, self._scale,
            self._shear, self._pivot, self._rotation_order,
        )
        if self._shape and not self._shape.is_infinite:
            self._world_aabb = self._shape.local_bounds().transform(self._matrix)

        self.dirty = False

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = value
        self.dirty = True

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.dirty = True

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self.dirty = True

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, value):
        self._shear = value
        self.dirty = True

    @property
    def pivot(self):
        return self._pivot

    @pivot.setter
    def pivot(self, value):
        self._pivot = value
        self.dirty = True

    @property
    def rotation_order(self):
        return self._rotation_order

    @rotation_order.setter
    def rotation_order(self, value):
        self._rotation_order = value
        self.dirty = True

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self.dirty = True

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
        return {
            "translation": self._translation.to_dict(),
            "rotation": self._rotation.to_dict(),
            "scale": self._scale.to_dict(),
            "shear": self._shear.to_dict(),
            "pivot": self._pivot.to_dict(),
            "rotation_order": self._rotation_order.value,  # enum to int
        }

    @classmethod
    def from_dict(cls, data, shape=None):
        return cls(
            translation=Vec3.from_dict(data["translation"]),
            rotation=Vec3.from_dict(data["rotation"]),
            scale=Vec3.from_dict(data["scale"]),
            shear=Vec3.from_dict(data["shear"]),
            pivot=Vec3.from_dict(data["pivot"]),
            rotation_order=RotationOrder(data["rotation_order"]),
            shape=shape,
        )
