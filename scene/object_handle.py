from __future__ import annotations

from core import Vec3
from scene import transform_ops


class ObjectHandle:
    """Script-facing wrapper around a raw SceneObject."""

    def __init__(self, session, scene_object):
        self._session = session
        self._scene_object = scene_object

    @property
    def raw(self):
        return self._scene_object

    @property
    def scene_object(self):
        return self._scene_object

    @property
    def name(self):
        return self._scene_object.name

    def move(self, x=0.0, y=0.0, z=0.0, local=False):
        delta = _coerce_xyz(x, y, z, default=0.0)
        self._session.move_objects(
            (self._scene_object,),
            x=delta.x,
            y=delta.y,
            z=delta.z,
            local=local,
            label=f"Move {self.name}",
        )
        return self

    def rotate(self, x=0.0, y=0.0, z=0.0, local=False, pivot=None):
        rotation = _coerce_xyz(x, y, z, default=0.0)
        self._session.rotate_objects(
            (self._scene_object,),
            x=rotation.x,
            y=rotation.y,
            z=rotation.z,
            local=local,
            pivot=pivot,
            label=f"Rotate {self.name}",
        )
        return self

    def scale(self, x=1.0, y=1.0, z=1.0, local=False, pivot=None):
        scale = _coerce_xyz(x, y, z, default=1.0)
        self._session.scale_objects(
            (self._scene_object,),
            x=scale.x,
            y=scale.y,
            z=scale.z,
            local=local,
            pivot=pivot,
            label=f"Scale {self.name}",
        )
        return self

    def transform(
        self,
        matrix=None,
        translate=None,
        translation=None,
        rotate=None,
        rotation=None,
        scale=None,
        local=False,
    ):
        if translate is not None and translation is not None:
            raise ValueError("use either translate or translation, not both")
        if rotate is not None and rotation is not None:
            raise ValueError("use either rotate or rotation, not both")
        if translation is not None:
            translate = translation
        if rotation is not None:
            rotate = rotation
        has_trs = translate is not None or rotate is not None or scale is not None
        if matrix is not None and has_trs:
            raise ValueError("matrix input cannot be mixed with TRS inputs")
        if matrix is None:
            if local:
                self._session.set_local_trs_objects(
                    (self._scene_object,),
                    translate=transform_ops.coerce_vec3(translate),
                    rotate=transform_ops.coerce_vec3(rotate),
                    scale=transform_ops.coerce_vec3(scale, x=1.0, y=1.0, z=1.0),
                    label=f"Transform {self.name}",
                )
                return self
            matrix = transform_ops.trs_matrix(
                translate=translate,
                rotate=rotate,
                scale=scale,
            )
        self._session.transform_objects(
            (self._scene_object,),
            matrix=matrix,
            local=local,
            relative=False,
            label=f"Transform {self.name}",
        )
        return self

    def __repr__(self):
        return f"ObjectHandle({self._scene_object!r})"


def _coerce_xyz(x, y, z, *, default):
    if isinstance(x, Vec3) and y == default and z == default:
        return x
    if isinstance(x, (tuple, list)) and y == default and z == default:
        return transform_ops.coerce_vec3(x)
    return Vec3(float(x), float(y), float(z))
