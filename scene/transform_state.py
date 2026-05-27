from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from core import Transform


@dataclass(frozen=True)
class TransformSnapshot:
    """Local transform state captured for undo/redo."""

    data: dict

    @classmethod
    def capture(cls, scene_object):
        return cls(deepcopy(scene_object.transform.to_dict()))

    def restore(self, scene_object):
        shape = scene_object.shapes[0].geometry if scene_object.shapes else None
        scene_object._transform = Transform.from_dict(deepcopy(self.data), shape=shape)


def capture_transforms(scene_objects):
    return {
        scene_object: TransformSnapshot.capture(scene_object)
        for scene_object in scene_objects
    }


def restore_transforms(snapshots):
    for scene_object, snapshot in snapshots.items():
        snapshot.restore(scene_object)
