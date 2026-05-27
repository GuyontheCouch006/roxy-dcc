from __future__ import annotations

import re

from scene.camera import Camera
from scene.materials import Material
from scene.scene_object import SceneObject
from scene.shape import Shape
from scene.world import World


_NUMERIC_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


def ensure_scene_node_names(world, extra_objects=(), extra_cameras=(), extra_lights=()):
    """Assign stable, unique names to all script-addressable scene nodes."""
    used = set()
    seen = set()

    if world is not None:
        _assign_name(world, default_node_name(world), used)

    for obj in tuple(_world_objects(world)) + tuple(extra_objects or ()):
        _assign_object_names(obj, used, seen)

    material_seen = set()
    for obj in _walk_unique_objects(_world_objects(world), extra_objects, set()):
        for shape in getattr(obj, "shapes", ()):
            for material in getattr(shape, "material_groups", {}).values():
                if material is not None and id(material) not in material_seen:
                    material_seen.add(id(material))
                    _assign_name(material, default_node_name(material), used)

    for camera in _walk_unique_items(_world_cameras(world), extra_cameras):
        _assign_name(camera, default_node_name(camera), used)

    for light in _walk_unique_items(_world_lights(world), extra_lights):
        if hasattr(light, "name"):
            _assign_name(light, "light", used)


def default_node_name(node, owner=None, index=0):
    if isinstance(node, World):
        return "world"
    if isinstance(node, SceneObject):
        return "transform"
    if isinstance(node, Shape):
        owner_name = clean_node_name(getattr(owner, "name", "")) if owner else ""
        if owner_name:
            return f"{owner_name}Shape"
        geometry = getattr(node, "geometry", None)
        geometry_name = type(geometry).__name__ if geometry is not None else "shape"
        suffix = "" if index == 0 else str(index + 1)
        return f"{geometry_name[:1].lower()}{geometry_name[1:]}Shape{suffix}"
    if isinstance(node, Material):
        type_name = type(node).__name__
        return f"{type_name[:1].lower()}{type_name[1:]}"
    if isinstance(node, Camera):
        return "camera"
    return "node"


def clean_node_name(name):
    return str(name).strip() if name is not None else ""


def unique_node_name(base, used):
    base = clean_node_name(base) or "node"
    if base not in used:
        return base

    match = _NUMERIC_SUFFIX_RE.match(base)
    if match:
        prefix = match.group("prefix")
        number = int(match.group("number")) + 1
        width = len(match.group("number"))
    else:
        prefix = base
        number = 1
        width = 0

    while True:
        suffix = f"{number:0{width}d}" if width else str(number)
        candidate = f"{prefix}{suffix}"
        if candidate not in used:
            return candidate
        number += 1


def _assign_object_names(obj, used, seen):
    key = id(obj)
    if key in seen:
        return
    seen.add(key)

    _assign_name(obj, default_node_name(obj), used)

    for child in getattr(obj, "children", ()):
        _assign_object_names(child, used, seen)

    for index, shape in enumerate(getattr(obj, "shapes", ())):
        current = clean_node_name(getattr(shape, "name", ""))
        default = default_node_name(shape, owner=obj, index=index)
        if not current or current == obj.name or current in used:
            shape.name = ""
        _assign_name(shape, default, used)


def _assign_name(node, default, used):
    current = clean_node_name(getattr(node, "name", ""))
    name = unique_node_name(current or default, used)
    if current != name:
        setattr(node, "name", name)
    used.add(name)
    return name


def _world_objects(world):
    return getattr(world, "objects", ()) if world is not None else ()


def _world_cameras(world):
    return getattr(world, "cameras", ()) if world is not None else ()


def _world_lights(world):
    return getattr(world, "lights", ()) if world is not None else ()


def _walk_unique_objects(objects, extra_objects, seen):
    for obj in tuple(objects or ()) + tuple(extra_objects or ()):
        yield from _walk_unique_object(obj, seen)


def _walk_unique_object(obj, seen):
    key = id(obj)
    if key in seen:
        return
    seen.add(key)
    yield obj
    for child in getattr(obj, "children", ()):
        yield from _walk_unique_object(child, seen)


def _walk_unique_items(items, extra_items):
    seen = set()
    for item in tuple(items or ()) + tuple(extra_items or ()):
        key = id(item)
        if key in seen:
            continue
        seen.add(key)
        yield item
