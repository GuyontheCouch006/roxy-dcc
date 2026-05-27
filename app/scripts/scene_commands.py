from __future__ import annotations

from pathlib import Path

from core import Color, Vec3
from scene import Camera, Cube, Diffuse, Emissive, Plane, SceneObject, Sphere, Torus
from scene.io import load_scene
from scene.io.obj_reader import OBJReader
from scene.world import World


DEFAULT_LAMBERT_COLOR = Color(0.7, 0.7, 0.7)


def new_scene():
    """Return a fresh world for File > New."""
    world = World(use_sky=False)
    world.add_camera(Camera(name="camera1"))
    return world


def create_primitive(
    session,
    primitive_type,
    *,
    name=None,
    x=0.0,
    y=0.0,
    z=0.0,
    radius=1.0,
    side_length=1.0,
    distance=0.0,
    major_radius=1.0,
    minor_radius=0.25,
):
    primitive_type = primitive_type.lower()
    geometry = _primitive_geometry(
        primitive_type,
        radius=radius,
        side_length=side_length,
        distance=distance,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    obj = SceneObject(
        shape=geometry,
        material=Diffuse(DEFAULT_LAMBERT_COLOR),
        translation=Vec3(x, y, z),
        name=name or _unique_name(session.world, primitive_type),
        renderable=(primitive_type != "torus"),
    )
    return session.add_object(obj, select=True)


def create_light(
    session,
    light_type="sphere",
    *,
    name=None,
    x=0.0,
    y=4.0,
    z=0.0,
    radius=0.5,
    side_length=1.0,
    intensity=20.0,
    color=(1.0, 0.92, 0.76),
):
    light_type = light_type.lower()
    if light_type in ("sphere", "sphere_light", "point"):
        geometry = Sphere(radius)
        label = "sphereLight"
    elif light_type in ("area", "area_light", "panel"):
        geometry = Cube(side_length)
        label = "areaLight"
    else:
        raise ValueError(f"Unsupported light type: {light_type}")

    obj = SceneObject(
        shape=geometry,
        material=Emissive(_color(color), intensity=intensity),
        translation=Vec3(x, y, z),
        name=name or _unique_name(session.world, label),
    )
    return session.add_object(obj, select=True)


def import_scene(session, path):
    """Import scene or OBJ contents into the current session."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        obj = _object_from_obj(path)
        return session.add_object(obj, select=True)

    imported = load_scene(path)
    for camera in imported.cameras:
        session.world.add_camera(camera)
    handles = session.add_objects(imported.objects, select=bool(imported.objects))
    if imported.cameras and not imported.objects:
        session.notify_world_changed()
    return tuple(handles)


def reference_scene(session, path):
    """Reference imported scene objects under a single root transform."""
    path = Path(path)
    root = SceneObject(name=_unique_name(session.world, f"{path.stem}_ref"))
    root.reference_path = str(path)

    if path.suffix.lower() == ".obj":
        root.add_child(_object_from_obj(path))
    else:
        imported = load_scene(path)
        for obj in imported.objects:
            root.add_child(obj)

    return session.add_object(root, select=True)


def _object_from_obj(path):
    mesh = OBJReader.load_as_indexed_mesh(path)
    return SceneObject(
        shape=mesh,
        material=Diffuse(DEFAULT_LAMBERT_COLOR),
        name=path.stem,
    )


def _primitive_geometry(
    primitive_type,
    *,
    radius,
    side_length,
    distance,
    major_radius,
    minor_radius,
):
    if primitive_type == "sphere":
        return Sphere(radius)
    if primitive_type == "cube":
        return Cube(side_length)
    if primitive_type == "plane":
        return Plane(distance=distance)
    if primitive_type == "torus":
        return Torus(major_radius=major_radius, minor_radius=minor_radius)
    raise ValueError(f"Unsupported primitive type: {primitive_type}")


def _color(value):
    if isinstance(value, Color):
        return value
    return Color(float(value[0]), float(value[1]), float(value[2]))


def _unique_name(world, base):
    existing = {obj.name for obj in _walk_objects(world.objects)}
    if base not in existing:
        return base
    index = 1
    while f"{base}{index}" in existing:
        index += 1
    return f"{base}{index}"


def _walk_objects(objects):
    for obj in objects:
        yield obj
        yield from _walk_objects(obj.children)
