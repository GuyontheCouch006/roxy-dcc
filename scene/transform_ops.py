from __future__ import annotations

from core import Mat4x4, Vec3


def coerce_vec3(value=None, *, x=0.0, y=0.0, z=0.0):
    if value is None:
        return Vec3(float(x), float(y), float(z))
    if isinstance(value, Vec3):
        return value
    return Vec3(float(value[0]), float(value[1]), float(value[2]))


def coerce_matrix(value):
    if isinstance(value, Mat4x4):
        return copy_matrix(value)
    return Mat4x4([[float(item) for item in row] for row in value])


def copy_matrix(matrix):
    return Mat4x4([row[:] for row in matrix.rows])


def trs_matrix(translate=None, rotate=None, scale=None):
    return Mat4x4.from_trs(
        coerce_vec3(translate),
        coerce_vec3(rotate),
        coerce_vec3(scale, x=1.0, y=1.0, z=1.0),
    )


def translation_matrix(x=0.0, y=0.0, z=0.0):
    return Mat4x4.translation(float(x), float(y), float(z))


def rotation_matrix(x=0.0, y=0.0, z=0.0):
    return trs_matrix(rotate=Vec3(float(x), float(y), float(z)))


def scale_matrix(x=1.0, y=1.0, z=1.0):
    return Mat4x4.scale(float(x), float(y), float(z))


def matrix_around_pivot(matrix, pivot):
    pivot = coerce_vec3(pivot)
    return (
        Mat4x4.translation(pivot.x, pivot.y, pivot.z)
        * matrix
        * Mat4x4.translation(-pivot.x, -pivot.y, -pivot.z)
    )


def local_matrix_from_world(scene_object, world_matrix):
    if scene_object.parent is not None:
        return scene_object.parent.world_inverse_matrix * world_matrix
    return world_matrix


def world_matrix_from_local(scene_object, local_matrix):
    if scene_object.parent is not None:
        return scene_object.parent.world_matrix * local_matrix
    return local_matrix


def apply_absolute_matrix(scene_object, matrix, *, local=False):
    matrix = coerce_matrix(matrix)
    scene_object.local_matrix = matrix if local else local_matrix_from_world(
        scene_object,
        matrix,
    )


def apply_relative_matrix(scene_object, matrix, *, local=False, pivot=None):
    matrix = coerce_matrix(matrix)
    if local:
        scene_object.local_matrix = scene_object.local_matrix * matrix
        return

    if pivot is not None:
        matrix = matrix_around_pivot(matrix, pivot)
    new_world = matrix * scene_object.world_matrix
    scene_object.local_matrix = local_matrix_from_world(scene_object, new_world)


def world_origin(scene_object):
    return scene_object.world_matrix.transform_point(Vec3(0, 0, 0))


def world_vector_to_parent_local(scene_object, world_vector):
    vector = coerce_vec3(world_vector)
    if scene_object.parent is not None:
        return scene_object.parent.world_inverse_matrix.transform_vector(vector)
    return vector


def parent_linear_is_identity(scene_object, eps=1e-6):
    parent = scene_object.parent
    if parent is None:
        return True
    axes = (
        (Vec3(1, 0, 0), Vec3(1, 0, 0)),
        (Vec3(0, 1, 0), Vec3(0, 1, 0)),
        (Vec3(0, 0, 1), Vec3(0, 0, 1)),
    )
    for source, expected in axes:
        actual = parent.world_matrix.transform_vector(source)
        if (
            abs(actual.x - expected.x) > eps
            or abs(actual.y - expected.y) > eps
            or abs(actual.z - expected.z) > eps
        ):
            return False
    return True


def walk_scene_objects(world):
    yield from walk_object_tree(getattr(world, "objects", ()))


def walk_object_tree(scene_objects):
    for scene_object in scene_objects or ():
        yield scene_object
        yield from walk_object_tree(scene_object.children)


def descendants(scene_object):
    for child in scene_object.children:
        yield child
        yield from descendants(child)


def expanded_with_descendants(scene_objects):
    expanded = []
    seen = set()
    for scene_object in scene_objects or ():
        for item in (scene_object, *tuple(descendants(scene_object))):
            key = id(item)
            if key not in seen:
                seen.add(key)
                expanded.append(item)
    return tuple(expanded)


def rootmost(scene_objects):
    selected = tuple(scene_objects or ())
    selected_ids = {id(scene_object) for scene_object in selected}
    roots = []
    for scene_object in selected:
        parent = scene_object.parent
        keep = True
        while parent is not None:
            if id(parent) in selected_ids:
                keep = False
                break
            parent = parent.parent
        if keep:
            roots.append(scene_object)
    return tuple(roots)
