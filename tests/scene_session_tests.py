# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Expected-contract tests for the planned pure SceneSession API:
#              stable object handles, selection state, transforms, and undo.
# ============================================

from core import Color, Mat4x4, Vec3
from scene import Camera, Diffuse, SceneObject, Sphere, World
from tests.utils import approx_eq, run_tests, vec3_approx_eq


def _scene_session_class():
    try:
        from scene import SceneSession
        return SceneSession
    except ImportError:
        pass

    try:
        from scene.scene_session import SceneSession
        return SceneSession
    except ImportError as exc:
        raise AssertionError(
            "Expected SceneSession to be importable from scene or scene.scene_session"
        ) from exc


def _session(world, **kwargs):
    return _scene_session_class()(world, **kwargs)


def _world_with_hierarchy():
    world = World(use_sky=False)
    root = SceneObject(name="root", translation=Vec3(10, 0, 0))
    child = SceneObject(name="child", translation=Vec3(1, 0, 0))
    grandchild = SceneObject(name="grandchild", translation=Vec3(0, 2, 0))
    sibling = SceneObject(name="sibling", translation=Vec3(-4, 0, 0))

    child.add_child(grandchild)
    root.add_child(child)
    world.add_object(root)
    world.add_object(sibling)
    return world, root, child, grandchild, sibling


def _value(obj, name):
    value = getattr(obj, name)
    return value() if callable(value) else value


def _selected_objects(session):
    return tuple(_value(session, "selected_objects"))


def _highlighted_objects(session):
    return tuple(_value(session, "highlighted_objects"))


def _transform_targets(session):
    return tuple(_value(session, "transform_targets"))


def _rootmost_selection(session):
    return tuple(_value(session, "rootmost_selection"))


def _active_object(session):
    return _value(session, "active_object")


def _can_undo(session):
    return bool(_value(session, "can_undo"))


def _origin(matrix):
    return matrix.transform_point(Vec3(0, 0, 0))


def _axis_x(matrix):
    return matrix.transform_vector(Vec3(1, 0, 0))


def _assert_transform_origin(matrix, expected):
    assert vec3_approx_eq(_origin(matrix), expected), (
        f"Expected transform origin {expected}, got {_origin(matrix)}"
    )


def _assert_axis_length(vector, expected):
    assert approx_eq(vector.length(), expected), (
        f"Expected axis length {expected}, got {vector.length()}"
    )


def test_session_returns_stable_handles_for_world_objects():
    world, root, child, grandchild, sibling = _world_with_hierarchy()
    session = _session(world)

    root_handle = session.object(root)
    child_handle = session.object(child)
    grandchild_handle = session.object(grandchild)
    sibling_handle = session.object(sibling)

    assert session.object(root) is root_handle
    assert session.object(child) is child_handle
    assert root_handle.scene_object is root
    assert child_handle.scene_object is child

    assert tuple(session.objects()) == (
        root_handle,
        child_handle,
        grandchild_handle,
        sibling_handle,
    )


def test_selection_replace_toggle_clear_and_active_object():
    world, root, child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world)
    root_handle = session.object(root)
    child_handle = session.object(child)

    session.select(root_handle)

    assert _selected_objects(session) == (root_handle,)
    assert _active_object(session) is root_handle

    session.select(child_handle)

    assert _selected_objects(session) == (child_handle,)
    assert _active_object(session) is child_handle

    session.toggle_selection(root_handle)

    assert _selected_objects(session) == (child_handle, root_handle)
    assert _active_object(session) is root_handle

    session.toggle_selection(root_handle)

    assert _selected_objects(session) == (child_handle,)
    assert _active_object(session) is child_handle

    session.clear_selection()

    assert _selected_objects(session) == ()
    assert _active_object(session) is None


def test_parent_selection_highlights_descendants_and_prunes_transform_targets():
    world, root, child, grandchild, sibling = _world_with_hierarchy()
    session = _session(world)
    root_handle = session.object(root)
    child_handle = session.object(child)
    grandchild_handle = session.object(grandchild)
    sibling_handle = session.object(sibling)

    session.select(root_handle)

    assert _highlighted_objects(session) == (
        root_handle,
        child_handle,
        grandchild_handle,
    )
    assert _transform_targets(session) == (root_handle,)

    session.select((root_handle, child_handle, grandchild_handle, sibling_handle))

    assert _rootmost_selection(session) == (root_handle, sibling_handle)
    assert _transform_targets(session) == (root_handle, sibling_handle)


def test_handle_move_rotate_scale_update_local_and_world_matrices():
    world, root, child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world)
    child_handle = session.object(child)

    child_handle.move(Vec3(2, 0, 0))

    assert vec3_approx_eq(child.translation, Vec3(3, 0, 0))
    _assert_transform_origin(child.local_matrix, Vec3(3, 0, 0))
    _assert_transform_origin(child.world_matrix, Vec3(13, 0, 0))

    child_handle.rotate(Vec3(0, 0, 90))

    assert vec3_approx_eq(child.rotation, Vec3(0, 0, 90))
    assert vec3_approx_eq(_axis_x(child.local_matrix), Vec3(0, 1, 0))

    child_handle.scale(Vec3(2, 2, 2))

    assert vec3_approx_eq(child.scale, Vec3(2, 2, 2))
    _assert_axis_length(_axis_x(child.local_matrix), 2.0)


def test_handle_transform_accepts_matrix_or_trs_but_not_both():
    world, root, child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world)
    child_handle = session.object(child)

    child_handle.transform(matrix=Mat4x4.translation(4, 5, 6))

    assert child.matrix_mode
    _assert_transform_origin(child.local_matrix, Vec3(-6, 5, 6))
    _assert_transform_origin(child.world_matrix, Vec3(4, 5, 6))

    child_handle.transform(
        translate=Vec3(2, 3, 4),
        rotate=Vec3(0, 0, 90),
        scale=Vec3(2, 2, 2),
        local=True,
    )

    assert not child.matrix_mode
    assert vec3_approx_eq(child.translation, Vec3(2, 3, 4))
    assert vec3_approx_eq(child.rotation, Vec3(0, 0, 90))
    assert vec3_approx_eq(child.scale, Vec3(2, 2, 2))

    child_handle.transform(
        translate=Vec3(2, 3, 4),
        rotate=Vec3(0, 0, 90),
        scale=Vec3(2, 2, 2),
    )

    assert child.matrix_mode
    _assert_transform_origin(child.world_matrix, Vec3(2, 3, 4))
    _assert_axis_length(_axis_x(child.world_matrix), 2.0)

    try:
        child_handle.transform(
            matrix=Mat4x4.identity(),
            translation=Vec3(1, 0, 0),
        )
        assert False, "Expected mixing matrix and TRS transform inputs to raise"
    except ValueError:
        pass


def test_undo_redo_restore_transform_and_selection_state():
    world, root, child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world)
    root_handle = session.object(root)
    child_handle = session.object(child)

    session.select(root_handle)
    root_handle.move(Vec3(5, 0, 0))
    session.select(child_handle)

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (child_handle,)

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (root_handle,)
    assert _active_object(session) is root_handle

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(10, 0, 0))
    assert _selected_objects(session) == (root_handle,)
    assert _active_object(session) is root_handle

    session.redo()

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (root_handle,)

    session.redo()

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (child_handle,)
    assert _active_object(session) is child_handle


def test_undo_disabled_records_no_history():
    world, root, _child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world, undo_enabled=False)
    root_handle = session.object(root)

    session.select(root_handle)
    root_handle.move(Vec3(5, 0, 0))

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (root_handle,)
    assert not _can_undo(session)

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(15, 0, 0))
    assert _selected_objects(session) == (root_handle,)


def test_undo_depth_trims_oldest_entries():
    world, root, _child, _grandchild, _sibling = _world_with_hierarchy()
    session = _session(world, undo_depth=2)
    root_handle = session.object(root)

    root_handle.move(Vec3(1, 0, 0))
    root_handle.move(Vec3(1, 0, 0))
    root_handle.move(Vec3(1, 0, 0))

    assert vec3_approx_eq(root.translation, Vec3(13, 0, 0))
    assert _can_undo(session)

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(12, 0, 0))

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(11, 0, 0))
    assert not _can_undo(session)

    session.undo()

    assert vec3_approx_eq(root.translation, Vec3(11, 0, 0))


def test_session_returns_stable_non_object_handles_and_payload_selection():
    mat = Diffuse(Color(0.2, 0.3, 0.4), name="sharedLambert")
    shape_obj = SceneObject(shape=Sphere(1.0), material=mat, name="ball")
    world = World(objects=[shape_obj], use_sky=False)
    session = _session(world)
    shape = shape_obj.shapes[0]

    shape_handle = session.shape(shape)
    material_handle = session.material("sharedLambert")
    history_handle = session.history_node(f"{shape.name or 'Sphere'}Source")

    assert session.shape(shape) is shape_handle
    assert session.material(mat) is material_handle
    assert shape_handle.raw is shape
    assert material_handle.raw is mat
    assert history_handle.raw.shape is shape

    session.set_selected_payload(material_handle)

    assert session.selected_payload() is material_handle
    assert session.selected_raw_payload() is mat


def test_session_assigns_unique_names_to_scriptable_nodes():
    first_mat = Diffuse(Color(0.2, 0.3, 0.4))
    second_mat = Diffuse(Color(0.7, 0.7, 0.7))
    first = SceneObject(shape=Sphere(1.0), material=first_mat)
    second = SceneObject(shape=Sphere(1.0), material=second_mat, name="transform")
    world = World(
        objects=[first, second],
        cameras=[Camera(name="camera"), Camera(name="camera")],
        use_sky=False,
    )

    session = _session(world)

    assert world.name == "world"
    assert first.name == "transform"
    assert first.shapes[0].name == "transformShape"
    assert first_mat.name == "diffuse"
    assert second.name == "transform1"
    assert second.shapes[0].name == "transform1Shape"
    assert second_mat.name == "diffuse1"
    assert [camera.name for camera in world.cameras] == ["camera", "camera1"]
    assert session.node("world").raw is world
    assert session.node("transform").raw is first
    assert session.node("transformShape").raw is first.shapes[0]
    assert session.node("diffuse1").raw is second_mat
    assert [handle.name for handle in session.nodes(kind="camera")] == ["camera", "camera1"]


def test_renaming_by_api_keeps_node_lookup_unique():
    mat = Diffuse(Color(0.2, 0.3, 0.4), name="mat")
    root = SceneObject(name="root")
    child = SceneObject(shape=Sphere(1.0), material=mat, name="child")
    root.add_child(child)
    world = World(objects=[root], use_sky=False)
    session = _session(world)

    assert child.name == "child"
    assert child.shapes[0].name == "childShape"

    session.object(child).set_attr("name", "root")

    assert child.name == "root1"
    assert session.node("root").raw is root
    assert session.node("root1").raw is child

    session.shape(child.shapes[0]).set_attr("name", "")

    assert child.shapes[0].name == "root1Shape"
    assert session.node("root1Shape").raw is child.shapes[0]


def test_handle_set_attr_and_shader_assignment_are_undoable():
    old_mat = Diffuse(Color(0.2, 0.3, 0.4), name="oldMat")
    new_mat = Diffuse(Color(0.7, 0.7, 0.7), name="newMat")
    obj = SceneObject(shape=Sphere(1.0), material=old_mat, name="ball")
    sibling = SceneObject(shape=Sphere(1.0), material=new_mat, name="other")
    world = World(objects=[obj, sibling], use_sky=False)
    session = _session(world)
    shape_handle = session.shape(obj.shapes[0])
    material_handle = session.material("newMat")

    shape_handle.set_attr("radius", 2.5)
    material_handle.set_attr("albedo", Color(0.1, 0.2, 0.3))
    shape_handle.connect_shader(material_handle)

    assert approx_eq(obj.shape._radius, 2.5)
    assert vec3_approx_eq(new_mat._albedo, Color(0.1, 0.2, 0.3))
    assert obj.material is new_mat

    session.undo()
    assert obj.material is old_mat
    session.undo()
    assert vec3_approx_eq(new_mat._albedo, Color(0.7, 0.7, 0.7))
    session.undo()
    assert approx_eq(obj.shape._radius, 1.0)


def test_connect_attr_rejects_invalid_connections():
    mat = Diffuse(Color(0.2, 0.3, 0.4), name="mat")
    obj = SceneObject(shape=Sphere(1.0), material=mat, name="ball")
    world = World(objects=[obj], use_sky=False)
    session = _session(world)

    try:
        session.connect_attr("mat.outColor", "ball.surfaceShader")
        assert False, "Expected unsupported source attr to raise"
    except ValueError:
        pass

    try:
        session.connect_attr("mat.outSurface", "ballShape.translate")
        assert False, "Expected unsupported destination attr to raise"
    except (KeyError, ValueError):
        pass


if __name__ == "__main__":
    run_tests([
        test_session_returns_stable_handles_for_world_objects,
        test_selection_replace_toggle_clear_and_active_object,
        test_parent_selection_highlights_descendants_and_prunes_transform_targets,
        test_handle_move_rotate_scale_update_local_and_world_matrices,
        test_handle_transform_accepts_matrix_or_trs_but_not_both,
        test_undo_redo_restore_transform_and_selection_state,
        test_undo_disabled_records_no_history,
        test_undo_depth_trims_oldest_entries,
        test_session_returns_stable_non_object_handles_and_payload_selection,
        test_session_assigns_unique_names_to_scriptable_nodes,
        test_renaming_by_api_keeps_node_lookup_unique,
        test_handle_set_attr_and_shader_assignment_are_undoable,
        test_connect_attr_rejects_invalid_connections,
    ])
