import os
import tempfile

from scene.io.obj_reader import OBJReader
from scene.mesh import IndexedMesh, Mesh, Triangle
from scene.scene_object import SceneObject
from core import Vec3
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _write_obj(content):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False)
    f.write(content)
    f.close()
    return f.name


def _cleanup(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _all_triangles(root):
    """Collect all triangles from a SceneObject hierarchy."""
    tris = []
    for child in root.children:
        if child.shapes:
            mesh = child.shapes[0].geometry
            if isinstance(mesh, Mesh):
                tris.extend(mesh._triangles)
        tris.extend(_all_triangles(child))
    return tris


# ── load_as_mesh: basic loading ──────────────────────────────────────────────

def test_load_returns_mesh():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        mesh = OBJReader.load_as_mesh(path)
        assert isinstance(mesh, Mesh)
    finally:
        _cleanup(path)

def test_single_triangle_vertex_positions():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        mesh = OBJReader.load_as_mesh(path)
        assert len(mesh._triangles) == 1
        tri = mesh._triangles[0]
        assert vec3_approx_eq(tri._v0, Vec3(0, 0, 0))
        assert vec3_approx_eq(tri._v1, Vec3(1, 0, 0))
        assert vec3_approx_eq(tri._v2, Vec3(0, 1, 0))
    finally:
        _cleanup(path)

def test_multiple_triangles():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "f 1 2 3\nf 4 5 6\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        assert len(mesh._triangles) == 2
    finally:
        _cleanup(path)


# ── Normals ───────────────────────────────────────────────────────────────────

def test_vertex_normals_loaded():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
        "f 1//1 2//2 3//3\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        tri = mesh._triangles[0]
        assert tri._n0 is not None
        assert vec3_approx_eq(tri._n0, Vec3(0, 0, 1))
    finally:
        _cleanup(path)

def test_face_without_normals_has_none():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        mesh = OBJReader.load_as_mesh(path)
        tri = mesh._triangles[0]
        assert tri._n0 is None
    finally:
        _cleanup(path)


# ── UVs ───────────────────────────────────────────────────────────────────────

def test_uvs_loaded():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "f 1/1 2/2 3/3\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        tri = mesh._triangles[0]
        assert tri._uv0 is not None
        assert approx_eq(tri._uv0[0], 0.0) and approx_eq(tri._uv0[1], 0.0)
        assert approx_eq(tri._uv1[0], 1.0)
    finally:
        _cleanup(path)

def test_uvs_normals_combined():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
        "f 1/1/1 2/2/2 3/3/3\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        tri = mesh._triangles[0]
        assert tri._uv0 is not None
        assert tri._n0 is not None
    finally:
        _cleanup(path)


# ── Quad fan triangulation ────────────────────────────────────────────────────

def test_quad_triangulated_into_two_triangles():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f 1 2 3 4\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        assert len(mesh._triangles) == 2, \
            f"Quad should produce 2 triangles, got {len(mesh._triangles)}"
    finally:
        _cleanup(path)

def test_pentagon_triangulated_into_three_triangles():
    obj = (
        "v 1 0 0\nv 0.309 0.951 0\nv -0.809 0.588 0\n"
        "v -0.809 -0.588 0\nv 0.309 -0.951 0\n"
        "f 1 2 3 4 5\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_mesh(path)
        assert len(mesh._triangles) == 3
    finally:
        _cleanup(path)


# ── load_as_mesh: ray intersection ───────────────────────────────────────────

def test_loaded_mesh_intersects_correctly():
    from core import Ray
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        mesh = OBJReader.load_as_mesh(path)
        ray = Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1))
        hit = mesh.intersect(ray)
        assert hit is not None
        assert approx_eq(hit.t, 1.0), f"Expected t=1, got {hit.t}"
    finally:
        _cleanup(path)


# ── load_as_indexed_mesh ──────────────────────────────────────────────────────

def test_load_as_indexed_mesh_returns_indexed_mesh():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        mesh = OBJReader.load_as_indexed_mesh(path)
        assert isinstance(mesh, IndexedMesh)
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
    finally:
        _cleanup(path)

def test_load_as_indexed_mesh_preserves_usemtl_groups():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "usemtl matA\nf 1 2 3\n"
        "usemtl matB\nf 4 5 6\n"
    )
    path = _write_obj(obj)
    try:
        mesh = OBJReader.load_as_indexed_mesh(path)
        assert mesh.groups == ["default", "matA", "matB"]
        assert mesh.group_for_triangle(0) == "matA"
        assert mesh.group_for_triangle(1) == "matB"
    finally:
        _cleanup(path)


# ── load: hierarchy ───────────────────────────────────────────────────────────

def test_load_returns_scene_object():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        root = OBJReader.load(path)
        assert isinstance(root, SceneObject)
    finally:
        _cleanup(path)

def test_load_hierarchy_has_children():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        root = OBJReader.load(path)
        assert len(root.children) > 0
    finally:
        _cleanup(path)

def test_load_group_has_shape_with_mesh():
    path = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    try:
        root = OBJReader.load(path)
        group = root.children[0]
        assert len(group.shapes) == 1
        assert isinstance(group.shapes[0].geometry, Mesh)
    finally:
        _cleanup(path)

def test_load_usemtl_produces_named_child():
    # Node names come from usemtl boundaries, not g tags.
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl myMat\nf 1 2 3\n"
    path = _write_obj(obj)
    try:
        root = OBJReader.load(path)
        names = [c.name for c in root.children]
        assert 'myMat' in names
    finally:
        _cleanup(path)

def test_load_two_usemtl_produce_two_children():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "usemtl matA\nf 1 2 3\n"
        "usemtl matB\nf 4 5 6\n"
    )
    path = _write_obj(obj)
    try:
        root = OBJReader.load(path)
        assert len(root.children) == 2
    finally:
        _cleanup(path)

def test_load_indexed_hierarchy_has_indexed_mesh_and_material_groups():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "usemtl matA\nf 1 2 3\n"
        "usemtl matB\nf 4 5 6\n"
    )
    path = _write_obj(obj)
    try:
        root = OBJReader.load(path, indexed=True)
        assert len(root.children) == 1
        shape = root.children[0].shapes[0]
        assert isinstance(shape.geometry, IndexedMesh)
        assert set(shape.material_groups.keys()) == {"matA", "matB"}
    finally:
        _cleanup(path)


if __name__ == "__main__":
    tests = [
        test_load_returns_mesh,
        test_single_triangle_vertex_positions,
        test_multiple_triangles,
        test_vertex_normals_loaded,
        test_face_without_normals_has_none,
        test_uvs_loaded,
        test_uvs_normals_combined,
        test_quad_triangulated_into_two_triangles,
        test_pentagon_triangulated_into_three_triangles,
        test_loaded_mesh_intersects_correctly,
        test_load_as_indexed_mesh_returns_indexed_mesh,
        test_load_as_indexed_mesh_preserves_usemtl_groups,
        test_load_returns_scene_object,
        test_load_hierarchy_has_children,
        test_load_group_has_shape_with_mesh,
        test_load_usemtl_produces_named_child,
        test_load_two_usemtl_produce_two_children,
        test_load_indexed_hierarchy_has_indexed_mesh_and_material_groups,
    ]
    run_tests(tests)
