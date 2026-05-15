from scene.mesh import IndexedMesh, Triangle, Mesh, indexed_triangle_arrays
from core import Vec2, Vec3, Ray, Mat4x4
from tests.utils import run_tests, approx_eq, vec3_approx_eq


def _unit_tri():
    """Triangle in the XY plane facing +Z."""
    return Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))


# ── Triangle.intersect ────────────────────────────────────────────────────────

def test_triangle_hit_center():
    tri = _unit_tri()
    ray = Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1))
    hit = tri.intersect(ray)
    assert hit is not None, "Expected hit through triangle center"
    t, u, v, _ = hit
    assert t > 0
    assert approx_eq(t, 1.0), f"Expected t=1, got {t}"

def test_triangle_miss_outside():
    tri = _unit_tri()
    ray = Ray(Vec3(2.0, 2.0, -1), Vec3(0, 0, 1))
    assert tri.intersect(ray) is None

def test_triangle_miss_parallel():
    tri = _unit_tri()
    ray = Ray(Vec3(0.25, 0.25, 0), Vec3(1, 0, 0))
    assert tri.intersect(ray) is None

def test_triangle_miss_behind():
    tri = _unit_tri()
    ray = Ray(Vec3(0.25, 0.25, 2), Vec3(0, 0, 1))
    assert tri.intersect(ray) is None

def test_triangle_barycentric_coords():
    tri = _unit_tri()
    ray = Ray(Vec3(0.3, 0.2, -1), Vec3(0, 0, 1))
    hit = tri.intersect(ray)
    assert hit is not None
    _, u, v, _ = hit
    assert u >= 0 and v >= 0 and u + v <= 1


# ── Triangle.normal_at ────────────────────────────────────────────────────────

def test_triangle_flat_normal_no_vertex_normals():
    tri = _unit_tri()
    n = tri.normal_at(0.25, 0.25)
    assert vec3_approx_eq(n, Vec3(0, 0, 1)) or vec3_approx_eq(n, Vec3(0, 0, -1)), \
        f"Expected ±Z normal for XY triangle, got {n}"

def test_triangle_interpolated_normals():
    n0 = Vec3(0, 0, 1)
    n1 = Vec3(1, 0, 0)
    n2 = Vec3(0, 1, 0)
    tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), n0, n1, n2)
    n = tri.normal_at(0.0, 0.0)   # u=0, v=0 → weight on n0
    assert approx_eq(n.length(), 1.0), "Interpolated normal should be unit length"


# ── Triangle.local_bounds ─────────────────────────────────────────────────────

def test_triangle_bounds_contain_vertices():
    tri = Triangle(Vec3(-1, 0, 2), Vec3(3, 0, 2), Vec3(1, 4, 2))
    b = tri.local_bounds()
    assert b.min.x <= -1 and b.max.x >= 3
    assert b.min.y <= 0  and b.max.y >= 4
    assert b.min.z <= 2  and b.max.z >= 2

def test_triangle_bounds_degenerate_axis():
    tri = _unit_tri()      # all z=0
    b = tri.local_bounds()
    assert b.min.z < b.max.z, "Degenerate axis should be padded"


# ── Triangle.centroid ─────────────────────────────────────────────────────────

def test_triangle_centroid():
    tri = Triangle(Vec3(0, 0, 0), Vec3(3, 0, 0), Vec3(0, 3, 0))
    c = tri.centroid()
    assert vec3_approx_eq(c, Vec3(1, 1, 0)), f"Expected (1,1,0), got {c}"


# ── Triangle.uv_at ────────────────────────────────────────────────────────────

def test_triangle_no_uvs_returns_none():
    tri = _unit_tri()
    assert tri.uv_at(0.25, 0.25) is None

def test_triangle_uv_interpolation():
    from core import Vec2
    tri = Triangle(
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
        uv0=Vec2(0, 0), uv1=Vec2(1, 0), uv2=Vec2(0, 1),
    )
    uv = tri.uv_at(1.0, 0.0)   # u=1, v=0 → at v1
    assert uv is not None
    assert approx_eq(uv[0], 1.0) and approx_eq(uv[1], 0.0)


# ── Mesh.intersect ────────────────────────────────────────────────────────────

def test_mesh_hit_closest():
    t1 = Triangle(Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1))
    t2 = Triangle(Vec3(0, 0, 3), Vec3(1, 0, 3), Vec3(0, 1, 3))
    mesh = Mesh([t1, t2])
    ray = Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1))
    hit = mesh.intersect(ray)
    assert hit is not None
    assert approx_eq(hit.t, 2.0), f"Should hit closer triangle (t=2), got {hit.t}"

def test_mesh_miss():
    mesh = Mesh([_unit_tri()])
    ray = Ray(Vec3(5, 5, -1), Vec3(0, 0, 1))
    assert mesh.intersect(ray) is None

def test_mesh_hit_returns_meshhit():
    from scene.mesh import MeshHit
    mesh = Mesh([_unit_tri()])
    ray = Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1))
    hit = mesh.intersect(ray)
    assert isinstance(hit, MeshHit)


# ── Mesh.local_bounds ────────────────────────────────────────────────────────

def test_mesh_bounds_union():
    t1 = Triangle(Vec3(-2, 0, 0), Vec3(-1, 0, 0), Vec3(-1, 1, 0))
    t2 = Triangle(Vec3( 1, 0, 0), Vec3( 2, 0, 0), Vec3( 2, 1, 0))
    mesh = Mesh([t1, t2])
    b = mesh.local_bounds()
    assert b.min.x <= -2 and b.max.x >= 2


# ── IndexedMesh storage ──────────────────────────────────────────────────────

def _indexed_mesh():
    return IndexedMesh(
        positions=[
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
            [0, 0, 3], [1, 0, 3], [0, 1, 3],
        ],
        tri_pos_idx=[
            [0, 1, 2],
            [3, 4, 5],
        ],
        normals=[
            [0, 0, 1], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ],
        tri_normal_idx=[
            [0, 1, 2],
            [3, 4, 5],
        ],
        uvs=[
            [0, 0], [1, 0], [0, 1],
            [0, 0], [1, 0], [0, 1],
        ],
        tri_uv_idx=[
            [0, 1, 2],
            [3, 4, 5],
        ],
        groups=["near", "far"],
        tri_group_idx=[0, 1],
        name="indexed",
    )

def test_indexed_mesh_stores_vertices_and_triangle_indices():
    mesh = _indexed_mesh()
    assert mesh.vertex_count == 6
    assert mesh.triangle_count == 2
    assert mesh.groups == ["near", "far"]
    v0, v1, v2 = mesh.triangle_vertices(0)
    assert vec3_approx_eq(v0, Vec3(0, 0, 1))
    assert vec3_approx_eq(v1, Vec3(1, 0, 1))
    assert vec3_approx_eq(v2, Vec3(0, 1, 1))

def test_indexed_mesh_hit_returns_meshhit_and_group_name():
    mesh = _indexed_mesh()
    ray = Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1))
    hit = mesh.intersect(ray)
    assert hit is not None
    assert approx_eq(hit.t, 2.0)
    assert hit.group == "near"

def test_indexed_mesh_miss():
    mesh = _indexed_mesh()
    ray = Ray(Vec3(5, 5, -1), Vec3(0, 0, 1))
    assert mesh.intersect(ray) is None

def test_indexed_mesh_interpolates_uvs():
    mesh = _indexed_mesh()
    uv = mesh.uv_at(0, 1.0, 0.0)
    assert uv is not None
    assert approx_eq(uv.x, 1.0)
    assert approx_eq(uv.y, 0.0)

def test_indexed_mesh_interpolates_normals():
    mesh = _indexed_mesh()
    normal = mesh.normal_at(0, 0.0, 0.0)
    assert approx_eq(normal.length(), 1.0)
    assert vec3_approx_eq(normal, Vec3(0, 0, 1))

def test_indexed_mesh_bounds_union():
    mesh = _indexed_mesh()
    bounds = mesh.local_bounds()
    assert bounds.min.z <= 1
    assert bounds.max.z >= 3

def test_indexed_mesh_from_triangles_preserves_group_and_uvs():
    tri = Triangle(
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
        uv0=Vec2(0, 0), uv1=Vec2(1, 0), uv2=Vec2(0, 1),
        group="matA",
    )
    mesh = IndexedMesh.from_triangles([tri])
    hit = mesh.intersect(Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1)))
    assert hit is not None
    assert hit.group == "matA"
    assert hit.uv is not None

def test_indexed_mesh_round_trip_dict():
    mesh = _indexed_mesh()
    restored = IndexedMesh.from_dict(mesh.to_dict())
    assert restored.vertex_count == mesh.vertex_count
    assert restored.triangle_count == mesh.triangle_count
    assert restored.groups == mesh.groups
    hit = restored.intersect(Ray(Vec3(0.25, 0.25, -1), Vec3(0, 0, 1)))
    assert hit is not None
    assert hit.group == "near"

def test_indexed_triangle_arrays_apply_transform_once():
    mesh = _indexed_mesh()
    arrays = indexed_triangle_arrays(
        mesh,
        matrix=Mat4x4.translation(10, 20, 30),
        normal_matrix=Mat4x4.identity(),
    )
    assert arrays['v0'].shape == (2, 3)
    assert approx_eq(arrays['v0'][0][0], 10.0)
    assert approx_eq(arrays['v0'][0][1], 20.0)
    assert approx_eq(arrays['v0'][0][2], 31.0)
    assert approx_eq(arrays['v1'][1][0], 11.0)
    assert arrays['groups'] == ["near", "far"]
    assert arrays['group_idx'][1] == 1
    assert arrays['has_uv'][0] == 1
    assert approx_eq(arrays['uv1'][0][0], 1.0)
    assert approx_eq(arrays['uv2'][0][1], 1.0)

def test_indexed_triangle_arrays_fallback_normals():
    mesh = IndexedMesh(
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        tri_pos_idx=[[0, 1, 2]],
    )
    arrays = indexed_triangle_arrays(mesh)
    assert approx_eq(arrays['n0'][0][2], 1.0)
    assert approx_eq(arrays['n1'][0][2], 1.0)
    assert approx_eq(arrays['n2'][0][2], 1.0)
    assert arrays['has_uv'][0] == 0


if __name__ == "__main__":
    tests = [
        test_triangle_hit_center,
        test_triangle_miss_outside,
        test_triangle_miss_parallel,
        test_triangle_miss_behind,
        test_triangle_barycentric_coords,
        test_triangle_flat_normal_no_vertex_normals,
        test_triangle_interpolated_normals,
        test_triangle_bounds_contain_vertices,
        test_triangle_bounds_degenerate_axis,
        test_triangle_centroid,
        test_triangle_no_uvs_returns_none,
        test_triangle_uv_interpolation,
        test_mesh_hit_closest,
        test_mesh_miss,
        test_mesh_hit_returns_meshhit,
        test_mesh_bounds_union,
        test_indexed_mesh_stores_vertices_and_triangle_indices,
        test_indexed_mesh_hit_returns_meshhit_and_group_name,
        test_indexed_mesh_miss,
        test_indexed_mesh_interpolates_uvs,
        test_indexed_mesh_interpolates_normals,
        test_indexed_mesh_bounds_union,
        test_indexed_mesh_from_triangles_preserves_group_and_uvs,
        test_indexed_mesh_round_trip_dict,
        test_indexed_triangle_arrays_apply_transform_once,
        test_indexed_triangle_arrays_fallback_normals,
    ]
    run_tests(tests)
