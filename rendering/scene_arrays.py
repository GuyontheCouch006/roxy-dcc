from dataclasses import dataclass

import numpy as np

from core import Color, HitRecord, Vec2, Vec3


@dataclass(frozen=True)
class TriangleSource:
    object_path: str
    shape_name: str
    group: str


@dataclass
class TriangleSceneArrays:
    """Flattened world-space triangle data for CPU intersector backends.

    The arrays intentionally mirror what an Embree backend needs: contiguous
    vertex/index buffers for intersection, plus primID-indexed side tables for
    shading data that Embree does not own.
    """

    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    has_uv: np.ndarray
    material_idx: np.ndarray
    materials: list
    sources: list
    skipped_primitives: int = 0

    @classmethod
    def empty(cls, skipped_primitives=0):
        return cls(
            vertices=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((0, 3, 3), dtype=np.float32),
            uvs=np.zeros((0, 3, 2), dtype=np.float32),
            has_uv=np.zeros(0, dtype=np.bool_),
            material_idx=np.zeros(0, dtype=np.int32),
            materials=[],
            sources=[],
            skipped_primitives=skipped_primitives,
        )

    @property
    def triangle_count(self):
        return len(self.indices)

    @property
    def vertex_count(self):
        return len(self.vertices)

    def material_for_prim(self, prim_id):
        return self.materials[int(self.material_idx[int(prim_id)])]

    def normal_at(self, prim_id, u, v):
        row = self.normals[int(prim_id)]
        w = 1.0 - float(u) - float(v)
        n = row[0] * w + row[1] * float(u) + row[2] * float(v)
        length = float(np.linalg.norm(n))
        if length <= 1e-12:
            return Vec3(0.0, 1.0, 0.0)
        n = n / length
        return Vec3(float(n[0]), float(n[1]), float(n[2]))

    def uv_at(self, prim_id, u, v):
        prim_id = int(prim_id)
        if not self.has_uv[prim_id]:
            return None
        row = self.uvs[prim_id]
        w = 1.0 - float(u) - float(v)
        uv = row[0] * w + row[1] * float(u) + row[2] * float(v)
        return Vec2(float(uv[0]), float(uv[1]))

    def hit_record(self, ray, prim_id, t, u, v):
        return HitRecord.from_ray(
            ray,
            float(t),
            self.normal_at(prim_id, u, v),
            material=self.material_for_prim(prim_id),
            uv=self.uv_at(prim_id, u, v),
        )


def flatten_world_triangles(world):
    """Return an Embree-ready triangle scene for all mesh geometry in world.

    Infinite primitives and analytic shapes are counted as skipped for now. That
    lets an Embree backend accelerate mesh-heavy assets while the existing Python
    path can still handle spheres/planes/cubes during a staged migration.
    """

    from scene.materials import Diffuse
    from scene.mesh import IndexedMesh, Mesh, indexed_triangle_arrays

    default_material = Diffuse(Color(0.8, 0.8, 0.8))
    vertices = []
    indices = []
    normals = []
    uvs = []
    has_uv = []
    material_idx = []
    materials = []
    material_to_idx = {}
    sources = []
    skipped_primitives = 0

    def add_material(material):
        material = material or default_material
        key = id(material)
        if key not in material_to_idx:
            material_to_idx[key] = len(materials)
            materials.append(material)
        return material_to_idx[key]

    def object_path(obj, parent_path):
        name = obj.name or obj.__class__.__name__
        return f"{parent_path}/{name}" if parent_path else name

    def add_triangle_batch(arrays, shape, obj_path, shape_name):
        base = len(vertices)
        tri_count = len(arrays["v0"])
        if tri_count == 0:
            return

        tri_vertices = np.stack(
            [arrays["v0"], arrays["v1"], arrays["v2"]], axis=1
        ).astype(np.float32, copy=False)
        tri_normals = np.stack(
            [arrays["n0"], arrays["n1"], arrays["n2"]], axis=1
        ).astype(np.float32, copy=False)
        tri_uvs = np.stack(
            [
                arrays.get("uv0", np.zeros((tri_count, 2), dtype=np.float32)),
                arrays.get("uv1", np.zeros((tri_count, 2), dtype=np.float32)),
                arrays.get("uv2", np.zeros((tri_count, 2), dtype=np.float32)),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        tri_has_uv = np.asarray(
            arrays.get("has_uv", np.zeros(tri_count, dtype=np.int32)),
            dtype=np.bool_,
        ).reshape((-1,))

        vertices.extend(tri_vertices.reshape((-1, 3)))
        indices.extend(np.arange(base, base + tri_count * 3, dtype=np.int32).reshape((-1, 3)))
        normals.extend(tri_normals)
        uvs.extend(tri_uvs)
        has_uv.extend(tri_has_uv)

        group_names = arrays.get("groups", ["default"])
        for raw_group_idx in arrays["group_idx"]:
            group = group_names[int(raw_group_idx)]
            material_idx.append(add_material(shape.material_for_group(group)))
            sources.append(TriangleSource(obj_path, shape_name, group))

    def collect_mesh(obj, parent_path=""):
        nonlocal skipped_primitives
        if not obj.renderable:
            return

        path = object_path(obj, parent_path)
        for shape in obj.shapes:
            geo = shape.geometry
            shape_name = shape.name or getattr(geo, "_name", "") or type(geo).__name__

            if isinstance(geo, IndexedMesh):
                arrays = indexed_triangle_arrays(
                    geo,
                    matrix=obj.world_matrix,
                    normal_matrix=obj.world_inverse_transpose_matrix,
                )
                add_triangle_batch(arrays, shape, path, shape_name)
            elif isinstance(geo, Mesh):
                arrays = _mesh_triangle_arrays(
                    geo,
                    obj.world_matrix,
                    obj.world_inverse_transpose_matrix,
                )
                add_triangle_batch(arrays, shape, path, shape_name)
            else:
                skipped_primitives += 1

        for child in obj.children:
            collect_mesh(child, path)

    for obj in world.objects:
        collect_mesh(obj)

    if not indices:
        return TriangleSceneArrays.empty(skipped_primitives=skipped_primitives)

    return TriangleSceneArrays(
        vertices=np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        indices=np.asarray(indices, dtype=np.int32).reshape((-1, 3)),
        normals=np.asarray(normals, dtype=np.float32).reshape((-1, 3, 3)),
        uvs=np.asarray(uvs, dtype=np.float32).reshape((-1, 3, 2)),
        has_uv=np.asarray(has_uv, dtype=np.bool_).reshape((-1,)),
        material_idx=np.asarray(material_idx, dtype=np.int32).reshape((-1,)),
        materials=materials,
        sources=sources,
        skipped_primitives=skipped_primitives,
    )


def _mesh_triangle_arrays(mesh, matrix, normal_matrix):
    v0, v1, v2 = [], [], []
    n0, n1, n2 = [], [], []
    uv0, uv1, uv2 = [], [], []
    has_uv = []
    group_idx = []
    groups = []
    group_to_idx = {}

    def group_id(group):
        if group not in group_to_idx:
            group_to_idx[group] = len(groups)
            groups.append(group)
        return group_to_idx[group]

    def transform_normal(n):
        transformed = normal_matrix.transform_vector(n)
        return list(transformed.normalize())

    def uv_or_zero(uv):
        return list(uv) if uv is not None else [0.0, 0.0]

    for tri in mesh._triangles:
        face_normal = tri._normal
        v0.append(list(matrix.transform_point(tri._v0)))
        v1.append(list(matrix.transform_point(tri._v1)))
        v2.append(list(matrix.transform_point(tri._v2)))
        n0.append(transform_normal(tri._n0 if tri._n0 is not None else face_normal))
        n1.append(transform_normal(tri._n1 if tri._n1 is not None else face_normal))
        n2.append(transform_normal(tri._n2 if tri._n2 is not None else face_normal))
        uv0.append(uv_or_zero(tri._uv0))
        uv1.append(uv_or_zero(tri._uv1))
        uv2.append(uv_or_zero(tri._uv2))
        has_uv.append(
            tri._uv0 is not None and tri._uv1 is not None and tri._uv2 is not None
        )
        group_idx.append(group_id(tri.group))

    return {
        "v0": np.asarray(v0, dtype=np.float32),
        "v1": np.asarray(v1, dtype=np.float32),
        "v2": np.asarray(v2, dtype=np.float32),
        "n0": np.asarray(n0, dtype=np.float32),
        "n1": np.asarray(n1, dtype=np.float32),
        "n2": np.asarray(n2, dtype=np.float32),
        "uv0": np.asarray(uv0, dtype=np.float32),
        "uv1": np.asarray(uv1, dtype=np.float32),
        "uv2": np.asarray(uv2, dtype=np.float32),
        "has_uv": np.asarray(has_uv, dtype=np.int32),
        "group_idx": np.asarray(group_idx, dtype=np.int32),
        "groups": groups or ["default"],
    }
