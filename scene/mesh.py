from abc import ABC
import taichi as ti
from collections import namedtuple
import numpy as np
from core import Vec2, Vec3, AABB
from core.bvh import BVHNode

Shape = ABC  # avoid circular import with primitives.py

# Raw mesh intersection result — group tag lets Shape look up the right material.
MeshHit = namedtuple('MeshHit', ['t', 'normal', 'uv', 'group'])

class Triangle(Shape):
    """Triangle defined by three vertices."""

    def __init__(self, v0, v1, v2, n0=None, n1=None, n2=None,
                 uv0=None, uv1=None, uv2=None, group="default"):
        self._v0, self._v1, self._v2 = v0, v1, v2
        self._n0, self._n1, self._n2 = n0, n1, n2
        self._uv0, self._uv1, self._uv2 = uv0, uv1, uv2
        self._normal = (v1 - v0).cross(v2 - v0).normalize()
        self._group = group

    @property
    def group(self): return self._group

    def intersect(self, ray):
        e1 = self._v1 - self._v0
        e2 = self._v2 - self._v0
        h  = ray.direction.cross(e2)
        a  = e1.dot(h)

        if abs(a) < 1e-8:       # parallel
            return None

        f = 1.0 / a
        s = ray.origin - self._v0
        u = f * s.dot(h)

        if u < 0 or u > 1:      # outside triangle
            return None

        q = s.cross(e1)
        v = f * ray.direction.dot(q)

        if v < 0 or u + v > 1:  # outside triangle
            return None

        t = f * e2.dot(q)

        if t < 0.001:            # behind ray
            return None

        return t, u, v, self
    
    def normal_at(self, u, v):
        if self._n0 and self._n1 and self._n2:
            return (self._n0 * (1 - u - v) + self._n1 * u + self._n2 * v).normalize()
        else:
            return self._normal
        
    def uv_at(self, u, v):
        if self._uv0 and self._uv1 and self._uv2:
            return self._uv0 * (1 - u - v) + self._uv1 * u + self._uv2 * v
        else:
            return None

    def local_bounds(self):
        eps = 1e-4
        min_point = Vec3(min(self._v0.x, self._v1.x, self._v2.x) - eps,
                        min(self._v0.y, self._v1.y, self._v2.y) - eps,
                        min(self._v0.z, self._v1.z, self._v2.z) - eps)
        max_point = Vec3(max(self._v0.x, self._v1.x, self._v2.x) + eps,
                        max(self._v0.y, self._v1.y, self._v2.y) + eps,
                        max(self._v0.z, self._v1.z, self._v2.z) + eps)
        return AABB(min_point, max_point)

    def centroid(self):
        return (self._v0 + self._v1 + self._v2) / 3.0
    
    @property
    def is_infinite(self):
        return False
    
    def __repr__(self):
        return f"Triangle(v0={self._v0}, v1={self._v1}, v2={self._v2})"

    def to_dict(self):
        def _v2(value):
            return value.to_dict() if value is not None else None

        def _v3(value):
            return value.to_dict() if value is not None else None

        return {
            "type": "triangle",
            "v0": self._v0.to_dict(),
            "v1": self._v1.to_dict(),
            "v2": self._v2.to_dict(),
            "n0": _v3(self._n0),
            "n1": _v3(self._n1),
            "n2": _v3(self._n2),
            "uv0": _v2(self._uv0),
            "uv1": _v2(self._uv1),
            "uv2": _v2(self._uv2),
            "group": self._group,
        }

    @classmethod
    def from_dict(cls, data):
        def _v2(value):
            return Vec2.from_dict(value) if value is not None else None

        def _v3(value):
            return Vec3.from_dict(value) if value is not None else None

        return cls(
            Vec3.from_dict(data["v0"]),
            Vec3.from_dict(data["v1"]),
            Vec3.from_dict(data["v2"]),
            _v3(data.get("n0")),
            _v3(data.get("n1")),
            _v3(data.get("n2")),
            _v2(data.get("uv0")),
            _v2(data.get("uv1")),
            _v2(data.get("uv2")),
            group=data.get("group", "default"),
        )

    def taichi_type_id(self): return 4
    def taichi_data(self): return list(self._v0) + list(self._v1) + list(self._v2)

    @staticmethod
    @ti.func
    def taichi_intersect(ro, rd, v0, v1, v2):
        e1 = v1 - v0
        e2 = v2 - v0
        h = rd.cross(e2)
        a = e1.dot(h)
        t = -1.0
        if ti.abs(a) > 1e-8:
            f = 1.0 / a
            s = ro - v0
            u = f * s.dot(h)
            if 0.0 <= u <= 1.0:
                q = s.cross(e1)
                v = f * rd.dot(q)
                if v >= 0.0 and u + v <= 1.0:
                    t_cand = f * e2.dot(q)
                    if t_cand > 0.001:
                        t = t_cand
        return t

    @staticmethod
    @ti.func
    def taichi_normal(v0, v1, v2):
        return (v1 - v0).cross(v2 - v0).normalized()

class Mesh(Shape):
    """Mesh composed of multiple triangles. For simplicity, we can treat it as a collection of triangles."""

    def __init__(self, triangles=None, name=''):
        self._triangles = triangles or []
        self._name = name
        self._bvh = BVHNode(triangles) if triangles else None
        
    def intersect(self, ray):
        hit = self._bvh.intersect(ray) if self._bvh else self._brute_force_intersect(ray)
        if hit is None:
            return None

        closest_t, closest_u, closest_v, closest_tri = hit
        normal = closest_tri.normal_at(closest_u, closest_v)
        uv     = closest_tri.uv_at(closest_u, closest_v)
        return MeshHit(t=closest_t, normal=normal, uv=uv, group=closest_tri.group)

    def _brute_force_intersect(self, ray):
        closest_t   = float('inf')
        closest_hit = None
        for tri in self._triangles:
            hit = tri.intersect(ray)
            if hit and hit[0] < closest_t:
                closest_t = hit[0]
                closest_hit = hit
        
        return closest_hit

    def local_bounds(self):
        bounds = self._triangles[0].local_bounds()
        for tri in self._triangles[1:]:
            bounds = bounds.union(tri.local_bounds())
        return bounds
    
    @property
    def is_infinite(self):
        return False
    
    def __repr__(self):
        return f"Mesh(num_triangles={len(self._triangles)})"
    

    def to_dict(self):
        return {"type": "mesh", "triangles": [tri.to_dict() for tri in self._triangles], "name": self._name}

    @classmethod
    def from_dict(cls, data):
        return cls(
            [Triangle.from_dict(tri_data) for tri_data in data.get("triangles", [])],
            name=data.get("name", ""),
        )
    
    def taichi_type_id(self): return 5
    def taichi_data(self): return [tri.taichi_data() for tri in self._triangles]


class _IndexedBVHNode:
    """BVH node that stores triangle indices instead of Triangle objects."""

    def __init__(self, mesh, tri_indices, depth=0, max_leaf_size=4, max_depth=24):
        self.bounds = mesh._bounds_for_indices(tri_indices)

        if len(tri_indices) <= max_leaf_size or depth >= max_depth:
            self.tri_indices = np.asarray(tri_indices, dtype=np.int32)
            self.left = self.right = None
            return

        axis = self._longest_axis(self.bounds)
        order = sorted(tri_indices, key=lambda i: mesh._centroid_component(int(i), axis))
        mid = len(order) // 2
        self.left = _IndexedBVHNode(mesh, order[:mid], depth + 1, max_leaf_size, max_depth)
        self.right = _IndexedBVHNode(mesh, order[mid:], depth + 1, max_leaf_size, max_depth)
        self.tri_indices = None

    def intersect(self, mesh, ray):
        if not self.bounds.intersect(ray):
            return None

        if self.tri_indices is not None:
            closest_t = float('inf')
            closest_hit = None
            for tri_i in self.tri_indices:
                hit = mesh._intersect_triangle(ray, int(tri_i))
                if hit and hit[0] < closest_t:
                    closest_t = hit[0]
                    closest_hit = hit
            return closest_hit

        left = self.left.intersect(mesh, ray) if self.left else None
        right = self.right.intersect(mesh, ray) if self.right else None
        if left and right:
            return left if left[0] < right[0] else right
        return left or right

    def _longest_axis(self, bounds):
        extents = bounds.max - bounds.min
        if extents.x >= extents.y and extents.x >= extents.z:
            return 0
        if extents.y >= extents.z:
            return 1
        return 2


class IndexedMesh(Shape):
    """Mesh stored as shared arrays plus triangle index tables.

    This is the storage model used by most real mesh formats:

    positions
        Nx3 float array of vertex positions.
    tri_pos_idx
        Tx3 int array. Row i stores the three position indices for triangle i.
    normals / tri_normal_idx
        Optional normal array plus per-triangle normal indices. Missing normals use -1.
    uvs / tri_uv_idx
        Optional UV array plus per-triangle UV indices. Missing UVs use -1.
    groups / tri_group_idx
        String group names stored once, with each triangle holding a small integer id.

    Intersections still return MeshHit, so Shape can resolve materials exactly like
    it does for the object-per-triangle Mesh class.
    """

    def __init__(
        self,
        positions,
        tri_pos_idx,
        normals=None,
        tri_normal_idx=None,
        uvs=None,
        tri_uv_idx=None,
        groups=None,
        tri_group_idx=None,
        name='',
        build_bvh=True,
    ):
        self._positions = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        self._tri_pos_idx = np.asarray(tri_pos_idx, dtype=np.int32).reshape((-1, 3))

        self._normals = None if normals is None else np.asarray(normals, dtype=np.float64).reshape((-1, 3))
        self._tri_normal_idx = self._optional_index_table(tri_normal_idx, len(self._tri_pos_idx))

        self._uvs = None if uvs is None else np.asarray(uvs, dtype=np.float64).reshape((-1, 2))
        self._tri_uv_idx = self._optional_index_table(tri_uv_idx, len(self._tri_pos_idx))

        self._groups = list(groups) if groups else ["default"]
        if tri_group_idx is None:
            self._tri_group_idx = np.zeros(len(self._tri_pos_idx), dtype=np.int32)
        else:
            self._tri_group_idx = np.asarray(tri_group_idx, dtype=np.int32).reshape((-1,))

        self._name = name
        self._bvh = (
            _IndexedBVHNode(self, np.arange(len(self._tri_pos_idx), dtype=np.int32))
            if build_bvh and len(self._tri_pos_idx)
            else None
        )

    @staticmethod
    def _optional_index_table(value, tri_count):
        if value is None:
            return np.full((tri_count, 3), -1, dtype=np.int32)
        return np.asarray(value, dtype=np.int32).reshape((tri_count, 3))

    @classmethod
    def from_triangles(cls, triangles, name=''):
        """Build an IndexedMesh from Triangle objects without deduplicating vertices."""
        positions, normals, uvs = [], [], []
        tri_pos_idx, tri_normal_idx, tri_uv_idx, tri_group_idx = [], [], [], []
        groups, group_to_idx = [], {}

        def _group_idx(group):
            if group not in group_to_idx:
                group_to_idx[group] = len(groups)
                groups.append(group)
            return group_to_idx[group]

        for tri in triangles:
            base_pos = len(positions)
            positions.extend([list(tri._v0), list(tri._v1), list(tri._v2)])
            tri_pos_idx.append([base_pos, base_pos + 1, base_pos + 2])

            n_row = []
            for normal in (tri._n0, tri._n1, tri._n2):
                if normal is None:
                    n_row.append(-1)
                else:
                    n_row.append(len(normals))
                    normals.append(list(normal))
            tri_normal_idx.append(n_row)

            uv_row = []
            for uv in (tri._uv0, tri._uv1, tri._uv2):
                if uv is None:
                    uv_row.append(-1)
                else:
                    uv_row.append(len(uvs))
                    uvs.append(list(uv))
            tri_uv_idx.append(uv_row)
            tri_group_idx.append(_group_idx(tri.group))

        return cls(
            positions,
            tri_pos_idx,
            normals=normals or None,
            tri_normal_idx=tri_normal_idx,
            uvs=uvs or None,
            tri_uv_idx=tri_uv_idx,
            groups=groups or ["default"],
            tri_group_idx=tri_group_idx,
            name=name,
        )

    @property
    def vertex_count(self): return len(self._positions)

    @property
    def triangle_count(self): return len(self._tri_pos_idx)

    @property
    def groups(self): return list(self._groups)

    def triangle_vertices(self, tri_i):
        row = self._positions[self._tri_pos_idx[tri_i]]
        return self._vec3(row[0]), self._vec3(row[1]), self._vec3(row[2])

    def triangle_normals(self, tri_i):
        idx = self._tri_normal_idx[tri_i]
        if self._normals is None or np.any(idx < 0):
            n = self._face_normal(tri_i)
            return n, n, n
        row = self._normals[idx]
        return self._vec3(row[0]), self._vec3(row[1]), self._vec3(row[2])

    def group_for_triangle(self, tri_i):
        return self._groups[int(self._tri_group_idx[tri_i])]

    def indexed_triangle_arrays(self, matrix=None, normal_matrix=None, dtype=np.float32):
        """Return renderer-ready triangle arrays without creating Triangle objects.

        The returned arrays are shaped for one row per triangle:
            v0/v1/v2, n0/n1/n2: (triangle_count, 3)
            group_idx:          (triangle_count,)

        If matrix and normal_matrix are provided, vertices and normals are returned
        in that transformed space. This lets SceneObject keep owning transforms
        while mesh storage remains shared and local.
        """
        tri_vertices = self._positions[self._tri_pos_idx]
        tri_vertices = self._transform_points_array(tri_vertices, matrix)

        tri_normals = self._indexed_normal_array()
        tri_normals = self._transform_vectors_array(tri_normals, normal_matrix)
        tri_normals = self._normalize_rows(tri_normals.reshape((-1, 3))).reshape((-1, 3, 3))

        return {
            'v0': tri_vertices[:, 0, :].astype(dtype, copy=False),
            'v1': tri_vertices[:, 1, :].astype(dtype, copy=False),
            'v2': tri_vertices[:, 2, :].astype(dtype, copy=False),
            'n0': tri_normals[:, 0, :].astype(dtype, copy=False),
            'n1': tri_normals[:, 1, :].astype(dtype, copy=False),
            'n2': tri_normals[:, 2, :].astype(dtype, copy=False),
            'group_idx': self._tri_group_idx.astype(np.int32, copy=False),
            'groups': list(self._groups),
        }

    def intersect(self, ray):
        hit = self._bvh.intersect(self, ray) if self._bvh else self._brute_force_intersect(ray)
        if hit is None:
            return None

        t, u, v, tri_i = hit
        return MeshHit(
            t=t,
            normal=self.normal_at(tri_i, u, v),
            uv=self.uv_at(tri_i, u, v),
            group=self.group_for_triangle(tri_i),
        )

    def _brute_force_intersect(self, ray):
        closest_t = float('inf')
        closest_hit = None
        for tri_i in range(self.triangle_count):
            hit = self._intersect_triangle(ray, tri_i)
            if hit and hit[0] < closest_t:
                closest_t = hit[0]
                closest_hit = hit
        return closest_hit

    def _intersect_triangle(self, ray, tri_i):
        v0, v1, v2 = self.triangle_vertices(tri_i)
        e1 = v1 - v0
        e2 = v2 - v0
        h = ray.direction.cross(e2)
        a = e1.dot(h)
        if abs(a) < 1e-8:
            return None

        f = 1.0 / a
        s = ray.origin - v0
        u = f * s.dot(h)
        if u < 0 or u > 1:
            return None

        q = s.cross(e1)
        v = f * ray.direction.dot(q)
        if v < 0 or u + v > 1:
            return None

        t = f * e2.dot(q)
        if t < 0.001:
            return None
        return t, u, v, tri_i

    def normal_at(self, tri_i, u, v):
        idx = self._tri_normal_idx[tri_i]
        if self._normals is not None and np.all(idx >= 0):
            n0, n1, n2 = self.triangle_normals(tri_i)
            return (n0 * (1 - u - v) + n1 * u + n2 * v).normalize()
        return self._face_normal(tri_i)

    def uv_at(self, tri_i, u, v):
        idx = self._tri_uv_idx[tri_i]
        if self._uvs is None or np.any(idx < 0):
            return None
        row = self._uvs[idx]
        uv0, uv1, uv2 = self._vec2(row[0]), self._vec2(row[1]), self._vec2(row[2])
        return uv0 * (1 - u - v) + uv1 * u + uv2 * v

    def local_bounds(self):
        if self.triangle_count == 0:
            return AABB(Vec3(0, 0, 0), Vec3(0, 0, 0))
        return self._bounds_for_indices(np.arange(self.triangle_count, dtype=np.int32))

    def _bounds_for_indices(self, tri_indices):
        idx = self._tri_pos_idx[np.asarray(tri_indices, dtype=np.int32)].reshape(-1)
        pts = self._positions[idx]
        mins = pts.min(axis=0) - 1e-4
        maxs = pts.max(axis=0) + 1e-4
        return AABB(self._vec3(mins), self._vec3(maxs))

    def _centroid_component(self, tri_i, axis):
        idx = self._tri_pos_idx[tri_i]
        return float(self._positions[idx, axis].mean())

    def _face_normal(self, tri_i):
        v0, v1, v2 = self.triangle_vertices(tri_i)
        return (v1 - v0).cross(v2 - v0).normalize()

    def _indexed_normal_array(self):
        if self.triangle_count == 0:
            return np.zeros((0, 3, 3), dtype=np.float64)

        tri_vertices = self._positions[self._tri_pos_idx]
        e1 = tri_vertices[:, 1, :] - tri_vertices[:, 0, :]
        e2 = tri_vertices[:, 2, :] - tri_vertices[:, 0, :]
        face_normals = self._normalize_rows(np.cross(e1, e2))
        tri_normals = np.repeat(face_normals[:, None, :], 3, axis=1)

        if self._normals is None:
            return tri_normals

        valid = np.all(self._tri_normal_idx >= 0, axis=1)
        if np.any(valid):
            tri_normals[valid] = self._normals[self._tri_normal_idx[valid]]
        return tri_normals

    @staticmethod
    def _matrix_array(matrix):
        return None if matrix is None else np.asarray(matrix.rows, dtype=np.float64)

    @classmethod
    def _transform_points_array(cls, points, matrix):
        out = np.asarray(points, dtype=np.float64)
        m = cls._matrix_array(matrix)
        if m is None:
            return out.copy()

        flat = out.reshape((-1, 3))
        transformed = flat @ m[:3, :3].T + m[:3, 3]
        w = flat @ m[3, :3].T + m[3, 3]
        needs_divide = np.abs(w) > 1e-12
        transformed[needs_divide] /= w[needs_divide, None]
        return transformed.reshape(out.shape)

    @classmethod
    def _transform_vectors_array(cls, vectors, matrix):
        out = np.asarray(vectors, dtype=np.float64)
        m = cls._matrix_array(matrix)
        if m is None:
            return out.copy()
        flat = out.reshape((-1, 3))
        return (flat @ m[:3, :3].T).reshape(out.shape)

    @staticmethod
    def _normalize_rows(rows):
        lengths = np.linalg.norm(rows, axis=1)
        out = rows.copy()
        valid = lengths > 1e-12
        out[valid] /= lengths[valid, None]
        return out

    @staticmethod
    def _vec3(row):
        return Vec3(float(row[0]), float(row[1]), float(row[2]))

    @staticmethod
    def _vec2(row):
        return Vec2(float(row[0]), float(row[1]))

    @property
    def is_infinite(self):
        return False

    def __repr__(self):
        return (f"IndexedMesh(vertices={self.vertex_count}, "
                f"triangles={self.triangle_count}, groups={len(self._groups)})")

    def to_dict(self):
        return {
            "type": "indexed_mesh",
            "name": self._name,
            "positions": self._positions.tolist(),
            "tri_pos_idx": self._tri_pos_idx.tolist(),
            "normals": None if self._normals is None else self._normals.tolist(),
            "tri_normal_idx": self._tri_normal_idx.tolist(),
            "uvs": None if self._uvs is None else self._uvs.tolist(),
            "tri_uv_idx": self._tri_uv_idx.tolist(),
            "groups": list(self._groups),
            "tri_group_idx": self._tri_group_idx.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["positions"],
            data["tri_pos_idx"],
            normals=data.get("normals"),
            tri_normal_idx=data.get("tri_normal_idx"),
            uvs=data.get("uvs"),
            tri_uv_idx=data.get("tri_uv_idx"),
            groups=data.get("groups"),
            tri_group_idx=data.get("tri_group_idx"),
            name=data.get("name", ""),
        )

    def taichi_type_id(self): return 6


def indexed_triangle_arrays(mesh, matrix=None, normal_matrix=None, dtype=np.float32):
    """Return triangle arrays for an IndexedMesh.

    Kept as a tiny public helper so renderer code can use the same name whether
    this later becomes a cache, view object, or shared mesh-buffer API.
    """
    if not isinstance(mesh, IndexedMesh):
        raise TypeError("indexed_triangle_arrays expects an IndexedMesh")
    return mesh.indexed_triangle_arrays(matrix, normal_matrix, dtype=dtype)
