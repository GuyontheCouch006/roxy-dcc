from dataclasses import dataclass

import numpy as np

import core.timing as timing


@dataclass
class TriangleBatch:
    """Array-backed triangle data for BVH construction."""

    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    n0: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    mat_type: np.ndarray
    albedo: np.ndarray
    roughness: np.ndarray
    ior: np.ndarray
    emission: np.ndarray
    uv0: np.ndarray = None
    uv1: np.ndarray = None
    uv2: np.ndarray = None
    has_uv: np.ndarray = None
    texture_idx: np.ndarray = None
    mat_idx: np.ndarray = None
    mat_palette: list = None

    def __post_init__(self):
        self.v0 = self._vec3_array(self.v0)
        self.v1 = self._vec3_array(self.v1)
        self.v2 = self._vec3_array(self.v2)
        self.n0 = self._vec3_array(self.n0)
        self.n1 = self._vec3_array(self.n1)
        self.n2 = self._vec3_array(self.n2)
        self.mat_type = np.asarray(self.mat_type, dtype=np.int32).reshape((-1,))
        self.albedo = self._vec3_array(self.albedo)
        self.roughness = np.asarray(self.roughness, dtype=np.float32).reshape((-1,))
        self.ior = np.asarray(self.ior, dtype=np.float32).reshape((-1,))
        self.emission = np.asarray(self.emission, dtype=np.float32).reshape((-1,))
        self.uv0 = self._vec2_array(self.uv0, self.triangle_count)
        self.uv1 = self._vec2_array(self.uv1, self.triangle_count)
        self.uv2 = self._vec2_array(self.uv2, self.triangle_count)
        if self.has_uv is None:
            self.has_uv = np.zeros(self.triangle_count, dtype=np.int32)
        else:
            self.has_uv = np.asarray(self.has_uv, dtype=np.int32).reshape((-1,))
        texture_count = self.triangle_count if self.mat_idx is None else len(self.mat_type)
        if self.texture_idx is None:
            self.texture_idx = np.full(texture_count, -1, dtype=np.int32)
        else:
            self.texture_idx = np.asarray(self.texture_idx, dtype=np.int32).reshape((-1,))

        tri_lengths = {
            len(self.v0), len(self.v1), len(self.v2),
            len(self.n0), len(self.n1), len(self.n2),
            len(self.uv0), len(self.uv1), len(self.uv2), len(self.has_uv),
        }
        if len(tri_lengths) != 1:
            raise ValueError("TriangleBatch geometry arrays must all have the same length")

        if self.mat_idx is None:
            material_lengths = {
                len(self.mat_type), len(self.albedo),
                len(self.roughness), len(self.ior), len(self.emission),
                len(self.texture_idx),
            }
            if len(material_lengths) != 1 or next(iter(material_lengths)) != self.triangle_count:
                raise ValueError("TriangleBatch per-triangle material arrays must match triangle count")
        else:
            self.mat_idx = np.asarray(self.mat_idx, dtype=np.int32).reshape((-1,))
            if len(self.mat_idx) != self.triangle_count:
                raise ValueError("TriangleBatch mat_idx must match triangle count")
            palette_lengths = {
                len(self.mat_type), len(self.albedo),
                len(self.roughness), len(self.ior), len(self.emission),
                len(self.texture_idx),
            }
            if len(palette_lengths) != 1:
                raise ValueError("TriangleBatch material palette arrays must all have the same length")
            if self.mat_palette is None:
                self.mat_palette = [
                    {
                        'type': int(self.mat_type[i]),
                        'albedo': self.albedo[i].tolist(),
                        'roughness': float(self.roughness[i]),
                        'ior': float(self.ior[i]),
                        'emission': float(self.emission[i]),
                        'texture_idx': int(self.texture_idx[i]),
                    }
                    for i in range(len(self.mat_type))
                ]

    @staticmethod
    def _vec3_array(value):
        return np.asarray(value, dtype=np.float32).reshape((-1, 3))

    @staticmethod
    def _vec2_array(value, triangle_count):
        if value is None:
            return np.zeros((triangle_count, 2), dtype=np.float32)
        return np.asarray(value, dtype=np.float32).reshape((-1, 2))

    @property
    def triangle_count(self):
        return len(self.v0)

    @classmethod
    def empty(cls):
        return cls(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            uv0=np.zeros((0, 2), dtype=np.float32),
            uv1=np.zeros((0, 2), dtype=np.float32),
            uv2=np.zeros((0, 2), dtype=np.float32),
            has_uv=np.zeros(0, dtype=np.int32),
            texture_idx=np.zeros(0, dtype=np.int32),
            mat_idx=np.zeros(0, dtype=np.int32),
            mat_palette=[],
        )

    @classmethod
    def from_dicts(cls, tris):
        if not tris:
            return cls.empty()
        return cls(
            [t['v0'] for t in tris],
            [t['v1'] for t in tris],
            [t['v2'] for t in tris],
            [t['n0'] for t in tris],
            [t['n1'] for t in tris],
            [t['n2'] for t in tris],
            [t['mat_type'] for t in tris],
            [t['albedo'] for t in tris],
            [t['roughness'] for t in tris],
            [t['ior'] for t in tris],
            [t['emission'] for t in tris],
            uv0=[t.get('uv0', [0.0, 0.0]) for t in tris],
            uv1=[t.get('uv1', [0.0, 0.0]) for t in tris],
            uv2=[t.get('uv2', [0.0, 0.0]) for t in tris],
            has_uv=[t.get('has_uv', 0) for t in tris],
            texture_idx=[t.get('texture_idx', -1) for t in tris],
        )

    @classmethod
    def from_indexed_arrays(cls, arrays, material_by_group_idx):
        group_idx = np.asarray(arrays['group_idx'], dtype=np.int32)
        used_groups = np.unique(group_idx)
        remap = np.zeros(max(int(used_groups.max()), 0) + 1, dtype=np.int32) if len(used_groups) else np.zeros(0, dtype=np.int32)
        for local_idx, source_idx in enumerate(used_groups):
            remap[int(source_idx)] = local_idx
        local_group_idx = remap[group_idx] if len(used_groups) else group_idx
        n_groups = len(used_groups)

        mat_type_by_group = np.zeros(n_groups, dtype=np.int32)
        albedo_by_group = np.zeros((n_groups, 3), dtype=np.float32)
        roughness_by_group = np.zeros(n_groups, dtype=np.float32)
        ior_by_group = np.ones(n_groups, dtype=np.float32)
        emission_by_group = np.zeros(n_groups, dtype=np.float32)
        texture_by_group = np.full(n_groups, -1, dtype=np.int32)

        for local_idx, source_idx in enumerate(used_groups):
            mat = material_by_group_idx[int(source_idx)]
            mat_type_by_group[local_idx] = mat['mat_type']
            albedo_by_group[local_idx] = mat['albedo']
            roughness_by_group[local_idx] = mat['roughness']
            ior_by_group[local_idx] = mat['ior']
            emission_by_group[local_idx] = mat['emission']
            texture_by_group[local_idx] = mat.get('texture_idx', -1)
        palette = [
            {
                'type': int(mat_type_by_group[i]),
                'albedo': albedo_by_group[i].tolist(),
                'roughness': float(roughness_by_group[i]),
                'ior': float(ior_by_group[i]),
                'emission': float(emission_by_group[i]),
                'texture_idx': int(texture_by_group[i]),
            }
            for i in range(n_groups)
        ]

        return cls(
            arrays['v0'],
            arrays['v1'],
            arrays['v2'],
            arrays['n0'],
            arrays['n1'],
            arrays['n2'],
            mat_type_by_group,
            albedo_by_group,
            roughness_by_group,
            ior_by_group,
            emission_by_group,
            uv0=arrays.get('uv0'),
            uv1=arrays.get('uv1'),
            uv2=arrays.get('uv2'),
            has_uv=arrays.get('has_uv'),
            texture_idx=texture_by_group,
            mat_idx=local_group_idx,
            mat_palette=palette,
        )

    @classmethod
    def concat(cls, batches):
        batches = [b for b in batches if b.triangle_count]
        if not batches:
            return cls.empty()
        if len(batches) == 1:
            return batches[0]
        if all(b.mat_idx is not None for b in batches):
            palette = []
            mat_idx = []
            for batch in batches:
                offset = len(palette)
                palette.extend(batch.mat_palette)
                mat_idx.append(batch.mat_idx + offset)
            mat_type = np.asarray([m['type'] for m in palette], dtype=np.int32)
            albedo = np.asarray([m['albedo'] for m in palette], dtype=np.float32).reshape((-1, 3))
            roughness = np.asarray([m['roughness'] for m in palette], dtype=np.float32)
            ior = np.asarray([m['ior'] for m in palette], dtype=np.float32)
            emission = np.asarray([m['emission'] for m in palette], dtype=np.float32)
            texture_idx = np.asarray(
                [m.get('texture_idx', -1) for m in palette], dtype=np.int32)
            return cls(
                np.concatenate([b.v0 for b in batches]),
                np.concatenate([b.v1 for b in batches]),
                np.concatenate([b.v2 for b in batches]),
                np.concatenate([b.n0 for b in batches]),
                np.concatenate([b.n1 for b in batches]),
                np.concatenate([b.n2 for b in batches]),
                mat_type,
                albedo,
                roughness,
                ior,
                emission,
                uv0=np.concatenate([b.uv0 for b in batches]),
                uv1=np.concatenate([b.uv1 for b in batches]),
                uv2=np.concatenate([b.uv2 for b in batches]),
                has_uv=np.concatenate([b.has_uv for b in batches]),
                texture_idx=texture_idx,
                mat_idx=np.concatenate(mat_idx),
                mat_palette=palette,
            )
        return cls(
            np.concatenate([b.v0 for b in batches]),
            np.concatenate([b.v1 for b in batches]),
            np.concatenate([b.v2 for b in batches]),
            np.concatenate([b.n0 for b in batches]),
            np.concatenate([b.n1 for b in batches]),
            np.concatenate([b.n2 for b in batches]),
            np.concatenate([b.mat_type for b in batches]),
            np.concatenate([b.albedo for b in batches]),
            np.concatenate([b.roughness for b in batches]),
            np.concatenate([b.ior for b in batches]),
            np.concatenate([b.emission for b in batches]),
            uv0=np.concatenate([b.uv0 for b in batches]),
            uv1=np.concatenate([b.uv1 for b in batches]),
            uv2=np.concatenate([b.uv2 for b in batches]),
            has_uv=np.concatenate([b.has_uv for b in batches]),
            texture_idx=np.concatenate([b.texture_idx for b in batches]),
        )


class GPUBVHBuilder:
    """Builds a BVH on the CPU and uploads it to Taichi fields.

    Triangles are reordered into BVH leaf order.
    Materials are deduplicated into a shared palette.
    """

    MAX_LEAF_SIZE = 8
    MAX_DEPTH     = 32
    BUILD_METHOD  = "morton"
    MORTON_BITS   = 10

    def __init__(self):
        self.nodes           = []
        self.ordered_tris    = np.zeros(0, dtype=np.int32)
        self.mat_palette     = []
        self._mat_key_to_idx = {}
        self._ordered_index_chunks = []
        self._ordered_count = 0
        self._batch = TriangleBatch.empty()
        self._tri_bounds_min = np.zeros((0, 3), dtype=np.float32)
        self._tri_bounds_max = np.zeros((0, 3), dtype=np.float32)
        self._centroids = np.zeros((0, 3), dtype=np.float32)
        self._source_mat_idx = np.zeros(0, dtype=np.int32)
        self._ordered_v0 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_v1 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_v2 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_n0 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_n1 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_n2 = np.zeros((0, 3), dtype=np.float32)
        self._ordered_uv0 = np.zeros((0, 2), dtype=np.float32)
        self._ordered_uv1 = np.zeros((0, 2), dtype=np.float32)
        self._ordered_uv2 = np.zeros((0, 2), dtype=np.float32)
        self._ordered_has_uv = np.zeros(0, dtype=np.int32)
        self._ordered_mat_idx = np.zeros(0, dtype=np.int32)

    @property
    def triangle_count(self):
        return len(self.ordered_tris)

    @timing.timer("BVH build", tag="bvh")
    def build(self, tri_data, method=None):
        """Build BVH from TriangleBatch objects or legacy triangle dicts."""
        batches = self._coerce_batches(tri_data)
        batch = TriangleBatch.concat(batches)
        self._reset_for_build(batch)

        if batch.triangle_count:
            method = method or self.BUILD_METHOD
            if method == "morton":
                self._build_morton_bvh()
            elif method == "median":
                all_indices = np.arange(batch.triangle_count, dtype=np.int32)
                self._build_median_node(all_indices, depth=0)
            else:
                raise ValueError(f"Unknown BVH build method: {method}")
            self._finish_ordered_arrays()

        if timing.LEVEL >= 1:
            timing.defer_print(f"    {batch.triangle_count:,} tris → {len(self.nodes):,} nodes, "
                               f"{len(self.mat_palette)} materials")
        return self

    def _coerce_batches(self, tri_data):
        if isinstance(tri_data, TriangleBatch):
            return [tri_data]
        if not tri_data:
            return []
        if all(isinstance(item, TriangleBatch) for item in tri_data):
            return list(tri_data)
        return [TriangleBatch.from_dicts(tri_data)]

    def _reset_for_build(self, batch):
        self.nodes = []
        self.ordered_tris = np.zeros(0, dtype=np.int32)
        self._ordered_index_chunks = []
        self._ordered_count = 0
        self._mat_key_to_idx = {}
        self.mat_palette = []
        self._batch = batch

        if batch.triangle_count == 0:
            self._tri_bounds_min = np.zeros((0, 3), dtype=np.float32)
            self._tri_bounds_max = np.zeros((0, 3), dtype=np.float32)
            self._centroids = np.zeros((0, 3), dtype=np.float32)
            self._source_mat_idx = np.zeros(0, dtype=np.int32)
            return

        stacked = np.stack([batch.v0, batch.v1, batch.v2], axis=1)
        eps = np.float32(1e-4)
        self._tri_bounds_min = stacked.min(axis=1) - eps
        self._tri_bounds_max = stacked.max(axis=1) + eps
        self._centroids = (batch.v0 + batch.v1 + batch.v2) / np.float32(3.0)
        self._source_mat_idx = self._build_material_indices(batch)

    def _build_material_indices(self, batch):
        if batch.mat_idx is not None:
            remap = np.empty(len(batch.mat_palette), dtype=np.int32)
            for i, mat in enumerate(batch.mat_palette):
                remap[i] = self._get_or_add_material(mat)
            return remap[batch.mat_idx]

        mat_idx = np.empty(batch.triangle_count, dtype=np.int32)
        for i in range(batch.triangle_count):
            mat_idx[i] = self._get_or_add_material({
                'mat_type': int(batch.mat_type[i]),
                'albedo': batch.albedo[i].tolist(),
                'roughness': float(batch.roughness[i]),
                'ior': float(batch.ior[i]),
                'emission': float(batch.emission[i]),
                'texture_idx': int(batch.texture_idx[i]),
            })
        return mat_idx

    def _get_or_add_material(self, mat):
        mat_type = mat.get('mat_type', mat.get('type'))
        texture_idx = int(mat.get('texture_idx', -1))
        key = (
            int(mat_type),
            tuple(float(v) for v in mat['albedo']),
            float(mat['roughness']),
            float(mat['ior']),
            float(mat['emission']),
            texture_idx,
        )
        if key not in self._mat_key_to_idx:
            self._mat_key_to_idx[key] = len(self.mat_palette)
            self.mat_palette.append({
                'type': key[0],
                'albedo': list(key[1]),
                'roughness': key[2],
                'ior': key[3],
                'emission': key[4],
                'texture_idx': key[5],
            })
        return self._mat_key_to_idx[key]

    def _build_morton_bvh(self):
        order = self._morton_order()
        self._build_morton_node(order, 0, len(order))

    def _morton_order(self):
        mins = self._centroids.min(axis=0)
        maxs = self._centroids.max(axis=0)
        extent = maxs - mins
        extent[extent <= 1e-12] = 1.0

        scale = np.float32((1 << self.MORTON_BITS) - 1)
        normalized = np.clip((self._centroids - mins) / extent, 0.0, 1.0)
        coords = (normalized * scale).astype(np.uint32)
        codes = self._morton3(coords[:, 0], coords[:, 1], coords[:, 2])
        return np.argsort(codes, kind='stable').astype(np.int32, copy=False)

    @classmethod
    def _morton3(cls, x, y, z):
        return (
            (cls._expand_morton_bits(x) << np.uint32(2))
            | (cls._expand_morton_bits(y) << np.uint32(1))
            | cls._expand_morton_bits(z)
        )

    @staticmethod
    def _expand_morton_bits(v):
        v = np.asarray(v, dtype=np.uint32) & np.uint32(0x000003ff)
        v = (v | (v << np.uint32(16))) & np.uint32(0x030000ff)
        v = (v | (v << np.uint32(8))) & np.uint32(0x0300f00f)
        v = (v | (v << np.uint32(4))) & np.uint32(0x030c30c3)
        v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
        return v

    def _build_morton_node(self, sorted_indices, start, end):
        node_idx = len(self.nodes)
        self.nodes.append({})

        count = end - start
        if count <= self.MAX_LEAF_SIZE:
            tri_indices = sorted_indices[start:end]
            aabb_min, aabb_max = self._compute_bounds(tri_indices)
            tri_start = self._ordered_count
            self._ordered_index_chunks.append(tri_indices)
            self._ordered_count += count
            self.nodes[node_idx] = {
                'aabb_min':  aabb_min,
                'aabb_max':  aabb_max,
                'left':      -1,
                'right':     -1,
                'tri_start': tri_start,
                'tri_count': count,
            }
        else:
            mid = (start + end) // 2
            left_idx = self._build_morton_node(sorted_indices, start, mid)
            right_idx = self._build_morton_node(sorted_indices, mid, end)
            aabb_min, aabb_max = self._combine_node_bounds(left_idx, right_idx)
            self.nodes[node_idx] = {
                'aabb_min':  aabb_min,
                'aabb_max':  aabb_max,
                'left':      left_idx,
                'right':     right_idx,
                'tri_start': -1,
                'tri_count': 0,
            }

        return node_idx

    def _combine_node_bounds(self, left_idx, right_idx):
        left = self.nodes[left_idx]
        right = self.nodes[right_idx]
        return (
            np.minimum(left['aabb_min'], right['aabb_min']).tolist(),
            np.maximum(left['aabb_max'], right['aabb_max']).tolist(),
        )

    def _build_median_node(self, tri_indices, depth):
        node_idx = len(self.nodes)
        self.nodes.append({})   # placeholder — filled below

        aabb_min, aabb_max = self._compute_bounds(tri_indices)

        if len(tri_indices) <= self.MAX_LEAF_SIZE or depth >= self.MAX_DEPTH:
            tri_start = self._ordered_count
            self._ordered_index_chunks.append(tri_indices)
            self._ordered_count += len(tri_indices)
            self.nodes[node_idx] = {
                'aabb_min':  aabb_min,
                'aabb_max':  aabb_max,
                'left':      -1,
                'right':     -1,
                'tri_start': tri_start,
                'tri_count': len(tri_indices),
            }
        else:
            axis = self._longest_axis(aabb_min, aabb_max)
            mid = len(tri_indices) // 2
            centroids = self._centroids[tri_indices, axis]
            order = np.argpartition(centroids, mid)
            partitioned_indices = tri_indices[order]

            left_idx  = self._build_median_node(partitioned_indices[:mid], depth + 1)
            right_idx = self._build_median_node(partitioned_indices[mid:], depth + 1)

            self.nodes[node_idx] = {
                'aabb_min':  aabb_min,
                'aabb_max':  aabb_max,
                'left':      left_idx,
                'right':     right_idx,
                'tri_start': -1,
                'tri_count': 0,
            }

        return node_idx

    def _finish_ordered_arrays(self):
        if self._ordered_index_chunks:
            self.ordered_tris = np.concatenate(self._ordered_index_chunks).astype(np.int32, copy=False)
        else:
            self.ordered_tris = np.zeros(0, dtype=np.int32)

        order = self.ordered_tris
        self._ordered_v0 = self._batch.v0[order]
        self._ordered_v1 = self._batch.v1[order]
        self._ordered_v2 = self._batch.v2[order]
        self._ordered_n0 = self._batch.n0[order]
        self._ordered_n1 = self._batch.n1[order]
        self._ordered_n2 = self._batch.n2[order]
        self._ordered_uv0 = self._batch.uv0[order]
        self._ordered_uv1 = self._batch.uv1[order]
        self._ordered_uv2 = self._batch.uv2[order]
        self._ordered_has_uv = self._batch.has_uv[order]
        self._ordered_mat_idx = self._source_mat_idx[order]

    def _compute_bounds(self, tri_indices):
        return (
            self._tri_bounds_min[tri_indices].min(axis=0).tolist(),
            self._tri_bounds_max[tri_indices].max(axis=0).tolist(),
        )

    def _longest_axis(self, aabb_min, aabb_max):
        ex, ey, ez = np.asarray(aabb_max) - np.asarray(aabb_min)
        if ex >= ey and ex >= ez:
            return 0
        if ey >= ez:
            return 1
        return 2

    @timing.timer("BVH upload", tag="bvh")
    def upload(self):
        """Upload BVH nodes, triangles, and material palette to Taichi fields via from_numpy."""
        from rendering.taichi.fields import (
            _bvh_aabb_min, _bvh_aabb_max, _bvh_left, _bvh_right,
            _bvh_tri_start, _bvh_tri_count, _bvh_n_nodes,
            _bvh_v0, _bvh_v1, _bvh_v2,
            _bvh_n0, _bvh_n1, _bvh_n2,
            _bvh_uv0, _bvh_uv1, _bvh_uv2, _bvh_has_uv,
            _bvh_mat_idx, _bvh_n_tris,
            _mat_type, _mat_albedo, _mat_roughness,
            _mat_ior, _mat_emission, _mat_texture, _mat_n_mats,
            MAX_BVH_NODES, MAX_TRIANGLES, MAX_MATERIALS,
        )

        assert len(self.nodes) <= MAX_BVH_NODES, \
            f"BVH has {len(self.nodes)} nodes, limit is {MAX_BVH_NODES}"
        assert self.triangle_count <= MAX_TRIANGLES, \
            f"BVH has {self.triangle_count} triangles, limit is {MAX_TRIANGLES}"
        assert len(self.mat_palette) <= MAX_MATERIALS, \
            f"BVH has {len(self.mat_palette)} materials, limit is {MAX_MATERIALS}"

        # from_numpy requires the array shape to exactly match the field shape,
        # so allocate full-capacity arrays and fill only the used prefix.

        n_nodes   = len(self.nodes)
        aabb_min  = np.zeros((MAX_BVH_NODES, 3), dtype=np.float32)
        aabb_max  = np.zeros((MAX_BVH_NODES, 3), dtype=np.float32)
        left      = np.zeros(MAX_BVH_NODES, dtype=np.int32)
        right     = np.zeros(MAX_BVH_NODES, dtype=np.int32)
        tri_start = np.zeros(MAX_BVH_NODES, dtype=np.int32)
        tri_count = np.zeros(MAX_BVH_NODES, dtype=np.int32)
        for i, node in enumerate(self.nodes):
            aabb_min[i]  = node['aabb_min']
            aabb_max[i]  = node['aabb_max']
            left[i]      = node['left']
            right[i]     = node['right']
            tri_start[i] = node['tri_start']
            tri_count[i] = node['tri_count']
        _bvh_aabb_min.from_numpy(aabb_min)
        _bvh_aabb_max.from_numpy(aabb_max)
        _bvh_left.from_numpy(left)
        _bvh_right.from_numpy(right)
        _bvh_tri_start.from_numpy(tri_start)
        _bvh_tri_count.from_numpy(tri_count)
        _bvh_n_nodes[None] = n_nodes

        n_tris  = self.triangle_count
        v0      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        v1      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        v2      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n0      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n1      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n2      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        uv0     = np.zeros((MAX_TRIANGLES, 2), dtype=np.float32)
        uv1     = np.zeros((MAX_TRIANGLES, 2), dtype=np.float32)
        uv2     = np.zeros((MAX_TRIANGLES, 2), dtype=np.float32)
        has_uv  = np.zeros(MAX_TRIANGLES, dtype=np.int32)
        mat_idx = np.zeros(MAX_TRIANGLES, dtype=np.int32)
        v0[:n_tris] = self._ordered_v0
        v1[:n_tris] = self._ordered_v1
        v2[:n_tris] = self._ordered_v2
        n0[:n_tris] = self._ordered_n0
        n1[:n_tris] = self._ordered_n1
        n2[:n_tris] = self._ordered_n2
        uv0[:n_tris] = self._ordered_uv0
        uv1[:n_tris] = self._ordered_uv1
        uv2[:n_tris] = self._ordered_uv2
        has_uv[:n_tris] = self._ordered_has_uv
        mat_idx[:n_tris] = self._ordered_mat_idx
        _bvh_v0.from_numpy(v0)
        _bvh_v1.from_numpy(v1)
        _bvh_v2.from_numpy(v2)
        _bvh_n0.from_numpy(n0)
        _bvh_n1.from_numpy(n1)
        _bvh_n2.from_numpy(n2)
        _bvh_uv0.from_numpy(uv0)
        _bvh_uv1.from_numpy(uv1)
        _bvh_uv2.from_numpy(uv2)
        _bvh_has_uv.from_numpy(has_uv)
        _bvh_mat_idx.from_numpy(mat_idx)
        _bvh_n_tris[None] = n_tris

        n_mats = len(self.mat_palette)
        for i, mat in enumerate(self.mat_palette):
            _mat_type[i]      = mat['type']
            _mat_albedo[i]    = mat['albedo']
            _mat_roughness[i] = mat['roughness']
            _mat_ior[i]       = mat['ior']
            _mat_emission[i]  = mat['emission']
            _mat_texture[i]   = mat.get('texture_idx', -1)
        _mat_n_mats[None] = n_mats

        if timing.LEVEL >= 1:
            timing.defer_print(f"    {n_nodes:,} nodes, {n_tris:,} tris → GPU")
        return self
