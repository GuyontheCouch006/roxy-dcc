from core.aabb import AABB
from core.vectors import Vec3


class GPUBVHBuilder:
    """Builds a BVH on the CPU and uploads it to Taichi fields.

    Triangles are reordered into BVH leaf order.
    Materials are deduplicated into a shared palette.
    """

    MAX_LEAF_SIZE = 4
    MAX_DEPTH     = 32

    def __init__(self):
        self.nodes           = []
        self.ordered_tris    = []
        self.mat_palette     = []
        self._mat_key_to_idx = {}

    def build(self, tri_data_list):
        """Build BVH from a list of triangle dicts.

        Each dict has:
            v0, v1, v2  – [x,y,z] world-space vertices
            n0, n1, n2  – [x,y,z] vertex normals
            mat_type    – int
            albedo      – [r,g,b]
            roughness   – float
            ior         – float
            emission    – float
        """
        self.nodes           = []
        self.ordered_tris    = []
        self._mat_key_to_idx = {}
        self.mat_palette     = []
        self._build_node(tri_data_list, depth=0)
        return self

    def _build_node(self, tris, depth):
        node_idx = len(self.nodes)
        self.nodes.append({})   # placeholder — filled below

        aabb = self._compute_bounds(tris)

        if len(tris) <= self.MAX_LEAF_SIZE or depth >= self.MAX_DEPTH:
            tri_start = len(self.ordered_tris)
            for tri in tris:
                mat_idx = self._get_or_add_material(tri)
                self.ordered_tris.append({**tri, 'mat_idx': mat_idx})
            self.nodes[node_idx] = {
                'aabb_min':  [aabb.min.x, aabb.min.y, aabb.min.z],
                'aabb_max':  [aabb.max.x, aabb.max.y, aabb.max.z],
                'left':      -1,
                'right':     -1,
                'tri_start': tri_start,
                'tri_count': len(tris),
            }
        else:
            axis       = self._longest_axis(aabb)
            tris_sorted = sorted(
                tris,
                key=lambda t: (t['v0'][axis] + t['v1'][axis] + t['v2'][axis]) / 3.0,
            )
            mid = len(tris_sorted) // 2

            left_idx  = self._build_node(tris_sorted[:mid], depth + 1)
            right_idx = self._build_node(tris_sorted[mid:], depth + 1)

            self.nodes[node_idx] = {
                'aabb_min':  [aabb.min.x, aabb.min.y, aabb.min.z],
                'aabb_max':  [aabb.max.x, aabb.max.y, aabb.max.z],
                'left':      left_idx,
                'right':     right_idx,
                'tri_start': -1,
                'tri_count': 0,
            }

        return node_idx

    def _get_or_add_material(self, tri):
        key = (
            tri['mat_type'],
            tuple(tri['albedo']),
            tri['roughness'],
            tri['ior'],
            tri['emission'],
        )
        if key not in self._mat_key_to_idx:
            idx = len(self.mat_palette)
            self.mat_palette.append({
                'type':      tri['mat_type'],
                'albedo':    tri['albedo'],
                'roughness': tri['roughness'],
                'ior':       tri['ior'],
                'emission':  tri['emission'],
            })
            self._mat_key_to_idx[key] = idx
        return self._mat_key_to_idx[key]

    def _compute_bounds(self, tris):
        xs = [v[0] for t in tris for v in (t['v0'], t['v1'], t['v2'])]
        ys = [v[1] for t in tris for v in (t['v0'], t['v1'], t['v2'])]
        zs = [v[2] for t in tris for v in (t['v0'], t['v1'], t['v2'])]
        eps = 1e-4
        return AABB(
            Vec3(min(xs) - eps, min(ys) - eps, min(zs) - eps),
            Vec3(max(xs) + eps, max(ys) + eps, max(zs) + eps),
        )

    def _longest_axis(self, aabb):
        ex = aabb.max.x - aabb.min.x
        ey = aabb.max.y - aabb.min.y
        ez = aabb.max.z - aabb.min.z
        if ex >= ey and ex >= ez:
            return 0
        if ey >= ez:
            return 1
        return 2

    def upload(self):
        """Upload BVH nodes, triangles, and material palette to Taichi fields."""
        from rendering.taichi.fields import (
            _bvh_aabb_min, _bvh_aabb_max, _bvh_left, _bvh_right,
            _bvh_tri_start, _bvh_tri_count, _bvh_n_nodes,
            _bvh_v0, _bvh_v1, _bvh_v2,
            _bvh_n0, _bvh_n1, _bvh_n2,
            _bvh_mat_idx, _bvh_n_tris,
            _mat_type, _mat_albedo, _mat_roughness,
            _mat_ior, _mat_emission, _mat_n_mats,
            MAX_BVH_NODES, MAX_TRIANGLES, MAX_MATERIALS,
        )

        assert len(self.nodes) <= MAX_BVH_NODES, \
            f"BVH has {len(self.nodes)} nodes, limit is {MAX_BVH_NODES}"
        assert len(self.ordered_tris) <= MAX_TRIANGLES, \
            f"BVH has {len(self.ordered_tris)} triangles, limit is {MAX_TRIANGLES}"
        assert len(self.mat_palette) <= MAX_MATERIALS, \
            f"BVH has {len(self.mat_palette)} materials, limit is {MAX_MATERIALS}"

        import numpy as np

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

        n_tris  = len(self.ordered_tris)
        v0      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        v1      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        v2      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n0      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n1      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        n2      = np.zeros((MAX_TRIANGLES, 3), dtype=np.float32)
        mat_idx = np.zeros(MAX_TRIANGLES, dtype=np.int32)
        for i, tri in enumerate(self.ordered_tris):
            v0[i]      = tri['v0']
            v1[i]      = tri['v1']
            v2[i]      = tri['v2']
            n0[i]      = tri['n0']
            n1[i]      = tri['n1']
            n2[i]      = tri['n2']
            mat_idx[i] = tri['mat_idx']
        _bvh_v0.from_numpy(v0)
        _bvh_v1.from_numpy(v1)
        _bvh_v2.from_numpy(v2)
        _bvh_n0.from_numpy(n0)
        _bvh_n1.from_numpy(n1)
        _bvh_n2.from_numpy(n2)
        _bvh_mat_idx.from_numpy(mat_idx)
        _bvh_n_tris[None] = n_tris

        n_mats = len(self.mat_palette)
        for i, mat in enumerate(self.mat_palette):
            _mat_type[i]      = mat['type']
            _mat_albedo[i]    = mat['albedo']
            _mat_roughness[i] = mat['roughness']
            _mat_ior[i]       = mat['ior']
            _mat_emission[i]  = mat['emission']
        _mat_n_mats[None] = n_mats

        return self
