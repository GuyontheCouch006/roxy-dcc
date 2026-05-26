from rendering.taichi.bvh_builder import GPUBVHBuilder, TriangleBatch


def _tri(x, mat_type=0, albedo=None):
    return {
        'v0': [x, 0, 0],
        'v1': [x + 1, 0, 0],
        'v2': [x, 1, 0],
        'n0': [0, 0, 1],
        'n1': [0, 0, 1],
        'n2': [0, 0, 1],
        'mat_type': mat_type,
        'albedo': albedo or [1, 1, 1],
        'roughness': 0.0,
        'ior': 1.0,
        'emission': 0.0,
    }


def test_bvh_builder_accepts_legacy_triangle_dicts():
    builder = GPUBVHBuilder().build([_tri(0), _tri(2)])
    assert builder.triangle_count == 2
    assert len(builder.nodes) == 1
    assert len(builder.mat_palette) == 1


def test_bvh_builder_accepts_triangle_batches():
    batch = TriangleBatch.from_dicts([_tri(0), _tri(2, albedo=[0.5, 0.5, 0.5])])
    builder = GPUBVHBuilder().build([batch])
    assert builder.triangle_count == 2
    assert len(builder.nodes) == 1
    assert len(builder.mat_palette) == 2
    assert builder._ordered_v0.shape == (2, 3)


def test_bvh_builder_splits_batch_by_centroid_axis():
    batch = TriangleBatch.from_dicts([_tri(x) for x in range(8)])
    builder = GPUBVHBuilder()
    builder.MAX_LEAF_SIZE = 4
    builder.build([batch])
    assert builder.triangle_count == 8
    assert len(builder.nodes) == 3
    assert builder.nodes[0]['left'] > 0
    assert builder.nodes[0]['right'] > 0


def test_bvh_builder_partition_split_separates_centroid_halves():
    batch = TriangleBatch.from_dicts([_tri(x) for x in [7, 1, 6, 0, 5, 2, 4, 3]])
    builder = GPUBVHBuilder()
    builder.MAX_LEAF_SIZE = 1
    builder.build([batch], method="median")

    def check_node(node_idx):
        node = builder.nodes[node_idx]
        left_idx = node['left']
        right_idx = node['right']
        if left_idx == -1:
            return

        left = builder.nodes[left_idx]
        right = builder.nodes[right_idx]
        axis = builder._longest_axis(node['aabb_min'], node['aabb_max'])
        assert left['aabb_max'][axis] <= right['aabb_max'][axis]
        check_node(left_idx)
        check_node(right_idx)

    check_node(0)


def test_bvh_builder_default_uses_morton_order():
    batch = TriangleBatch.from_dicts([_tri(x) for x in [7, 1, 6, 0, 5, 2, 4, 3]])
    builder = GPUBVHBuilder()
    builder.MAX_LEAF_SIZE = 1
    builder.build([batch])
    assert list(builder._ordered_v0[:, 0]) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_bvh_builder_sah_path_splits_large_nodes():
    tri_count = GPUBVHBuilder.SAH_HYBRID_LIMIT + 32
    batch = TriangleBatch.from_dicts([_tri(x) for x in range(tri_count)])

    builder = GPUBVHBuilder()
    builder.MAX_LEAF_SIZE = 4
    builder.build([batch], method="sah")

    root = builder.nodes[0]
    leaf_counts = [
        node["tri_count"]
        for node in builder.nodes
        if node["left"] == -1 and node["right"] == -1
    ]

    assert root["left"] != -1
    assert root["right"] != -1
    assert sum(leaf_counts) == tri_count
    assert set(builder.ordered_tris.tolist()) == set(range(tri_count))


def test_bvh_builder_sah_degenerate_centroids_become_single_leaf():
    tri_count = GPUBVHBuilder.SAH_HYBRID_LIMIT + 32
    batch = TriangleBatch.from_dicts([_tri(0) for _ in range(tri_count)])

    builder = GPUBVHBuilder()
    builder.MAX_LEAF_SIZE = 4
    builder.build([batch], method="sah")

    assert len(builder.nodes) == 1
    assert builder.nodes[0]["tri_count"] == tri_count
    assert set(builder.ordered_tris.tolist()) == set(range(tri_count))


def test_indexed_triangle_batch_keeps_compact_material_palette():
    arrays = {
        'v0': [[0, 0, 0], [2, 0, 0], [4, 0, 0]],
        'v1': [[1, 0, 0], [3, 0, 0], [5, 0, 0]],
        'v2': [[0, 1, 0], [2, 1, 0], [4, 1, 0]],
        'n0': [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        'n1': [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        'n2': [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        'uv0': [[0, 0], [0, 0], [0, 0]],
        'uv1': [[1, 0], [1, 0], [1, 0]],
        'uv2': [[0, 1], [0, 1], [0, 1]],
        'has_uv': [1, 1, 1],
        'group_idx': [0, 1, 0],
    }
    materials = {
        0: {'mat_type': 0, 'albedo': [1, 0, 0], 'roughness': 0, 'ior': 1, 'emission': 0, 'texture_idx': 3},
        1: {'mat_type': 1, 'albedo': [0, 1, 0], 'roughness': 0.25, 'ior': 1, 'emission': 0, 'texture_idx': -1},
        2: {'mat_type': 0, 'albedo': [0, 0, 1], 'roughness': 0, 'ior': 1, 'emission': 0, 'texture_idx': -1},
    }
    batch = TriangleBatch.from_indexed_arrays(arrays, materials)
    assert batch.triangle_count == 3
    assert len(batch.mat_palette) == 2
    assert len(batch.mat_type) == 2
    assert list(batch.mat_idx) == [0, 1, 0]
    assert list(batch.has_uv) == [1, 1, 1]
    assert batch.mat_palette[0]['texture_idx'] == 3

    builder = GPUBVHBuilder().build([batch])
    assert len(builder.mat_palette) == 2
    assert builder._ordered_mat_idx.shape == (3,)
    assert builder._ordered_uv1.shape == (3, 2)


if __name__ == "__main__":
    from tests.utils import run_tests

    run_tests([
        test_bvh_builder_accepts_legacy_triangle_dicts,
        test_bvh_builder_accepts_triangle_batches,
        test_bvh_builder_splits_batch_by_centroid_axis,
        test_bvh_builder_partition_split_separates_centroid_halves,
        test_bvh_builder_default_uses_morton_order,
        test_bvh_builder_sah_path_splits_large_nodes,
        test_bvh_builder_sah_degenerate_centroids_become_single_leaf,
        test_indexed_triangle_batch_keeps_compact_material_palette,
    ])
