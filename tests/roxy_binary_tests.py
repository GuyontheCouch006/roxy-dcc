import os
import tempfile

import numpy as np

from scene.io.roxy_binary import (
    list_rxb_meshes,
    load_rxb_metadata,
    load_rxb_mesh,
    load_rxb_mesh_ref,
    save_rxb_meshes,
)
from scene.mesh import IndexedMesh
from tests.utils import run_tests


def _mesh():
    return IndexedMesh(
        positions=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        tri_pos_idx=[
            [0, 1, 2],
            [0, 2, 3],
        ],
        normals=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        tri_normal_idx=[
            [0, 1, 2],
            [0, 2, 3],
        ],
        uvs=[
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        tri_uv_idx=[
            [0, 1, 2],
            [0, 2, 3],
        ],
        groups=["paint", "rubber"],
        tri_group_idx=[0, 1],
        name="testMesh",
        build_bvh=False,
    )


def _temp_path(suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def test_save_and_load_rxb_indexed_mesh_payload():
    path = _temp_path(".rxb")
    try:
        source = _mesh()
        save_rxb_meshes(path, {"meshA": source})

        restored = load_rxb_mesh(path, "meshA")

        assert restored.vertex_count == source.vertex_count
        assert restored.triangle_count == source.triangle_count
        assert restored.groups == ["paint", "rubber"]
        assert restored.group_for_triangle(1) == "rubber"
        assert np.allclose(restored._positions, source._positions)
        assert np.array_equal(restored._tri_pos_idx, source._tri_pos_idx)
        assert np.allclose(restored._normals, source._normals)
        assert np.allclose(restored._uvs, source._uvs)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_list_rxb_meshes_returns_payload_names():
    path = _temp_path(".rxb")
    try:
        save_rxb_meshes(path, {"meshA": _mesh(), "meshB": _mesh()})

        assert list_rxb_meshes(path) == ["meshA", "meshB"]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_load_rxb_mesh_ref_resolves_relative_path():
    path = _temp_path(".rxb")
    try:
        save_rxb_meshes(path, {"meshA": _mesh()})
        ref = f"{os.path.basename(path)}:meshes/meshA"

        restored = load_rxb_mesh_ref(ref, base_dir=os.path.dirname(path))

        assert restored.triangle_count == 2
        assert restored.group_for_triangle(0) == "paint"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_shared_vertex_buffers_are_saved_once_and_reused_on_load():
    path = _temp_path(".rxb")
    positions = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    mesh_a = IndexedMesh(
        positions,
        [[0, 1, 2]],
        groups=["default"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    mesh_b = IndexedMesh(
        positions,
        [[0, 2, 3]],
        groups=["default"],
        tri_group_idx=[0],
        build_bvh=False,
    )
    try:
        save_rxb_meshes(path, {"meshA": mesh_a, "meshB": mesh_b})
        metadata = load_rxb_metadata(path)
        position_buffers = [
            buffer for buffer in metadata["buffers"].values()
            if buffer["kind"] == "positions"
        ]
        cache = {}

        restored_a = load_rxb_mesh(path, "meshA", cache=cache)
        restored_b = load_rxb_mesh(path, "meshB", cache=cache)

        assert len(position_buffers) == 1
        assert np.shares_memory(restored_a._positions, restored_b._positions)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    run_tests([
        test_save_and_load_rxb_indexed_mesh_payload,
        test_list_rxb_meshes_returns_payload_names,
        test_load_rxb_mesh_ref_resolves_relative_path,
        test_shared_vertex_buffers_are_saved_once_and_reused_on_load,
    ])
