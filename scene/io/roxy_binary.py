import json
from pathlib import Path

import numpy as np

from scene.mesh import IndexedMesh


RXB_FORMAT_VERSION = 1
_METADATA_KEY = "__metadata__"


def save_rxb_meshes(path, meshes):
    """Save named IndexedMesh payloads to a Roxy Binary file.

    The first RXB version is intentionally narrow: it stores mesh arrays that
    can be loaded without OBJ parsing or triangle object creation. The container
    is an uncompressed NumPy archive so each array remains directly addressable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(meshes.items()) if hasattr(meshes, "items") else list(meshes)

    metadata = {
        "format": "rxb",
        "version": RXB_FORMAT_VERSION,
        "meshes": {},
    }
    arrays = {}

    for index, (name, mesh) in enumerate(items):
        if not isinstance(mesh, IndexedMesh):
            raise TypeError(f"RXB mesh payload {name!r} is not an IndexedMesh")

        prefix = f"mesh_{index}"
        metadata["meshes"][name] = {
            "prefix": prefix,
            "name": mesh._name,
            "groups": list(mesh._groups),
            "has_normals": mesh._normals is not None,
            "has_uvs": mesh._uvs is not None,
            "vertex_count": mesh.vertex_count,
            "triangle_count": mesh.triangle_count,
        }

        arrays[f"{prefix}.positions"] = np.asarray(mesh._positions, dtype=np.float64)
        arrays[f"{prefix}.tri_pos_idx"] = np.asarray(mesh._tri_pos_idx, dtype=np.int32)
        arrays[f"{prefix}.tri_normal_idx"] = np.asarray(mesh._tri_normal_idx, dtype=np.int32)
        arrays[f"{prefix}.tri_uv_idx"] = np.asarray(mesh._tri_uv_idx, dtype=np.int32)
        arrays[f"{prefix}.tri_group_idx"] = np.asarray(mesh._tri_group_idx, dtype=np.int32)
        if mesh._normals is not None:
            arrays[f"{prefix}.normals"] = np.asarray(mesh._normals, dtype=np.float64)
        if mesh._uvs is not None:
            arrays[f"{prefix}.uvs"] = np.asarray(mesh._uvs, dtype=np.float64)

    arrays[_METADATA_KEY] = _metadata_to_array(metadata)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_rxb_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return _metadata_from_array(archive[_METADATA_KEY])


def list_rxb_meshes(path):
    return list(load_rxb_metadata(path).get("meshes", {}).keys())


def load_rxb_mesh(path, name, build_bvh=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = _metadata_from_array(archive[_METADATA_KEY])
        _validate_metadata(metadata, path)
        try:
            mesh_meta = metadata["meshes"][name]
        except KeyError as exc:
            available = ", ".join(metadata["meshes"].keys())
            raise KeyError(f"RXB mesh {name!r} not found in {path}; available: {available}") from exc

        prefix = mesh_meta["prefix"]
        normals = (
            np.array(archive[f"{prefix}.normals"], copy=True)
            if mesh_meta.get("has_normals")
            else None
        )
        uvs = (
            np.array(archive[f"{prefix}.uvs"], copy=True)
            if mesh_meta.get("has_uvs")
            else None
        )

        return IndexedMesh(
            np.array(archive[f"{prefix}.positions"], copy=True),
            np.array(archive[f"{prefix}.tri_pos_idx"], copy=True),
            normals=normals,
            tri_normal_idx=np.array(archive[f"{prefix}.tri_normal_idx"], copy=True),
            uvs=uvs,
            tri_uv_idx=np.array(archive[f"{prefix}.tri_uv_idx"], copy=True),
            groups=mesh_meta.get("groups") or ["default"],
            tri_group_idx=np.array(archive[f"{prefix}.tri_group_idx"], copy=True),
            name=mesh_meta.get("name", name),
            build_bvh=build_bvh,
        )


def load_rxb_mesh_ref(ref, base_dir=None, build_bvh=False):
    path, mesh_name = split_rxb_mesh_ref(ref, base_dir=base_dir)
    return load_rxb_mesh(path, mesh_name, build_bvh=build_bvh)


def split_rxb_mesh_ref(ref, base_dir=None):
    if ":" not in ref:
        raise ValueError(f"RXB mesh references must look like file.rxb:meshes/name: {ref}")
    path_text, mesh_path = ref.rsplit(":", 1)
    mesh_name = mesh_path[len("meshes/"):] if mesh_path.startswith("meshes/") else mesh_path
    if not mesh_name:
        raise ValueError(f"RXB mesh reference is missing a mesh name: {ref}")

    path = Path(path_text)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path, mesh_name


def _metadata_to_array(metadata):
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


def _metadata_from_array(array):
    return json.loads(bytes(np.asarray(array, dtype=np.uint8)).decode("utf-8"))


def _validate_metadata(metadata, path):
    if metadata.get("format") != "rxb":
        raise ValueError(f"{path} is not a Roxy Binary file")
    if metadata.get("version") != RXB_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported RXB version {metadata.get('version')} in {path}; "
            f"expected {RXB_FORMAT_VERSION}"
        )
