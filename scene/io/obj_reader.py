from pathlib import Path
import time

import numpy as np
import core.timing as timing
from scene.mesh import IndexedMesh, Mesh, Triangle
from scene.materials import Diffuse
from core import Vec3, Vec2, Color


def _print_mesh_stats(root, via):
    if timing.LEVEL < 1:
        return

    def iter_nodes(node):
        for child in node.children:
            yield child
            yield from iter_nodes(child)

    nodes = list(iter_nodes(root))
    tris = sum(
        len(s.geometry._triangles) if hasattr(s.geometry, '_triangles')
        else (s.geometry.triangle_count if hasattr(s.geometry, 'triangle_count') else 0)
        for c in nodes for s in c.shapes
    )
    meshes = sum(1 for c in nodes for s in c.shapes)
    timing.defer_print(f"    via {via} — {meshes} meshes, {tris:,} tris")


class _OBJProgress:
    def __init__(self, file_path, callback, phase="Parsing OBJ"):
        self._callback = callback
        self._phase = phase
        self._total_bytes = max(1, file_path.stat().st_size)
        self._bytes_read = 0
        self._line_count = 0
        self._vertices = 0
        self._normals = 0
        self._uvs = 0
        self._faces = 0
        self._triangles = 0
        self._object_name = ""
        self._group_name = ""
        self._material_name = "default"
        self._last_emit_time = 0.0
        self._last_emit_percent = -1.0
        self._last_assembly_emit_time = 0.0

    def advance(self, line):
        if not self._callback:
            return
        self._bytes_read += len(line.encode("utf-8"))
        self._line_count += 1

    def count_vertex(self):
        self._vertices += 1

    def count_normal(self):
        self._normals += 1

    def count_uv(self):
        self._uvs += 1

    def count_face(self, vertex_count):
        self._faces += 1
        self._triangles += max(0, vertex_count - 2)

    def set_object(self, name):
        self._object_name = name

    def set_group(self, name):
        self._group_name = name

    def set_material(self, name):
        self._material_name = name

    def emit(self, force=False, phase=None, detail=None):
        if not self._callback:
            return

        percent = min(100.0, self._bytes_read / self._total_bytes * 100.0)
        now = time.perf_counter()
        if not force and percent - self._last_emit_percent < 1.0 and now - self._last_emit_time < 0.25:
            return

        self._last_emit_time = now
        self._last_emit_percent = percent
        self._callback(phase or self._phase, detail or self._detail(percent))

    def emit_assembly(self, current, total, triangle_count):
        if not self._callback:
            return
        now = time.perf_counter()
        if current != total and now - self._last_assembly_emit_time < 0.25:
            return
        self._last_assembly_emit_time = now
        self._callback(
            "Assembling OBJ meshes",
            f"{current:,}/{total:,} mesh leaves • {triangle_count:,} triangles in current leaf",
        )

    def finish(self, phase="Loaded OBJ"):
        self.emit(force=True, phase=phase, detail=self._detail(100.0))

    def _detail(self, percent):
        labels = []
        if self._object_name:
            labels.append(f"object: {self._object_name}")
        if self._group_name:
            labels.append(f"group: {self._group_name}")
        if self._material_name:
            labels.append(f"material: {self._material_name}")
        suffix = " • ".join(labels)
        if suffix:
            suffix = " • " + suffix
        return (
            f"{percent:5.1f}% • {self._line_count:,} lines • "
            f"{self._vertices:,} verts • {self._uvs:,} uvs • "
            f"{self._faces:,} faces • {self._triangles:,} tris"
            f"{suffix}"
        )


class OBJReader:
    """OBJ file reader.

    load(path)                  → SceneObject hierarchy
    load_as_mesh(path)          → flat Mesh made of Triangle objects
    load_as_indexed_mesh(path)  → flat IndexedMesh backed by shared arrays
    """

    INDEXED_AUTO_THRESHOLD_BYTES = 32 * 1024 * 1024

    @staticmethod
    @timing.timer(tag="load", label_fn=lambda path, *_, **__: Path(path).name)
    def load(path, name=None, indexed=None, build_bvh=False, progress=None):
        """Load an OBJ file. Uses Assimp if available, falls back to pure Python."""
        file_path = Path(path)
        if indexed is None:
            indexed = (
                file_path.suffix.lower() == '.obj'
                and file_path.stat().st_size >= OBJReader.INDEXED_AUTO_THRESHOLD_BYTES
            )

        if indexed:
            result = OBJReader._load_python_indexed(
                path,
                name,
                build_bvh=build_bvh,
                progress=progress,
            )
            _print_mesh_stats(result, "indexed Python")
            return result

        from scene.io.assimp_importer import AssimpImporter

        if AssimpImporter.is_available():
            try:
                if progress:
                    progress("Loading OBJ with Assimp", str(file_path))
                result = AssimpImporter.load(path, name)
                if result and result.children:
                    if OBJReader._obj_material_names_missing(path, result):
                        print(f"    Assimp lost material groups, falling back to Python parser")
                    else:
                        _print_mesh_stats(result, "Assimp")
                        return result
            except Exception as e:
                print(f"    Assimp failed ({e}), falling back to Python parser")

        result = OBJReader._load_python(path, name, progress=progress)
        _print_mesh_stats(result, "Python")
        return result

    @staticmethod
    def _load_python_indexed(path, name=None, build_bvh=False, progress=None):
        """Load an OBJ hierarchy with IndexedMesh leaves.

        Unlike _load_python(), this keeps vertices/normals/UVs in shared arrays and
        stores each triangle as index rows plus a material-group id. OBJ `o` and `g`
        records become SceneObject hierarchy nodes; material runs become mesh leaves.
        """
        from scene.scene_object import SceneObject
        from scene.shape import Shape as ShapeNode
        from scene.io.mtl_loader import MTLLoader

        file_path = Path(path)
        tracker = _OBJProgress(file_path, progress)
        root = SceneObject(name=name or file_path.stem)
        positions, normals, uvs = [], [], []
        mtl_materials = {}
        group_names, group_to_idx = [], {}

        current_obj_node = root
        current_group_node = root
        current_mtl = 'default'

        pending_pos_idx = []
        pending_normal_idx = []
        pending_uv_idx = []
        pending_group_idx = []
        pending_groups = set()
        leaves = []

        def _idx(raw, length):
            i = int(raw)
            return i - 1 if i > 0 else length + i

        def group_id(name):
            if name not in group_to_idx:
                group_to_idx[name] = len(group_names)
                group_names.append(name)
            return group_to_idx[name]

        current_gid = group_id(current_mtl)

        def flush():
            nonlocal pending_pos_idx, pending_normal_idx, pending_uv_idx
            nonlocal pending_group_idx, pending_groups

            if not pending_pos_idx:
                return

            leaves.append({
                "parent": current_group_node,
                "name": current_mtl,
                "tri_pos_idx": pending_pos_idx,
                "tri_normal_idx": pending_normal_idx,
                "tri_uv_idx": pending_uv_idx,
                "tri_group_idx": pending_group_idx,
                "groups": pending_groups,
            })

            pending_pos_idx = []
            pending_normal_idx = []
            pending_uv_idx = []
            pending_group_idx = []
            pending_groups = set()

        with open(file_path, 'r') as f:
            for line in f:
                tracker.advance(line)
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    tracker.emit()
                    continue
                token = parts[0]

                if token == 'mtllib':
                    mtl_path = file_path.parent / parts[1]
                    tracker.emit(
                        force=True,
                        phase="Loading material library",
                        detail=str(mtl_path),
                    )
                    if mtl_path.exists():
                        raw = MTLLoader.load(str(mtl_path))
                        mtl_materials.update({n: m.to_material() for n, m in raw.items()})
                elif token == 'v':
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    tracker.count_vertex()
                elif token == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    tracker.count_normal()
                elif token == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                    tracker.count_uv()
                elif token == 'o':
                    flush()
                    obj_name = parts[1] if len(parts) > 1 else 'object'
                    tracker.set_object(obj_name)
                    current_obj_node = SceneObject(name=obj_name)
                    root.add_child(current_obj_node)
                    current_group_node = current_obj_node
                elif token == 'g':
                    flush()
                    group_name = parts[1] if len(parts) > 1 else 'group'
                    tracker.set_group(group_name)
                    current_group_node = SceneObject(name=group_name)
                    current_obj_node.add_child(current_group_node)
                elif token == 'usemtl':
                    new_mtl = parts[1] if len(parts) > 1 else 'default'
                    if new_mtl != current_mtl:
                        flush()
                    current_mtl = new_mtl
                    tracker.set_material(current_mtl)
                    current_gid = group_id(current_mtl)
                elif token == 'f':
                    face = parts[1:]
                    tracker.count_face(len(face))
                    vi, ni, uvi = [], [], []
                    for part in face:
                        vals = part.split('/')
                        vi.append(_idx(vals[0], len(positions)))
                        uvi.append(_idx(vals[1], len(uvs)) if len(vals) > 1 and vals[1] else -1)
                        ni.append(_idx(vals[2], len(normals)) if len(vals) > 2 and vals[2] else -1)
                    for i in range(1, len(vi) - 1):
                        pending_pos_idx.append([vi[0], vi[i], vi[i + 1]])
                        pending_uv_idx.append([uvi[0], uvi[i], uvi[i + 1]])
                        pending_normal_idx.append([ni[0], ni[i], ni[i + 1]])
                        pending_group_idx.append(current_gid)
                        pending_groups.add(current_mtl)
                elif token in ('s',):
                    continue
                else:
                    print(f"Warning: Unrecognized OBJ token: {token!r}")
                tracker.emit()

        flush()
        tracker.finish("Parsed OBJ")
        positions_arr = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        normals_arr = (
            np.asarray(normals, dtype=np.float64).reshape((-1, 3))
            if normals else None
        )
        uvs_arr = (
            np.asarray(uvs, dtype=np.float64).reshape((-1, 2))
            if uvs else None
        )

        for i, leaf in enumerate(leaves, 1):
            tracker.emit_assembly(i, len(leaves), len(leaf["tri_pos_idx"]))
            mesh = IndexedMesh(
                positions_arr,
                leaf["tri_pos_idx"],
                normals=normals_arr,
                tri_normal_idx=leaf["tri_normal_idx"],
                uvs=uvs_arr,
                tri_uv_idx=leaf["tri_uv_idx"],
                groups=group_names or ['default'],
                tri_group_idx=leaf["tri_group_idx"],
                name=leaf["name"],
                build_bvh=build_bvh,
            )
            mat_groups = {
                group: mtl_materials.get(group, Diffuse(Color(0.8, 0.8, 0.8)))
                for group in (leaf["groups"] or {'default'})
            }
            shape = ShapeNode(mesh, mat_groups, name=leaf["name"])
            leaf["parent"].add_child(SceneObject(shapes=[shape], name=leaf["name"]))

        tracker.emit(force=True, phase="Finished OBJ hierarchy", detail=f"{len(leaves):,} mesh leaves")
        return root

    @staticmethod
    def load_as_indexed_mesh(path, build_bvh=False, progress=None):
        """Load an OBJ into IndexedMesh storage without material objects."""
        mesh, _ = OBJReader._parse_indexed(path, build_bvh=build_bvh, progress=progress)
        return mesh

    @staticmethod
    def _parse_indexed(path, build_bvh=False, progress=None):
        from scene.io.mtl_loader import MTLLoader

        file_path = Path(path)
        tracker = _OBJProgress(file_path, progress)
        positions, normals, uvs = [], [], []
        tri_pos_idx, tri_normal_idx, tri_uv_idx, tri_group_idx = [], [], [], []
        group_names, group_to_idx = [], {}
        mtl_materials = {}
        used_groups = set()

        def _idx(raw, length):
            i = int(raw)
            return i - 1 if i > 0 else length + i

        def group_id(name):
            if name not in group_to_idx:
                group_to_idx[name] = len(group_names)
                group_names.append(name)
            return group_to_idx[name]

        current_mtl = 'default'
        current_gid = group_id(current_mtl)

        with open(file_path, 'r') as f:
            for line in f:
                tracker.advance(line)
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    tracker.emit()
                    continue
                token = parts[0]

                if token == 'mtllib':
                    mtl_path = file_path.parent / parts[1]
                    tracker.emit(
                        force=True,
                        phase="Loading material library",
                        detail=str(mtl_path),
                    )
                    if mtl_path.exists():
                        raw = MTLLoader.load(str(mtl_path))
                        mtl_materials.update({n: m.to_material() for n, m in raw.items()})
                elif token == 'v':
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    tracker.count_vertex()
                elif token == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    tracker.count_normal()
                elif token == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                    tracker.count_uv()
                elif token == 'usemtl':
                    current_mtl = parts[1] if len(parts) > 1 else 'default'
                    tracker.set_material(current_mtl)
                    current_gid = group_id(current_mtl)
                elif token == 'f':
                    face = parts[1:]
                    tracker.count_face(len(face))
                    vi, ni, uvi = [], [], []
                    for part in face:
                        vals = part.split('/')
                        vi.append(_idx(vals[0], len(positions)))
                        uvi.append(_idx(vals[1], len(uvs)) if len(vals) > 1 and vals[1] else -1)
                        ni.append(_idx(vals[2], len(normals)) if len(vals) > 2 and vals[2] else -1)
                    for i in range(1, len(vi) - 1):
                        tri_pos_idx.append([vi[0], vi[i], vi[i + 1]])
                        tri_uv_idx.append([uvi[0], uvi[i], uvi[i + 1]])
                        tri_normal_idx.append([ni[0], ni[i], ni[i + 1]])
                        tri_group_idx.append(current_gid)
                        used_groups.add(current_mtl)
                elif token in ('o', 'g', 's'):
                    if token == 'o':
                        tracker.set_object(parts[1] if len(parts) > 1 else 'object')
                    elif token == 'g':
                        tracker.set_group(parts[1] if len(parts) > 1 else 'group')
                    continue
                else:
                    print(f"Warning: Unrecognized OBJ token: {token!r}")
                tracker.emit()

        tracker.finish("Parsed OBJ")
        tracker.emit(force=True, phase="Assembling indexed mesh")
        mesh = IndexedMesh(
            positions,
            tri_pos_idx,
            normals=normals or None,
            tri_normal_idx=tri_normal_idx,
            uvs=uvs or None,
            tri_uv_idx=tri_uv_idx,
            groups=group_names or ['default'],
            tri_group_idx=tri_group_idx,
            name=file_path.stem,
            build_bvh=build_bvh,
        )
        mat_groups = {
            group: mtl_materials.get(group, Diffuse(Color(0.8, 0.8, 0.8)))
            for group in (used_groups or {'default'})
        }
        return mesh, mat_groups

    @staticmethod
    def _obj_material_names_missing(path, root):
        file_path = Path(path)
        if file_path.suffix.lower() != '.obj':
            return False

        wanted = set()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts and parts[0] == 'usemtl' and len(parts) > 1:
                        wanted.add(parts[1])
        except OSError:
            return False

        if not wanted:
            return False

        found = set()

        def collect(node):
            found.add(node.name)
            for shape in node.shapes:
                found.update(shape.material_groups.keys())
            for child in node.children:
                collect(child)

        collect(root)
        return not wanted.issubset(found)

    @staticmethod
    def _load_python(path, name=None, progress=None):
        """Load an OBJ and return a SceneObject hierarchy.

        Structure:
            SceneObject(name=filename)      root, no shapes
                SceneObject(name=group)     one per 'g'/'o'/'usemtl' boundary
                    shapes=[Shape(mesh, {mtl_name: Material})]
        """
        from scene.scene_object import SceneObject
        from scene.shape import Shape as ShapeNode
        from scene.io.mtl_loader import MTLLoader

        file_path = Path(path)
        tracker = _OBJProgress(file_path, progress)
        root = SceneObject(name=name or file_path.stem)

        vertices, normals, uvs = [], [], []
        mtl_materials    = {}
        current_obj_node = root
        current_mtl      = 'default'
        pending          = []

        def _idx(raw, length):
            i = int(raw)
            return i - 1 if i > 0 else length + i

        def flush():
            if not pending:
                return
            mesh = Mesh(list(pending))
            # Name node after the material group of these triangles.
            node_name = pending[0].group
            mat_groups = {}
            for tri in pending:
                g = tri.group
                if g not in mat_groups:
                    mat_groups[g] = mtl_materials.get(g, Diffuse(Color(0.8, 0.8, 0.8)))
            shape = ShapeNode(mesh, mat_groups, name=node_name)
            group_obj = SceneObject(shapes=[shape], name=node_name)
            current_obj_node.add_child(group_obj)
            pending.clear()

        with open(file_path, 'r') as f:
            for line in f:
                tracker.advance(line)
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    tracker.emit()
                    continue
                token = parts[0]

                if token == 'mtllib':
                    mtl_path = file_path.parent / parts[1]
                    tracker.emit(
                        force=True,
                        phase="Loading material library",
                        detail=str(mtl_path),
                    )
                    if mtl_path.exists():
                        raw = MTLLoader.load(str(mtl_path))
                        mtl_materials = {n: m.to_material() for n, m in raw.items()}

                elif token == 'v':
                    vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                    tracker.count_vertex()
                elif token == 'vn':
                    normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                    tracker.count_normal()
                elif token == 'vt':
                    uvs.append(Vec2(float(parts[1]), float(parts[2])))
                    tracker.count_uv()

                elif token == 'o':
                    flush()
                    obj_name = parts[1] if len(parts) > 1 else 'object'
                    tracker.set_object(obj_name)
                    current_obj_node = SceneObject(name=obj_name)
                    root.add_child(current_obj_node)

                elif token == 'g':
                    tracker.set_group(parts[1] if len(parts) > 1 else 'group')
                    pass  # Don't flush — 'g' often appears AFTER its faces in OBJ files.

                elif token == 'usemtl':
                    new_mtl = parts[1] if len(parts) > 1 else 'default'
                    if new_mtl != current_mtl and pending:
                        flush()
                    current_mtl = new_mtl
                    tracker.set_material(current_mtl)

                elif token == 'f':
                    face = parts[1:]
                    tracker.count_face(len(face))
                    vi, ni, uvi = [], [], []
                    for part in face:
                        vals = part.split('/')
                        vi.append(_idx(vals[0], len(vertices)))
                        uvi.append(_idx(vals[1], len(uvs))     if len(vals) > 1 and vals[1] else None)
                        ni.append (_idx(vals[2], len(normals))  if len(vals) > 2 and vals[2] else None)
                    for i in range(1, len(vi) - 1):
                        pending.append(Triangle(
                            vertices[vi[0]], vertices[vi[i]], vertices[vi[i+1]],
                            normals[ni[0]]   if ni[0]   is not None else None,
                            normals[ni[i]]   if ni[i]   is not None else None,
                            normals[ni[i+1]] if ni[i+1] is not None else None,
                            uvs[uvi[0]]      if uvi[0]  is not None else None,
                            uvs[uvi[i]]      if uvi[i]  is not None else None,
                            uvs[uvi[i+1]]    if uvi[i+1]is not None else None,
                            group=current_mtl,
                        ))

                elif token in ('s',):
                    continue
                else:
                    print(f"Warning: Unrecognized OBJ token: {token!r}")
                tracker.emit()

        flush()
        tracker.finish("Loaded OBJ")
        return root

    @staticmethod
    def load_as_mesh(path) -> Mesh:
        """Load an OBJ and return a single flat Mesh (all triangles, no materials)."""
        vertices, normals, uvs = [], [], []
        triangles = []

        def _idx(raw, length):
            i = int(raw)
            return i - 1 if i > 0 else length + i

        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    continue
                token = parts[0]
                if token == 'v':
                    vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif token == 'vn':
                    normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif token == 'vt':
                    uvs.append(Vec2(float(parts[1]), float(parts[2])))
                elif token == 'f':
                    face = parts[1:]
                    vi, ni, uvi = [], [], []
                    for part in face:
                        vals = part.split('/')
                        vi.append(_idx(vals[0], len(vertices)))
                        uvi.append(_idx(vals[1], len(uvs))    if len(vals) > 1 and vals[1] else None)
                        ni.append (_idx(vals[2], len(normals)) if len(vals) > 2 and vals[2] else None)
                    for i in range(1, len(vi) - 1):
                        triangles.append(Triangle(
                            vertices[vi[0]], vertices[vi[i]], vertices[vi[i+1]],
                            normals[ni[0]]   if ni[0]   is not None else None,
                            normals[ni[i]]   if ni[i]   is not None else None,
                            normals[ni[i+1]] if ni[i+1] is not None else None,
                            uvs[uvi[0]]      if uvi[0]  is not None else None,
                            uvs[uvi[i]]      if uvi[i]  is not None else None,
                            uvs[uvi[i+1]]    if uvi[i+1]is not None else None,
                        ))
        return Mesh(triangles)
