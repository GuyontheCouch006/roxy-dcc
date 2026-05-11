from pathlib import Path

import core.timing as timing
from scene.mesh import IndexedMesh, Mesh, Triangle
from scene.materials import Diffuse
from core import Vec3, Vec2, Color


def _print_mesh_stats(root, via):
    if timing.LEVEL < 1:
        return
    children = root.children
    tris = sum(
        len(s.geometry._triangles) if hasattr(s.geometry, '_triangles')
        else (s.geometry.triangle_count if hasattr(s.geometry, 'triangle_count') else 0)
        for c in children for s in c.shapes
    )
    timing.defer_print(f"    via {via} — {len(children)} meshes, {tris:,} tris")


class OBJReader:
    """OBJ file reader.

    load(path)                  → SceneObject hierarchy
    load_as_mesh(path)          → flat Mesh made of Triangle objects
    load_as_indexed_mesh(path)  → flat IndexedMesh backed by shared arrays
    """

    INDEXED_AUTO_THRESHOLD_BYTES = 32 * 1024 * 1024

    @staticmethod
    @timing.timer(tag="load", label_fn=lambda path, *_: Path(path).name)
    def load(path, name=None, indexed=None, build_bvh=False):
        """Load an OBJ file. Uses Assimp if available, falls back to pure Python."""
        file_path = Path(path)
        if indexed is None:
            indexed = (
                file_path.suffix.lower() == '.obj'
                and file_path.stat().st_size >= OBJReader.INDEXED_AUTO_THRESHOLD_BYTES
            )

        if indexed:
            result = OBJReader._load_python_indexed(path, name, build_bvh=build_bvh)
            _print_mesh_stats(result, "indexed Python")
            return result

        from scene.io.assimp_importer import AssimpImporter

        if AssimpImporter.is_available():
            try:
                result = AssimpImporter.load(path, name)
                if result and result.children:
                    if OBJReader._obj_material_names_missing(path, result):
                        print(f"    Assimp lost material groups, falling back to Python parser")
                    else:
                        _print_mesh_stats(result, "Assimp")
                        return result
            except Exception as e:
                print(f"    Assimp failed ({e}), falling back to Python parser")

        result = OBJReader._load_python(path, name)
        _print_mesh_stats(result, "Python")
        return result

    @staticmethod
    def _load_python_indexed(path, name=None, build_bvh=False):
        """Load an OBJ hierarchy with one IndexedMesh child.

        Unlike _load_python(), this keeps vertices/normals/UVs in shared arrays and
        stores each triangle as index rows plus a material-group id.
        """
        from scene.scene_object import SceneObject
        from scene.shape import Shape as ShapeNode

        file_path = Path(path)
        root = SceneObject(name=name or file_path.stem)
        mesh, mat_groups = OBJReader._parse_indexed(path, build_bvh=build_bvh)
        shape = ShapeNode(mesh, mat_groups, name=file_path.stem)
        root.add_child(SceneObject(shapes=[shape], name=file_path.stem))
        return root

    @staticmethod
    def load_as_indexed_mesh(path, build_bvh=False):
        """Load an OBJ into IndexedMesh storage without material objects."""
        mesh, _ = OBJReader._parse_indexed(path, build_bvh=build_bvh)
        return mesh

    @staticmethod
    def _parse_indexed(path, build_bvh=False):
        from scene.io.mtl_loader import MTLLoader

        file_path = Path(path)
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
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    continue
                token = parts[0]

                if token == 'mtllib':
                    mtl_path = file_path.parent / parts[1]
                    if mtl_path.exists():
                        raw = MTLLoader.load(str(mtl_path))
                        mtl_materials.update({n: m.to_material() for n, m in raw.items()})
                elif token == 'v':
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif token == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif token == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                elif token == 'usemtl':
                    current_mtl = parts[1] if len(parts) > 1 else 'default'
                    current_gid = group_id(current_mtl)
                elif token == 'f':
                    face = parts[1:]
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
                    continue
                else:
                    print(f"Warning: Unrecognized OBJ token: {token!r}")

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
    def _load_python(path, name=None):
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
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    continue
                token = parts[0]

                if token == 'mtllib':
                    mtl_path = file_path.parent / parts[1]
                    if mtl_path.exists():
                        raw = MTLLoader.load(str(mtl_path))
                        mtl_materials = {n: m.to_material() for n, m in raw.items()}

                elif token == 'v':
                    vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif token == 'vn':
                    normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif token == 'vt':
                    uvs.append(Vec2(float(parts[1]), float(parts[2])))

                elif token == 'o':
                    flush()
                    obj_name = parts[1] if len(parts) > 1 else 'object'
                    current_obj_node = SceneObject(name=obj_name)
                    root.add_child(current_obj_node)

                elif token == 'g':
                    pass  # Don't flush — 'g' often appears AFTER its faces in OBJ files.

                elif token == 'usemtl':
                    new_mtl = parts[1] if len(parts) > 1 else 'default'
                    if new_mtl != current_mtl and pending:
                        flush()
                    current_mtl = new_mtl

                elif token == 'f':
                    face = parts[1:]
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

        flush()
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
