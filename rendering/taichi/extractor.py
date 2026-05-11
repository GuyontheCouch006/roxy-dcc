import core.timing as timing
from rendering.taichi.fields import (
    _obj_type, _obj_center, _obj_radius,
    _obj_normal, _obj_offset, _obj_extra,
    _obj_mat_type, _obj_albedo,
    _obj_roughness, _obj_ior, _obj_emission,
    _obj_v0, _obj_v1, _obj_v2,
    _n_objects, MAX_OBJECTS,
    _bvh_n_nodes, _bvh_n_tris, _mat_n_mats,
)

_ZERO3 = [0.0, 0.0, 0.0]


def _write_slot(i, data):
    _obj_type[i]      = data['type']
    _obj_center[i]    = data['center']
    _obj_radius[i]    = data['radius']
    _obj_normal[i]    = data['normal']
    _obj_offset[i]    = data['offset']
    _obj_extra[i]     = data['extra']
    _obj_v0[i]        = data.get('v0', _ZERO3)
    _obj_v1[i]        = data.get('v1', _ZERO3)
    _obj_v2[i]        = data.get('v2', _ZERO3)
    _obj_albedo[i]    = data['albedo']
    _obj_mat_type[i]  = data['mat_type']
    _obj_roughness[i] = data['roughness']
    _obj_ior[i]       = data['ior']
    _obj_emission[i]  = data['emission']


@timing.timer("extract scene", tag="taichi")
def extract_scene(world):
    from scene.mesh import IndexedMesh, Mesh
    from scene.primitives import Sphere, Plane, Cube

    primitive_slots = []
    mesh_tri_data   = []

    def _collect(obj):
        if not obj.renderable:
            return

        for shape in obj.shapes:
            geo = shape.geometry

            if isinstance(geo, Mesh):
                M     = obj.world_matrix
                inv_T = obj.world_inverse_transpose_matrix
                face_fallback = None

                for tri in geo._triangles:
                    v0 = list(M.transform_point(tri._v0))
                    v1 = list(M.transform_point(tri._v1))
                    v2 = list(M.transform_point(tri._v2))

                    face_n = tri._normal
                    def _xn(n):
                        return list(inv_T.transform_vector(n).normalize())
                    n0 = _xn(tri._n0 if tri._n0 is not None else face_n)
                    n1 = _xn(tri._n1 if tri._n1 is not None else face_n)
                    n2 = _xn(tri._n2 if tri._n2 is not None else face_n)

                    mat = shape.material_for_group(tri._group)
                    if mat is None:
                        from scene.materials import Diffuse
                        from core import Color
                        mat = Diffuse(Color(0.8, 0.8, 0.8))

                    mt     = mat.taichi_type_id()
                    params = mat.taichi_params()
                    mesh_tri_data.append({
                        'v0': v0, 'v1': v1, 'v2': v2,
                        'n0': n0, 'n1': n1, 'n2': n2,
                        'mat_type':  mt,
                        'albedo':    list(mat._albedo),
                        'roughness': params[0] if mt in (1, 4) else 0.0,
                        'ior':       params[0] if mt == 2 else 1.0,
                        'emission':  params[0] if mt == 3 else 0.0,
                    })
            elif isinstance(geo, IndexedMesh):
                M     = obj.world_matrix
                inv_T = obj.world_inverse_transpose_matrix

                def _xn(n):
                    return list(inv_T.transform_vector(n).normalize())

                for tri_i in range(geo.triangle_count):
                    v0, v1, v2 = geo.triangle_vertices(tri_i)
                    n0, n1, n2 = geo.triangle_normals(tri_i)
                    group = geo.group_for_triangle(tri_i)
                    mat = shape.material_for_group(group)
                    if mat is None:
                        from scene.materials import Diffuse
                        from core import Color
                        mat = Diffuse(Color(0.8, 0.8, 0.8))

                    mt     = mat.taichi_type_id()
                    params = mat.taichi_params()
                    mesh_tri_data.append({
                        'v0': list(M.transform_point(v0)),
                        'v1': list(M.transform_point(v1)),
                        'v2': list(M.transform_point(v2)),
                        'n0': _xn(n0), 'n1': _xn(n1), 'n2': _xn(n2),
                        'mat_type':  mt,
                        'albedo':    list(mat._albedo),
                        'roughness': params[0] if mt in (1, 4) else 0.0,
                        'ior':       params[0] if mt == 2 else 1.0,
                        'emission':  params[0] if mt == 3 else 0.0,
                    })
            else:
                # Primitive (Sphere, Plane, Cube, or legacy Triangle slot)
                exported = obj.taichi_export()
                if isinstance(exported, list):
                    primitive_slots.extend(exported)
                elif exported:
                    primitive_slots.append(exported)
                break   # taichi_export handles all shapes on this object

        for child in obj.children:
            _collect(child)

    for obj in world.objects:
        _collect(obj)

    # ── Upload primitives ─────────────────────────────────────────────────────
    n = len(primitive_slots)
    assert n <= MAX_OBJECTS, f"Scene has {n} primitives but MAX_OBJECTS={MAX_OBJECTS}"
    _n_objects[None] = n
    for i, data in enumerate(primitive_slots):
        _write_slot(i, data)

    # ── Build and upload BVH for mesh triangles ───────────────────────────────
    if mesh_tri_data:
        import sys
        sys.setrecursionlimit(max(sys.getrecursionlimit(), len(mesh_tri_data) * 2))

        from rendering.taichi.bvh_builder import GPUBVHBuilder
        builder = GPUBVHBuilder()
        print(f"  BVH: building from {len(mesh_tri_data):,} triangles …")
        builder.build(mesh_tri_data)
        builder.upload()
        print(f"  BVH: {len(builder.nodes):,} nodes, "
              f"{len(builder.ordered_tris):,} triangles, "
              f"{len(builder.mat_palette)} materials")
    else:
        _bvh_n_nodes[None] = 0
        _bvh_n_tris[None]  = 0
        _mat_n_mats[None]  = 0
