import core.timing as timing
from rendering.taichi.fields import (
    _obj_type, _obj_center, _obj_radius,
    _obj_normal, _obj_offset, _obj_extra,
    _obj_mat_type, _obj_albedo,
    _obj_roughness, _obj_ior, _obj_emission,
    _obj_v0, _obj_v1, _obj_v2,
    _n_objects, MAX_OBJECTS,
    _light_center, _light_radius, _light_albedo, _light_emission,
    _n_lights, MAX_LIGHTS,
    _bvh_n_nodes, _bvh_n_tris, _mat_n_mats,
    _tex_pixels, _tex_width, _tex_height, _tex_flip_v, _tex_n,
    MAX_TEXTURES, MAX_TEXTURE_SIZE,
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


def _upload_lights(primitive_slots):
    lights = [
        data for data in primitive_slots
        if data['type'] == 0 and data['mat_type'] == 3 and data['emission'] > 0.0
    ]
    assert len(lights) <= MAX_LIGHTS, (
        f"Scene has {len(lights)} emissive sphere lights but MAX_LIGHTS={MAX_LIGHTS}"
    )

    _n_lights[None] = len(lights)
    for i, data in enumerate(lights):
        _light_center[i]   = data['center']
        _light_radius[i]   = data['radius']
        _light_albedo[i]   = data['albedo']
        _light_emission[i] = data['emission']


def _upload_textures(textures):
    _tex_n[None] = len(textures)
    for i in range(MAX_TEXTURES):
        _tex_width[i] = 0
        _tex_height[i] = 0
        _tex_flip_v[i] = 0

    if not textures:
        return

    import numpy as np

    pixels = np.zeros(
        (MAX_TEXTURES, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE, 3),
        dtype=np.uint8,
    )
    for i, texture in enumerate(textures):
        data = texture.load_pixels_u8(max_size=MAX_TEXTURE_SIZE)
        h, w = data.shape[:2]
        pixels[i, :h, :w] = data
        _tex_width[i] = w
        _tex_height[i] = h
        _tex_flip_v[i] = 1 if texture.flip_v else 0

        if timing.LEVEL >= 1:
            label = texture.path or f"texture_{i}"
            timing.defer_print(f"    texture[{i}] {label} → {w}×{h}")

    _tex_pixels.from_numpy(pixels)


@timing.timer("extract scene", tag="taichi")
def extract_scene(world):
    from scene.mesh import IndexedMesh, Mesh, indexed_triangle_arrays
    from scene.primitives import Sphere, Plane, Cube
    from rendering.taichi.bvh_builder import GPUBVHBuilder, TriangleBatch

    primitive_slots = []
    mesh_tri_data   = []
    mesh_batches    = []
    textures         = []
    texture_indices  = {}

    def _texture_idx(mat):
        texture = getattr(mat, '_albedo_texture', None)
        if texture is None:
            return -1

        key = (
            texture.path,
            bool(texture.flip_v),
            id(texture) if texture.path is None else None,
        )
        if key in texture_indices:
            return texture_indices[key]

        if len(textures) >= MAX_TEXTURES:
            raise ValueError(
                f"Scene has more than {MAX_TEXTURES} image textures; "
                "increase MAX_TEXTURES in rendering/taichi/fields.py"
            )

        texture_indices[key] = len(textures)
        textures.append(texture)
        return texture_indices[key]

    def _material_data(mat):
        mt = mat.taichi_type_id()
        params = mat.taichi_params()
        return {
            'mat_type': mt,
            'albedo': list(mat._albedo),
            'roughness': params[0] if mt in (1, 4) else 0.0,
            'ior': params[0] if mt == 2 else 1.0,
            'emission': params[0] if mt == 3 else 0.0,
            'texture_idx': _texture_idx(mat),
        }

    def _uv2(uv):
        return [float(uv[0]), float(uv[1])] if uv is not None else [0.0, 0.0]

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

                    mat_data = _material_data(mat)
                    mesh_tri_data.append({
                        'v0': v0, 'v1': v1, 'v2': v2,
                        'n0': n0, 'n1': n1, 'n2': n2,
                        'uv0': _uv2(tri._uv0),
                        'uv1': _uv2(tri._uv1),
                        'uv2': _uv2(tri._uv2),
                        'has_uv': 1 if tri._uv0 is not None and tri._uv1 is not None and tri._uv2 is not None else 0,
                        **mat_data,
                    })
            elif isinstance(geo, IndexedMesh):
                arrays = indexed_triangle_arrays(
                    geo,
                    matrix=obj.world_matrix,
                    normal_matrix=obj.world_inverse_transpose_matrix,
                )
                mat_by_group_idx = {}

                for group_idx, group in enumerate(arrays['groups']):
                    mat = shape.material_for_group(group)
                    if mat is None:
                        from scene.materials import Diffuse
                        from core import Color
                        mat = Diffuse(Color(0.8, 0.8, 0.8))

                    mat_by_group_idx[group_idx] = _material_data(mat)

                mesh_batches.append(TriangleBatch.from_indexed_arrays(arrays, mat_by_group_idx))
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

    _upload_textures(textures)

    # ── Upload primitives ─────────────────────────────────────────────────────
    n = len(primitive_slots)
    assert n <= MAX_OBJECTS, f"Scene has {n} primitives but MAX_OBJECTS={MAX_OBJECTS}"
    _n_objects[None] = n
    for i, data in enumerate(primitive_slots):
        _write_slot(i, data)
    _upload_lights(primitive_slots)

    # ── Build and upload BVH for mesh triangles ───────────────────────────────
    mesh_tri_count = len(mesh_tri_data) + sum(batch.triangle_count for batch in mesh_batches)
    if mesh_tri_count:
        import sys
        sys.setrecursionlimit(max(sys.getrecursionlimit(), mesh_tri_count * 2))

        if mesh_tri_data:
            mesh_batches.append(TriangleBatch.from_dicts(mesh_tri_data))
        builder = GPUBVHBuilder()
        print(f"  BVH: building from {mesh_tri_count:,} triangles …")
        builder.build(mesh_batches)
        builder.upload()
        print(f"  BVH: {len(builder.nodes):,} nodes, "
              f"{builder.triangle_count:,} triangles, "
              f"{len(builder.mat_palette)} materials")
        return {
            'primitive_count': n,
            'bvh_nodes': len(builder.nodes),
            'bvh_triangles': builder.triangle_count,
            'bvh_materials': len(builder.mat_palette),
            'bvh_leaf_size': builder.MAX_LEAF_SIZE,
        }
    else:
        _bvh_n_nodes[None] = 0
        _bvh_n_tris[None]  = 0
        _mat_n_mats[None]  = 0
        return {
            'primitive_count': n,
            'bvh_nodes': 0,
            'bvh_triangles': 0,
            'bvh_materials': 0,
            'bvh_leaf_size': 0,
        }
