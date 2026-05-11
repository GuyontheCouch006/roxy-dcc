# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Universal mesh importer using the Assimp C library via pyassimp.
#              Handles OBJ, FBX, glTF, DAE, PLY, STL and 50+ other formats.
#              Falls back gracefully if pyassimp is not installed.
# ============================================

from pathlib import Path

from core import Vec3, Color
from core.vectors import Vec2
from scene.mesh import Mesh, Triangle
from scene.materials import Diffuse, Metal, Dielectric, Emissive, Glossy
from scene.scene_object import SceneObject


class AssimpImporter:

    SUPPORTED_EXTENSIONS = {
        '.obj', '.fbx', '.gltf', '.glb', '.dae', '.collada',
        '.ply', '.stl', '.3ds', '.blend', '.x3d', '.lwo',
    }

    @staticmethod
    def is_available() -> bool:
        try:
            import pyassimp  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    @staticmethod
    def load(path, name=None) -> SceneObject:
        import pyassimp
        import pyassimp.postprocess as pp

        flags = (
            pp.aiProcess_Triangulate |
            pp.aiProcess_GenSmoothNormals |
            pp.aiProcess_JoinIdenticalVertices |
            pp.aiProcess_FlipUVs
        )

        with pyassimp.load(str(path), processing=flags) as scene:
            root_name = name or Path(path).stem
            root = SceneObject(name=root_name)

            for ai_mesh in scene.meshes:
                child = AssimpImporter._convert_mesh(ai_mesh, scene, path)
                if child:
                    root.add_child(child)

            return root

    @staticmethod
    def _convert_mesh(ai_mesh, scene, base_path) -> SceneObject:
        verts   = ai_mesh.vertices
        normals = ai_mesh.normals
        has_uvs = (len(ai_mesh.texturecoords) > 0 and
                   ai_mesh.texturecoords[0] is not None)
        uvs = ai_mesh.texturecoords[0] if has_uvs else None

        triangles = []
        for face in ai_mesh.faces:
            if len(face) != 3:
                continue

            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

            v0 = Vec3(float(verts[i0][0]), float(verts[i0][1]), float(verts[i0][2]))
            v1 = Vec3(float(verts[i1][0]), float(verts[i1][1]), float(verts[i1][2]))
            v2 = Vec3(float(verts[i2][0]), float(verts[i2][1]), float(verts[i2][2]))

            n0 = n1 = n2 = None
            if normals is not None and len(normals) > 0:
                n0 = Vec3(float(normals[i0][0]), float(normals[i0][1]), float(normals[i0][2]))
                n1 = Vec3(float(normals[i1][0]), float(normals[i1][1]), float(normals[i1][2]))
                n2 = Vec3(float(normals[i2][0]), float(normals[i2][1]), float(normals[i2][2]))

            uv0 = uv1 = uv2 = None
            if uvs is not None:
                uv0 = Vec2(float(uvs[i0][0]), float(uvs[i0][1]))
                uv1 = Vec2(float(uvs[i1][0]), float(uvs[i1][1]))
                uv2 = Vec2(float(uvs[i2][0]), float(uvs[i2][1]))

            group = ai_mesh.name or 'default'
            triangles.append(Triangle(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, group=group))

        if not triangles:
            return None

        mesh     = Mesh(triangles=triangles)
        material = AssimpImporter._convert_material(
            scene.materials[ai_mesh.materialindex], base_path)

        from scene.shape import Shape as ShapeNode
        group_name = ai_mesh.name or 'default'
        shape = ShapeNode(mesh, {group_name: material}, name=ai_mesh.name)

        return SceneObject(shapes=[shape], name=ai_mesh.name or 'mesh')

    @staticmethod
    def _convert_material(ai_mat, base_path):
        props = ai_mat.properties

        def get_color(key, default):
            val = props.get((key, 0))
            if val is not None and hasattr(val, '__len__') and len(val) >= 3:
                return Color(float(val[0]), float(val[1]), float(val[2]))
            return default

        def get_float(key, default):
            val = props.get((key, 0))
            return float(val) if val is not None else default

        kd = get_color('diffuse',  Color(0.8, 0.8, 0.8))
        ks = get_color('specular', Color(0.0, 0.0, 0.0))
        ke = get_color('emissive', Color(0.0, 0.0, 0.0))
        ns = get_float('shininess', 10.0)
        d  = get_float('opacity',   1.0)
        ni = get_float('refracti',  1.0)

        if ke.r > 0 or ke.g > 0 or ke.b > 0:
            return Emissive(ke, intensity=1.0)
        if d < 0.99:
            return Dielectric(Color(1.0, 1.0, 1.0), ior=ni)
        if ks.r > 0 or ks.g > 0 or ks.b > 0:
            roughness = max(0.0, 1.0 - ns / 1000.0)
            return Glossy(kd, roughness=roughness)
        return Diffuse(kd)
