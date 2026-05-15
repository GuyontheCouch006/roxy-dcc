# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Public API for the scene package.
# ============================================

from scene.primitives import Sphere, Plane, Cube, Torus, Primitive, create_primitive_from_dict
from scene.shape import Shape
from scene.mesh import IndexedMesh, Mesh, Triangle
from scene.scene_object import SceneObject
from scene.world import World
from scene.camera import Camera
from scene.materials import (Material, Diffuse, Metal, Dielectric,
                              Emissive, Glossy, create_material_from_dict)
from scene.textures import ImageTexture, create_texture_from_dict
from scene.io import load_scene, save_scene
