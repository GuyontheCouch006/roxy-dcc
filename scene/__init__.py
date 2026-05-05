# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Public API for the scene package.
# ============================================

from scene.shapes import Sphere, Plane, Cube
from scene.materials import Material, Diffuse, Metal, Dielectric
from scene.scene_object import SceneObject
from scene.world import World
from scene.camera import Camera
from scene.io import load_scene, save_scene
