import taichi as ti
import os

_ARCH = os.environ.get("RAY_TRACER_TAICHI_ARCH", "gpu").lower()
ti.init(arch=ti.cpu if _ARCH == "cpu" else ti.gpu)

from rendering.taichi.fields import _pixels, _n_objects, MAX_W, MAX_H, MAX_OBJECTS  # noqa: E402
from rendering.taichi.sky import sky_color  # noqa: E402
from rendering.taichi.camera import get_ray_direction  # noqa: E402
from rendering.taichi.scatter import random_unit_vector, diffuse_scatter, metal_scatter, dielectric_scatter  # noqa: E402
from rendering.taichi.intersect import scene_intersect  # noqa: E402
from rendering.taichi.trace import trace, render_kernel  # noqa: E402
from rendering.taichi.extractor import extract_scene  # noqa: E402
from rendering.taichi.bvh_builder import GPUBVHBuilder  # noqa: E402
