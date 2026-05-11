# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Abstract Primitive base class and concrete geometry implementations.
#              Each primitive provides ray intersection, surface normal, and local bounds.
#              intersect() returns a PrimitiveHit (raw geometry result, no material).
# ============================================

import math
from abc import ABC, abstractmethod
from collections import namedtuple

try:
    import taichi as ti
    _TAICHI_AVAILABLE = True
except ImportError:
    _TAICHI_AVAILABLE = False
    class ti:
        @staticmethod
        def func(f): return f

from core import Vec3, AABB
from scene.mesh import IndexedMesh, Mesh, Triangle

# Raw geometry intersection result — no material, no front_face correction.
PrimitiveHit = namedtuple('PrimitiveHit', ['t', 'normal', 'uv'])


class Primitive(ABC):
    """Abstract base class for all raw geometry primitives."""

    @abstractmethod
    def intersect(self, ray): ...

    @abstractmethod
    def normal_at(self, point): ...

    @abstractmethod
    def local_bounds(self): ...

    @property
    def is_infinite(self):
        return False


class Sphere(Primitive):
    """Unit-sphere (or arbitrary radius) centered at the local origin."""

    def __init__(self, radius=1.0):
        self._radius = radius

    def intersect(self, ray):
        oc = ray._origin
        h = ray._direction.dot(oc)
        c = oc.length_sq() - self._radius ** 2
        discriminant = h ** 2 - c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)

        t = -h - sqrt_disc
        if t > 0.001:
            return PrimitiveHit(t=t, normal=self.normal_at(ray.at(t)), uv=None)

        t = -h + sqrt_disc
        if t > 0.001:
            return PrimitiveHit(t=t, normal=self.normal_at(ray.at(t)), uv=None)

        return None

    def normal_at(self, point):
        return point.normalize()

    def local_bounds(self):
        r = self._radius
        return AABB(Vec3(-r, -r, -r), Vec3(r, r, r))

    def __repr__(self):
        return f"Sphere(radius={self._radius})"

    def to_dict(self):
        return {"type": "sphere", "radius": self._radius}

    def taichi_type_id(self): return 0
    def taichi_data(self): return [self._radius]

    @staticmethod
    @ti.func
    def taichi_intersect(ro, rd, center, radius):
        oc = ro - center
        h = rd.dot(oc)
        c = oc.dot(oc) - radius * radius
        disc = h * h - c
        t = -1.0
        if disc >= 0.0:
            sqrt_disc = ti.sqrt(disc)
            t = -h - sqrt_disc
            if t < 0.001:
                t = -h + sqrt_disc
            if t < 0.001:
                t = -1.0
        return t


class Plane(Primitive):
    """Infinite plane defined by a normal and distance from the origin."""

    def __init__(self, normal=Vec3(0, 1, 0), distance=0):
        self._normal = normal.normalize()
        self._distance = distance

    def intersect(self, ray):
        denom = self._normal.dot(ray.direction)
        if abs(denom) < 1e-6:
            return None

        t = -(self._normal.dot(ray.origin) + self._distance) / denom
        if t < 0.001:
            return None

        return PrimitiveHit(t=t, normal=self._normal, uv=None)

    def normal_at(self, point):
        return self._normal

    def local_bounds(self):
        return None

    @property
    def is_infinite(self):
        return True

    def __repr__(self):
        return f"Plane(normal={self._normal}, distance={self._distance})"

    def to_dict(self):
        return {"type": "plane", "normal": self._normal.to_dict(), "distance": self._distance}

    def taichi_type_id(self): return 1
    def taichi_data(self): return list(self._normal) + [self._distance]

    @staticmethod
    @ti.func
    def taichi_intersect(ro, rd, normal, offset):
        denom = normal.dot(rd)
        t = -1.0
        if ti.abs(denom) >= 1e-6:
            t_cand = -(normal.dot(ro) + offset) / denom
            if t_cand >= 0.001:
                t = t_cand
        return t


class Cube(Primitive):
    """Axis-aligned cube centered at the local origin with given side length."""

    def __init__(self, side_length=1.0):
        self._side_length = side_length
        half = side_length / 2
        self._bounds = AABB(Vec3(-half, -half, -half), Vec3(half, half, half))

    def intersect(self, ray):
        t = self._bounds.intersect(ray)
        if t is None:
            return None
        normal = self.normal_at(ray.at(t))
        return PrimitiveHit(t=t, normal=normal, uv=None)

    def normal_at(self, point):
        abs_point = Vec3(abs(point.x), abs(point.y), abs(point.z))
        max_coord = max(abs_point.x, abs_point.y, abs_point.z)
        if max_coord == abs_point.x:
            return Vec3(1 if point.x > 0 else -1, 0, 0)
        elif max_coord == abs_point.y:
            return Vec3(0, 1 if point.y > 0 else -1, 0)
        else:
            return Vec3(0, 0, 1 if point.z > 0 else -1)

    def local_bounds(self):
        return self._bounds

    def taichi_type_id(self): return 2

    @staticmethod
    @ti.func
    def taichi_intersect(ro, rd, center, half_extents):
        aabb_min = center - half_extents
        aabb_max = center + half_extents
        t_min = (aabb_min - ro) / rd
        t_max = (aabb_max - ro) / rd
        t1 = ti.min(t_min, t_max)
        t2 = ti.max(t_min, t_max)
        t_enter = ti.max(t1[0], ti.max(t1[1], t1[2]))
        t_exit  = ti.min(t2[0], ti.min(t2[1], t2[2]))
        t = -1.0
        if t_exit >= t_enter and t_exit >= 0.001:
            t = t_enter if t_enter >= 0.001 else t_exit
        return t

    @staticmethod
    @ti.func
    def taichi_normal(hit_p, center, half_extents):
        d = (hit_p - center) / half_extents
        abs_d = ti.abs(d)
        normal = ti.Vector([0.0, 1.0, 0.0])
        if abs_d[0] >= abs_d[1] and abs_d[0] >= abs_d[2]:
            normal = ti.Vector([1.0 if d[0] > 0.0 else -1.0, 0.0, 0.0])
        elif abs_d[1] >= abs_d[2]:
            normal = ti.Vector([0.0, 1.0 if d[1] > 0.0 else -1.0, 0.0])
        else:
            normal = ti.Vector([0.0, 0.0, 1.0 if d[2] > 0.0 else -1.0])
        return normal

    def __repr__(self):
        return f"Cube(side_length={self._side_length})"

    def to_dict(self):
        return {"type": "cube", "side_length": self._side_length}


class Torus(Primitive):
    """Torus centered at the local origin with given major and minor radii."""

    def __init__(self, major_radius=1.0, minor_radius=0.25):
        self._major_radius = major_radius
        self._minor_radius = minor_radius

    def intersect(self, ray):
        raise NotImplementedError("Torus intersection not implemented.")

    def normal_at(self, point):
        raise NotImplementedError("Torus normal not implemented.")

    def local_bounds(self):
        r = self._major_radius + self._minor_radius
        return AABB(Vec3(-r, -self._minor_radius, -r), Vec3(r, self._minor_radius, r))

    def __repr__(self):
        return f"Torus(major_radius={self._major_radius}, minor_radius={self._minor_radius})"

    def to_dict(self):
        return {"type": "torus", "major_radius": self._major_radius, "minor_radius": self._minor_radius}

    def taichi_type_id(self): return 3
    def taichi_data(self): return [self._major_radius, self._minor_radius]


def create_primitive_from_dict(data):
    if data is None:
        return None

    shape_type = data["type"]
    if shape_type == "sphere":
        return Sphere(data["radius"])
    elif shape_type == "plane":
        return Plane(Vec3.from_dict(data["normal"]), data["distance"])
    elif shape_type == "cube":
        return Cube(data["side_length"])
    elif shape_type == "triangle":
        return Triangle.from_dict(data)
    elif shape_type == "torus":
        return Torus(data["major_radius"], data["minor_radius"])
    elif shape_type == "mesh":
        return Mesh.from_dict(data)
    elif shape_type == "indexed_mesh":
        return IndexedMesh.from_dict(data)
    else:
        raise ValueError(f"Unknown primitive type: {shape_type}")


# Backward compat alias
create_shape_from_dict = create_primitive_from_dict
