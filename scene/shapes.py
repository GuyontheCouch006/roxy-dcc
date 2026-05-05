# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Abstract Shape base class and concrete shape implementations.
#              Each shape provides ray intersection, surface normal, and local bounds.
# ============================================

import math
from abc import ABC, abstractmethod

import taichi as ti

from core import Vec3, AABB, HitRecord


class Shape(ABC):
    """Abstract base class for all renderable geometry."""

    @abstractmethod
    def intersect(self, ray): ...

    @abstractmethod
    def normal_at(self, point): ...

    @abstractmethod
    def local_bounds(self): ...

    @property
    def is_infinite(self):
        """Indicates whether this shape is infinite (e.g. Plane) and thus has no bounds."""
        return False


class Sphere(Shape):
    """Unit-sphere (or arbitrary radius) centered at the local origin."""

    def __init__(self, radius=1.0):
        self._radius = radius

    def intersect(self, ray):
        """Return a HitRecord for the nearest valid intersection, or None.

        Uses the simplified h-substitution to avoid the full quadratic:
          h = d · o,  c = |o|² - r²,  disc = h² - c
        where o is the ray origin (sphere assumed at world origin in local space).
        """
        oc = ray._origin

        h = ray._direction.dot(oc)
        c = oc.length_sq() - self._radius ** 2
        discriminant = h ** 2 - c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)

        # Try the nearer root first.
        t = -h - sqrt_disc
        if t > 0.001:
            return HitRecord.from_ray(ray, t, self.normal_at(ray.at(t)))

        # Fall back to the far root (ray origin is inside the sphere).
        t = -h + sqrt_disc
        if t > 0.001:
            return HitRecord.from_ray(ray, t, self.normal_at(ray.at(t)))

        return None

    def normal_at(self, point):
        """Return the outward surface normal at point (assumes point is on the surface)."""
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
    def taichi_intersect(ro, rd, center, radius):  # Taichi renderer
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


class Plane(Shape):
    """Infinite plane defined by a normal and distance from the origin."""

    def __init__(self, normal=Vec3(0, 1, 0), distance=0):
        self._normal = normal.normalize()
        self._distance = distance

    def intersect(self, ray):
        """Return a HitRecord for the intersection with this plane, or None."""
        denom = self._normal.dot(ray.direction)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to the plane.

        t = -(self._normal.dot(ray.origin) + self._distance) / denom
        if t < 0.001:
            return None  # Intersection is behind the ray origin.

        return HitRecord.from_ray(ray, t, self._normal)

    def normal_at(self, point):
        """Return the constant normal of the plane."""
        return self._normal

    def local_bounds(self):
        pass
        """Planes are infinite, so we can return an empty AABB or raise an exception."""
        # raise NotImplementedError("Plane does not have finite bounds.")
    
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


class Cube(Shape):
    """Axis-aligned cube centered at the local origin with given side length."""

    def __init__(self, side_length=1.0):
        self._side_length = side_length
        half = side_length / 2
        self._bounds = AABB(Vec3(-half, -half, -half), Vec3(half, half, half))

    def intersect(self, ray):
        """Return a HitRecord for the intersection with this cube, or None."""
        t = self._bounds.intersect(ray)
        if t is None:
            return None

        hit_point = ray.at(t)
        normal = self.normal_at(hit_point)
        return HitRecord.from_ray(ray, t, normal)

    def normal_at(self, point):
        """Return the normal of the cube face that point is on."""
        # Determine which face is closest to the point and return its normal.
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


def create_shape_from_dict(data):
    shape_type = data["type"]
    if shape_type == "sphere":
        return Sphere(data["radius"])
    elif shape_type == "plane":
        return Plane(Vec3.from_dict(data["normal"]), data["distance"])
    elif shape_type == "cube":
        return Cube(data["side_length"])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")