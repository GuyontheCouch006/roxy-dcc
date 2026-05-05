# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Axis-Aligned Bounding Box (AABB) with slab-method ray intersection,
#              union, centroid, and matrix transform support.
# ============================================

from core.vectors import Vec3


class AABB:
    """Axis-aligned bounding box defined by min and max corner points."""

    def __init__(self, min, max):
        if isinstance(min, Vec3) and isinstance(max, Vec3):
            self.min = min
            self.max = max
        elif isinstance(min, list) and isinstance(max, list):
            self.min = Vec3(min[0], min[1], min[2])
            self.max = Vec3(max[0], max[1], max[2])
        else:
            raise ValueError("min and max must be Vec3 or list of 3 floats")

    @classmethod
    def from_points(cls, points):
        """Construct the tightest AABB that contains all given points."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        return cls(
            Vec3(min(xs), min(ys), min(zs)),
            Vec3(max(xs), max(ys), max(zs)),
        )

    def union(self, other):
        """Return the smallest AABB that contains both this box and other."""
        new_min = Vec3(
            min(self.min.x, other.min.x),
            min(self.min.y, other.min.y),
            min(self.min.z, other.min.z),
        )
        new_max = Vec3(
            max(self.max.x, other.max.x),
            max(self.max.y, other.max.y),
            max(self.max.z, other.max.z),
        )
        return AABB(new_min, new_max)

    def intersect(self, ray):
        """Test ray against this box using the slab method. Returns t_enter or None."""
        ox, oy, oz = ray._origin
        dx, dy, dz = ray._direction
        min_x, min_y, min_z = self.min
        max_x, max_y, max_z = self.max

        t_min_x = (min_x - ox) / dx if dx > 1e-10 or dx < -1e-10 else (float('inf') if min_x >= ox else float('-inf'))
        t_max_x = (max_x - ox) / dx if dx > 1e-10 or dx < -1e-10 else (float('inf') if max_x >= ox else float('-inf'))
        t_min_y = (min_y - oy) / dy if dy > 1e-10 or dy < -1e-10 else (float('inf') if min_y >= oy else float('-inf'))
        t_max_y = (max_y - oy) / dy if dy > 1e-10 or dy < -1e-10 else (float('inf') if max_y >= oy else float('-inf'))
        t_min_z = (min_z - oz) / dz if dz > 1e-10 or dz < -1e-10 else (float('inf') if min_z >= oz else float('-inf'))
        t_max_z = (max_z - oz) / dz if dz > 1e-10 or dz < -1e-10 else (float('inf') if max_z >= oz else float('-inf'))

        if dx < 0: t_min_x, t_max_x = t_max_x, t_min_x
        if dy < 0: t_min_y, t_max_y = t_max_y, t_min_y
        if dz < 0: t_min_z, t_max_z = t_max_z, t_min_z

        t_enter = max(t_min_x, t_min_y, t_min_z)
        t_exit  = min(t_max_x, t_max_y, t_max_z)

        if t_enter > t_exit or t_exit < 0:
            return None
        return t_enter
        
    def centroid(self):
        return Vec3(
            (self.min.x + self.max.x) / 2,
            (self.min.y + self.max.y) / 2,
            (self.min.z + self.max.z) / 2,
        )

    def transform(self, matrix):
        """Return a new AABB that bounds all 8 transformed corners of this box."""
        corners = [
            Vec3(self.min.x, self.min.y, self.min.z),
            Vec3(self.min.x, self.min.y, self.max.z),
            Vec3(self.min.x, self.max.y, self.min.z),
            Vec3(self.min.x, self.max.y, self.max.z),
            Vec3(self.max.x, self.min.y, self.min.z),
            Vec3(self.max.x, self.min.y, self.max.z),
            Vec3(self.max.x, self.max.y, self.min.z),
            Vec3(self.max.x, self.max.y, self.max.z),
        ]
        transformed = [matrix.transform_point(c) for c in corners]
        return AABB.from_points(transformed)

    def __repr__(self):
        return f"AABB(min={self.min}, max={self.max})"
