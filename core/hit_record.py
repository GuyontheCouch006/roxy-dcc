# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: HitRecord stores the result of a ray-surface intersection —
#              hit distance, world-space point, shading normal, material, and face side.
# ============================================


class HitRecord:
    """Result of a ray-surface intersection.

    The normal is always oriented against the incoming ray direction:
    outward for front-face hits, flipped for back-face (interior) hits.
    """

    def __init__(self, t, point, normal, material, front_face):
        self._t = t
        self._point = point
        self._normal = normal
        self._material = material
        self._front_face = front_face

    @classmethod
    def from_ray(cls, ray, t, outward_normal, material=None):
        """Construct a HitRecord, orienting the normal against the ray."""
        point = ray.at(t)
        front_face = ray.direction.dot(outward_normal) < 0
        normal = outward_normal if front_face else -outward_normal
        return cls(t, point, normal, material, front_face)

    @property
    def t(self): return self._t

    @property
    def point(self): return self._point

    @property
    def normal(self): return self._normal

    @property
    def material(self): return self._material

    @property
    def front_face(self): return self._front_face

    # Ordering is by hit distance, enabling min() on a list of hits.
    def __eq__(self, other): return self._t == other._t
    def __lt__(self, other): return self._t < other._t
    def __le__(self, other): return self._t <= other._t
    def __gt__(self, other): return self._t > other._t
    def __ge__(self, other): return self._t >= other._t
    def __ne__(self, other): return self._t != other._t

    def __bool__(self):
        """True when the hit is in front of the ray origin (t > 0)."""
        return self._t > 0

    def __repr__(self):
        return (f"HitRecord(t={self._t}, point={self._point}, "
                f"normal={self._normal}, material={self._material}, "
                f"front_face={self._front_face})")
    
    def __iter__(self):
        yield self._t
        yield self._point
        yield self._normal
        yield self._material
        yield self._front_face
