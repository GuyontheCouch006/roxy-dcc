# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Core vector math types — Vec2, Vec3, Color, Ray, and Point aliases.
#              All types are immutable and use __slots__ for performance.
# ============================================

import math


class Vec2:
    """2D vector with standard arithmetic and linear-algebra operations."""

    __slots__ = ('_x', '_y')

    def __init__(self, x=0.0, y=0.0):
        self._x, self._y = x, y

    def __add__(self, other):
        return self.__class__(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        return self.__class__(self._x - other._x, self._y - other._y)

    def __mul__(self, scalar):
        return self.__class__(self._x * scalar, self._y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    def dot(self, other):
        return self._x * other._x + self._y * other._y

    def length_sq(self): return self.dot(self)
    def length(self): return math.sqrt(self.length_sq())

    def normalize(self):
        mag = self.length()
        return self / mag if mag > 0 else self.__class__(0.0, 0.0)

    def __neg__(self): return self.__class__(-self._x, -self._y)
    def __repr__(self): return f"Vec2({self._x}, {self._y})"
    def __hash__(self): return hash((self._x, self._y))

    def __eq__(self, other):
        return (isinstance(other, Vec2)
                and self._x == other._x
                and self._y == other._y)

    def __iter__(self):
        yield self._x
        yield self._y

    def __getitem__(self, index):
        if index == 0: return self._x
        if index == 1: return self._y
        raise IndexError(f"Vec2 index {index} out of range")

    def to_dict(self):
        return [self._x, self._y]

    @classmethod
    def from_dict(cls, data):
        return cls(data[0], data[1])

class Vec3:
    """3D vector with standard arithmetic, dot/cross products, and normalization."""

    __slots__ = ('_x', '_y', '_z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self._x, self._y, self._z = x, y, z

    def __add__(self, other):
        return self.__class__(
            self._x + other._x,
            self._y + other._y,
            self._z + other._z,
        )

    def __sub__(self, other):
        return self.__class__(
            self._x - other._x,
            self._y - other._y,
            self._z - other._z,
        )

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return self.__class__(
                self._x * other._x,
                self._y * other._y,
                self._z * other._z,
            )
        return self.__class__(
            self._x * other,
            self._y * other,
            self._z * other,
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    @property
    def z(self): return self._z

    def dot(self, other):
        return (self._x * other._x
                + self._y * other._y
                + self._z * other._z)

    def cross(self, other):
        return self.__class__(
            self._y * other._z - self._z * other._y,
            self._z * other._x - self._x * other._z,
            self._x * other._y - self._y * other._x,
        )

    def length_sq(self): return self.dot(self)
    def length(self): return math.sqrt(self.length_sq())

    def normalize(self):
        mag = self.length()
        return self / mag if mag > 0 else self.__class__(0.0, 0.0, 0.0)

    def __neg__(self): return self.__class__(-self._x, -self._y, -self._z)
    def __repr__(self): return f"Vec3({self._x}, {self._y}, {self._z})"
    def __hash__(self): return hash((self._x, self._y, self._z))

    def __eq__(self, other):
        return (isinstance(other, Vec3)
                and self._x == other._x
                and self._y == other._y
                and self._z == other._z)

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z

    def __getitem__(self, index):
        if index == 0: return self._x
        if index == 1: return self._y
        if index == 2: return self._z
        raise IndexError(f"Vec3 index {index} out of range")

    def to_dict(self):
        return [self._x, self._y, self._z]

    @classmethod
    def from_dict(cls, data):
        return cls(data[0], data[1], data[2])

# Semantic aliases — clarify intent at call sites without extra overhead.
Point2 = Vec2
Point3 = Vec3


class Color(Vec3):
    """RGB color stored as (r, g, b) floats, typically in [0, 1]."""

    __slots__ = ()

    @property
    def r(self): return self._x

    @property
    def g(self): return self._y

    @property
    def b(self): return self._z

    def __repr__(self):
        return f"Color(r={self._x}, g={self._y}, b={self._z})"


class Ray:
    """A ray defined by an origin point and a normalized direction vector."""

    __slots__ = ('_origin', '_direction', '_ior_stack')

    def __init__(self, origin, direction, ior_stack=None):
        self._origin = origin
        self._direction = direction.normalize()
        self._ior_stack = ior_stack or [1.0]  # Start in air by default

    @property
    def current_ior(self):
        return self._ior_stack[-1]
    
    @property
    def ior_stack(self):
        return self._ior_stack
    

    
    @property
    def origin(self): return self._origin

    @property
    def direction(self): return self._direction

    def at(self, t):
        """Return the point along the ray at parameter t: origin + t * direction."""
        return self._origin + self._direction * t

    def __repr__(self):
        return f"Ray(origin={self._origin}, direction={self._direction})"

    def __iter__(self):
        yield self._origin
        yield self._direction   

    def __getitem__(self, index):
        if index == 0: return self._origin
        if index == 1: return self._direction
        raise IndexError(f"Ray index {index} out of range")