# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Perspective camera with configurable position, orientation, and FOV.
#              Generates world-space rays from pixel coordinates via shoot().
# ============================================

import math
from core.vectors import Point3, Vec3, Ray


class Camera:
    """Perspective camera that generates rays through an image plane."""

    __slots__ = (
        '_position', '_forward', '_up', '_right',
        '_fov', '_aspect_ratio', 'name',
    )

    def __init__(
        self,
        position=Point3(),
        forward=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        fov=90,
        width=800,
        height=600,
        name="camera",
    ):
        self._position = position
        self._forward = forward.normalize()
        self._right = self._forward.cross(up.normalize()).normalize()
        self._up = self._right.cross(self._forward).normalize()
        self._fov = fov
        self._aspect_ratio = width / height if height != 0 else 1.0
        self.name = name

    def shoot(self, x, y, width, height):
        """Generate a ray through pixel (x, y) in screen space.

        Converts the pixel to NDC in [-1, 1], scales by FOV and aspect ratio,
        then delegates to get_ray.
        """
        assert abs(width/height - self._aspect_ratio) < 0.01, \
            f"Image dimensions {width}x{height} don't match camera aspect ratio {self._aspect_ratio:.3f}"
        u = (x + 0.5) / width * 2 - 1
        v = 1 - (y + 0.5) / height * 2
        half_fov = math.tan(math.radians(self._fov) / 2)
        ndc_x = u * self._aspect_ratio * half_fov
        ndc_y = v * half_fov
        return self.get_ray(ndc_x, ndc_y)

    def get_ray(self, u, v):
        """Return a ray from the camera origin through NDC point (u, v)."""
        direction = (self._forward + self._right * u + self._up * v).normalize()
        return Ray(self._position, direction)

    def set_width_height(self, width, height):
        """Update the camera's aspect ratio based on new image dimensions."""
        self._aspect_ratio = width / height if height != 0 else 1.0

    @property
    def position(self): return self._position

    @position.setter
    def position(self, pos): self._position = pos

    @property
    def forward(self): return self._forward

    @forward.setter
    def forward(self, fwd):
        self._forward = fwd.normalize()
        self._right = self._forward.cross(self._up).normalize()
        self._up = self._right.cross(self._forward).normalize()

    @property
    def up(self): return self._up

    @up.setter
    def up(self, up):
        self._up = up.normalize()
        self._right = self._forward.cross(self._up).normalize()
        self._up = self._right.cross(self._forward).normalize()

    @property
    def fov(self): return self._fov

    @fov.setter
    def fov(self, fov): self._fov = fov

    @property
    def aspect_ratio(self): return self._aspect_ratio

    @property
    def right(self): return self._right

    
    def __repr__(self):
        return (f"Camera(position={self._position}, forward={self._forward}, "
                f"up={self._up}, fov={self._fov}, "
                f"aspect_ratio={self._aspect_ratio})")

    def to_dict(self):
        return {
            "name": self.name,
            "position": self._position.to_dict(),
            "forward": self._forward.to_dict(),
            "up": self._up.to_dict(),
            "fov": self._fov,
            "aspect_ratio": self._aspect_ratio,
        }

    @classmethod
    def from_dict(cls, data):
        cam = cls(
            position=Point3.from_dict(data["position"]),
            forward=Vec3.from_dict(data["forward"]),
            up=Vec3.from_dict(data["up"]),
            fov=data["fov"],
            name=data.get("name", "camera"),
        )
        return cam