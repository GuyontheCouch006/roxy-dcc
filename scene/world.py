# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: World scene container holding all renderable objects, lights,
#              and cameras. Owns the closest-hit ray query and sky gradient.
# ============================================

from core.vectors import Color
from scene.scene_object import SceneObject
from scene.camera import Camera


class World:
    """Container for all objects, lights, and cameras in the scene.

    The first camera added automatically becomes the active camera.
    intersect() skips non-renderable objects and returns the closest hit.
    """

    def __init__(
        self,
        objects=None,
        lights=None,
        cameras=None,
        background_color=None,
        use_sky=True,
    ):
        self._objects = objects if objects is not None else []
        self._lights = lights if lights is not None else []
        self._cameras = cameras if cameras is not None else []
        self._background_color = (
            background_color if background_color is not None else Color(0, 0, 0)
        )
        self._use_sky = use_sky
        self._active_camera = self._cameras[0] if self._cameras else None

    # ─── Add / remove ─────────────────────────────────────────────────────────

    def add_object(self, obj):
        """Add a SceneObject to the world."""
        self._objects.append(obj)

    def add_light(self, light):
        """Add a Light to the world."""
        self._lights.append(light)

    def add_camera(self, camera):
        """Add a Camera to the world, making it active if none is set yet."""
        self._cameras.append(camera)
        if not self._active_camera:
            self._active_camera = camera

    def remove_object(self, obj):
        """Remove a SceneObject from the world."""
        self._objects.remove(obj)

    def remove_light(self, light):
        """Remove a Light from the world."""
        self._lights.remove(light)

    def remove_camera(self, camera):
        """Remove a Camera from the world, reassigning active_camera if needed."""
        self._cameras.remove(camera)
        if self._active_camera == camera:
            self._active_camera = self._cameras[0] if self._cameras else None

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def use_sky(self): return self._use_sky

    @use_sky.setter
    def use_sky(self, status):
        self._use_sky = status

    @property
    def objects(self): return self._objects

    @property
    def lights(self): return self._lights

    @property
    def cameras(self): return self._cameras

    @property
    def background_color(self): return self._background_color

    @background_color.setter
    def background_color(self, color): self._background_color = color

    @property
    def active_camera(self): return self._active_camera

    @active_camera.setter
    def active_camera(self, camera):
        """Set the active camera. Raises ValueError if camera is not in the world."""
        if camera not in self._cameras:
            raise ValueError(
                "Camera must be added to the world before it can be set as active."
            )
        self._active_camera = camera

    # ─── Ray queries ──────────────────────────────────────────────────────────

    def intersect(self, ray):
        """Return the closest HitRecord along ray, skipping non-renderable objects."""
        closest = None
        for obj in self._objects:
            if not obj.renderable:
                continue
            hit = obj.intersect(ray)
            if hit and (closest is None or hit < closest):
                closest = hit
        return closest

    def occluded(self, ray, max_t):
        """Return True if any renderable object blocks ray before max_t."""
        for obj in self._objects:
            if not obj.renderable:
                continue
            if obj.occluded(ray, max_t):
                return True
        return False

    def sky_color(self, ray):
        """Return the background color for a ray that hit nothing.

        When use_sky is True, lerps from white (horizon) to blue (zenith)
        based on the ray's y direction. Falls back to background_color otherwise.
        """
        if not self._use_sky:
            return self._background_color
        t = 0.5 * (ray.direction.y + 1.0)
        white = Color(1.0, 1.0, 1.0)
        blue = Color(0.5, 0.7, 1.0)
        return white * (1.0 - t) + blue * t

    def __repr__(self):
        return (f"World(objects={len(self._objects)}, "
                f"lights={len(self._lights)}, "
                f"cameras={len(self._cameras)})")

    def to_dict(self):
        return {
            "objects": [obj.to_dict() for obj in self._objects],
            "lights": [],  # TODO: implement lights
            "cameras": [cam.to_dict() for cam in self._cameras],
            "background_color": self._background_color.to_dict(),
            "use_sky": self._use_sky,
        }

    @classmethod
    def from_dict(cls, data):
        world = cls(
            background_color=Color.from_dict(data["background_color"]),
            use_sky=data["use_sky"],
        )
        for obj_data in data["objects"]:
            world.add_object(SceneObject.from_dict(obj_data))
        for cam_data in data["cameras"]:
            world.add_camera(Camera.from_dict(cam_data))
        return world
