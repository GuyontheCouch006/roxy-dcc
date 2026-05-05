# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: SceneObject combines a shape, material, and transform into a
#              renderable entity. Ray intersection is performed in local space
#              and results are transformed back to world space.
# ============================================

from core import Transform, HitRecord, Ray, RotationOrder, Vec3


class SceneObject:
    """A renderable object in the scene, defined by a shape, material, and transform.

    Intersection works in two passes:
      1. AABB early-out in world space (cheap reject).
      2. Full intersection in local object space, result transformed back to world.
    """

    def __init__(
        self,
        shape,
        material,
        name="",
        translation=None,
        rotation=None,
        scale=None,
        visible=True,
        renderable=True,
        selectable=True,
        rotation_order=RotationOrder.XYZ,
        parent=None,
    ):
        self._name = name
        self._shape = shape
        self._material = material
        self._transform = Transform(
            translation=translation,
            rotation=rotation,
            scale=scale,
            shape=shape,
            rotation_order=rotation_order,
        )
        self._visible = visible
        self._renderable = renderable
        self._selectable = selectable
        self._parent = parent

    def intersect(self, world_ray):
        """Test world_ray against this object. Returns a HitRecord or None."""
        if not self._shape.is_infinite:
            # AABB early-out — avoids expensive per-triangle tests when possible.
            if not self._transform.world_aabb.intersect(world_ray):
                return None

        # Transform the ray into local object space for the shape test.
        inv = self._transform.world_inverse_matrix
        local_origin = inv.transform_point(world_ray._origin)
        local_direction = inv.transform_vector(world_ray._direction)
        local_ray = Ray(local_origin, local_direction)

        hit = self._shape.intersect(local_ray)
        if hit is None:
            return None

        # Transform hit results back to world space.
        # Recompute t in world space — local t is not valid across a scaled transform.
        M = self._transform.world_matrix
        world_point = M.transform_point(hit.point)
        world_t = (world_point - world_ray._origin).dot(world_ray._direction)
        world_normal = inv.transpose().transform_vector(hit.normal).normalize()

        return HitRecord.from_ray(world_ray, world_t, world_normal, self._material)

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self): return self._name

    @name.setter
    def name(self, value): self._name = value

    @property
    def shape(self): return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self._transform.shape = value  # Keeps the world AABB in sync.

    @property
    def material(self): return self._material

    @material.setter
    def material(self, value): self._material = value

    @property
    def transform(self): return self._transform

    @property
    def translation(self): return self._transform.translation

    @translation.setter
    def translation(self, value): self._transform.translation = value

    @property
    def rotation(self): return self._transform.rotation

    @rotation.setter
    def rotation(self, value): self._transform.rotation = value

    @property
    def scale(self): return self._transform.scale

    @scale.setter
    def scale(self, value): self._transform.scale = value

    @property
    def shear(self): return self._transform.shear

    @shear.setter
    def shear(self, value): self._transform.shear = value

    @property
    def pivot(self): return self._transform.pivot

    @pivot.setter
    def pivot(self, value): self._transform.pivot = value

    @property
    def rotation_order(self): return self._transform.rotation_order

    @rotation_order.setter
    def rotation_order(self, value): self._transform.rotation_order = value

    @property
    def visible(self): return self._visible

    @visible.setter
    def visible(self, value): self._visible = value

    @property
    def renderable(self): return self._renderable

    @renderable.setter
    def renderable(self, value): self._renderable = value

    @property
    def selectable(self): return self._selectable

    @selectable.setter
    def selectable(self, value): self._selectable = value

    def taichi_export(self):
        """Return a flat dict of world-space GPU data for this object."""
        from scene.shapes import Sphere, Plane, Cube
        shape = self._shape
        M   = self._transform.world_matrix
        inv = self._transform.world_inverse_matrix
        mat      = self._material
        albedo   = list(mat._albedo)
        mat_type = mat.taichi_type_id()
        params   = mat.taichi_params()
        roughness = params[0] if mat_type == 1 else 0.0
        ior       = params[0] if mat_type == 2 else 1.0
        emission  = params[0] if mat_type == 3 else 0.0

        def _mat():
            return {'albedo': albedo, 'mat_type': mat_type,
                    'roughness': roughness, 'ior': ior, 'emission': emission}

        if isinstance(shape, Sphere):
            center = list(M.transform_point(Vec3(0, 0, 0)))
            s = self._transform.scale
            radius = shape._radius * max(abs(s.x), abs(s.y), abs(s.z))
            return {'type': 0, 'center': center, 'radius': radius,
                    'normal': [0.0, 1.0, 0.0], 'offset': 0.0, 'extra': [0.0, 0.0, 0.0],
                    **_mat()}

        if isinstance(shape, Plane):
            n_world = inv.transpose().transform_vector(shape._normal).normalize()
            p_local = Vec3(shape._normal.x * -shape._distance,
                           shape._normal.y * -shape._distance,
                           shape._normal.z * -shape._distance)
            p_world = M.transform_point(p_local)
            offset  = -n_world.dot(p_world)
            return {'type': 1, 'center': [0.0, 0.0, 0.0], 'radius': 0.0,
                    'normal': list(n_world), 'offset': offset, 'extra': [0.0, 0.0, 0.0],
                    **_mat()}

        if isinstance(shape, Cube):
            center = list(M.transform_point(Vec3(0, 0, 0)))
            s = self._transform.scale
            half = shape._side_length / 2
            return {'type': 2, 'center': center, 'radius': 0.0,
                    'normal': [0.0, 1.0, 0.0], 'offset': 0.0,
                    'extra': [half * abs(s.x), half * abs(s.y), half * abs(s.z)],
                    **_mat()}

        raise NotImplementedError(f"taichi_export not supported for {type(shape).__name__}")

    def __repr__(self):
        return (f"SceneObject(name={self._name!r}, shape={self._shape}, "
                f"material={self._material})")

    def to_dict(self):
        return {
            "name": self._name,
            "shape": self._shape.to_dict(),
            "material": self._material.to_dict(),
            "transform": self._transform.to_dict(),
            "visible": self._visible,
            "renderable": self._renderable,
            "selectable": self._selectable,
        }

    @classmethod
    def from_dict(cls, data):
        from scene.shapes import create_shape_from_dict
        from scene.materials import create_material_from_dict
        return cls(
            name=data["name"],
            shape=create_shape_from_dict(data["shape"]),
            material=create_material_from_dict(data["material"]),
            translation=Vec3.from_dict(data["transform"]["translation"]),
            rotation=Vec3.from_dict(data["transform"]["rotation"]),
            scale=Vec3.from_dict(data["transform"]["scale"]),
            shear=Vec3.from_dict(data["transform"]["shear"]),
            pivot=Vec3.from_dict(data["transform"]["pivot"]),
            rotation_order=RotationOrder(data["transform"]["rotation_order"]),
            visible=data["visible"],
            renderable=data["renderable"],
            selectable=data["selectable"],
        )
