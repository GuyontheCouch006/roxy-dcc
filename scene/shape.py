# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Shape — pairs geometry (Primitive or Mesh) with material groups.
#              Shape is the bridge between raw geometry and material assignment.
#              One Shape per geometry/material combination under a SceneObject.
# ============================================

from core import HitRecord


class Shape:
    """Pairs a geometry primitive or mesh with a material group mapping.

    material_groups is a dict mapping group name string → Material.
    Faces/triangles carry a group tag. Shape looks up the material at intersect time.
    """

    def __init__(self, geometry, material_groups=None, name=""):
        self._geometry        = geometry
        self._material_groups = material_groups or {}
        self._default_material = None
        self._name            = name

    @property
    def geometry(self): return self._geometry

    @property
    def material_groups(self): return self._material_groups

    @property
    def name(self): return self._name

    def set_default_material(self, material):
        self._default_material = material

    def material_for_group(self, group_name):
        return (self._material_groups.get(group_name)
                or self._default_material)

    def local_bounds(self):
        return self._geometry.local_bounds()

    @property
    def is_infinite(self):
        return getattr(self._geometry, 'is_infinite', False)

    def intersect(self, ray):
        """Intersect ray with geometry, attach material, return HitRecord or None."""
        hit = self._geometry.intersect(ray)
        if hit is None:
            return None

        group    = getattr(hit, 'group', 'default')
        material = self.material_for_group(group)
        uv       = getattr(hit, 'uv', None)

        return HitRecord.from_ray(ray, hit.t, hit.normal, material=material, uv=uv)

    def __repr__(self):
        return f"Shape(geometry={self._geometry}, groups={list(self._material_groups.keys())})"

    def to_dict(self):
        return {
            "name":     self._name,
            "geometry": self._geometry.to_dict(),
            "material_groups": {
                k: v.to_dict() for k, v in self._material_groups.items()
            },
        }

    @classmethod
    def from_dict(cls, data):
        from scene.primitives import create_primitive_from_dict
        from scene.materials import create_material_from_dict

        return cls(
            create_primitive_from_dict(data["geometry"]),
            {
                k: create_material_from_dict(v)
                for k, v in data.get("material_groups", {}).items()
            },
            name=data.get("name", ""),
        )
