# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: SceneObject — renderable entity with a list of Shapes, a transform,
#              and optional parent/children hierarchy. Ray intersection is performed
#              in local space; results are transformed back to world space.
# ============================================

from core import Transform, HitRecord, Ray, RotationOrder, Vec3


class SceneObject:
    """A renderable node in the scene graph.

    Each SceneObject owns one or more Shape objects (geometry + material groups)
    and a Transform. It may have children that are tested recursively.

    Legacy convenience: pass shape= and material= to automatically wrap them
    in a Shape with a single 'default' material group.
    """

    def __init__(
        self,
        shapes=None,
        name="",
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        pivot=None,
        visible=True,
        renderable=True,
        selectable=True,
        rotation_order=RotationOrder.XYZ,
        # Legacy convenience params
        shape=None,
        material=None,
    ):
        self._name       = name
        self._shapes     = list(shapes) if shapes else []
        self._children   = []
        self._parent     = None
        self._visible    = visible
        self._renderable = renderable
        self._selectable = selectable

        if shape is not None:
            from scene.shape import Shape as ShapeNode
            mat_groups = {"default": material} if material is not None else {}
            self._shapes.append(ShapeNode(shape, mat_groups, name=name))

        first_geo = self._shapes[0].geometry if self._shapes else None
        self._transform = Transform(
            translation=translation,
            rotation=rotation,
            scale=scale,
            shear=shear,
            pivot=pivot,
            shape=first_geo,
            rotation_order=rotation_order,
        )

    # ─── Hierarchy ────────────────────────────────────────────────────────────

    def add_child(self, child):
        child._parent = self
        if child not in self._children:
            self._children.append(child)
        return child

    def remove_child(self, child):
        child._parent = None
        if child in self._children:
            self._children.remove(child)

    @property
    def children(self): return list(self._children)

    @property
    def parent(self): return self._parent

    @property
    def world_matrix(self):
        if self._parent:
            return self._parent.world_matrix * self._transform.world_matrix
        return self._transform.world_matrix

    @property
    def world_inverse_matrix(self):
        if self._parent:
            return self._transform.world_inverse_matrix * self._parent.world_inverse_matrix
        return self._transform.world_inverse_matrix

    @property
    def world_inverse_transpose_matrix(self):
        return self.world_inverse_matrix.transpose()

    @property
    def world_aabb(self):
        bounds = None

        if self._shapes:
            # union of all shape bounds transformed to world space
            local_bounds = None
            for shape in self._shapes:
                b = shape.local_bounds()
                if b:
                    local_bounds = b if local_bounds is None else local_bounds.union(b)
            if local_bounds:
                bounds = local_bounds.transform(self.world_matrix)
        
        if self._children:
            # union of all children world AABBs
            for child in self._children:
                child_bounds = child.world_aabb
                if child_bounds:
                    bounds = child_bounds if bounds is None else bounds.union(child_bounds)
            if bounds:
                return bounds
        
        if bounds:
            return bounds
        
        # fallback — use transform's AABB
        return self._transform.world_aabb
    
    # ─── Intersection ─────────────────────────────────────────────────────────

    def intersect(self, world_ray):
        if not self._renderable:
            return None

        if self._can_use_aabb_early_out():
            if not self.world_aabb.intersect(world_ray):
                return None

        inv = self.world_inverse_matrix
        local_origin    = inv.transform_point(world_ray._origin)
        local_direction = inv.transform_vector(world_ray._direction)
        local_ray       = Ray(local_origin, local_direction)

        closest = None
        for shape in self._shapes:
            hit = shape.intersect(local_ray)
            if hit and (closest is None or hit < closest):
                closest = hit

        if closest is not None:
            closest = self._to_world_hit(world_ray, closest)

        for child in self._children:
            hit = child.intersect(world_ray)
            if hit and (closest is None or hit < closest):
                closest = hit

        return closest

    def occluded(self, world_ray, max_t):
        """Return True if this node blocks world_ray before max_t."""
        if not self._renderable:
            return False

        if self._can_use_aabb_early_out():
            if not self.world_aabb.intersect(world_ray):
                return False

        inv = self.world_inverse_matrix
        local_origin    = inv.transform_point(world_ray._origin)
        local_direction = inv.transform_vector(world_ray._direction)
        local_ray       = Ray(local_origin, local_direction)

        M = self.world_matrix
        for shape in self._shapes:
            hit = shape.geometry.intersect(local_ray)
            if hit is not None:
                world_point = M.transform_point(local_ray.at(hit.t))
                world_t = (world_point - world_ray._origin).dot(world_ray._direction)
                if 0.001 < world_t < max_t:
                    return True

        for child in self._children:
            if child.occluded(world_ray, max_t):
                return True

        return False

    def _can_use_aabb_early_out(self):
        if not self._shapes and not self._children:
            return False
        return not self._has_infinite_shape_recursive()

    def _has_infinite_shape_recursive(self):
        if any(shape.is_infinite for shape in self._shapes):
            return True
        return any(child._has_infinite_shape_recursive() for child in self._children)

    def _to_world_hit(self, world_ray, hit):
        M            = self.world_matrix
        world_point  = M.transform_point(hit.point)
        world_t      = (world_point - world_ray._origin).dot(world_ray._direction)
        world_normal = self.world_inverse_transpose_matrix.transform_vector(hit.normal).normalize()
        return HitRecord.from_ray(world_ray, world_t, world_normal,
                                  hit.material, uv=hit.uv)

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self): return self._name

    @name.setter
    def name(self, value): self._name = value

    @property
    def shapes(self): return self._shapes

    # Legacy single-shape accessors
    @property
    def shape(self):
        return self._shapes[0].geometry if self._shapes else None

    @shape.setter
    def shape(self, value):
        if self._shapes:
            self._shapes[0]._geometry = value
        self._transform.shape = value

    @property
    def material(self):
        if self._shapes:
            return self._shapes[0].material_for_group('default')
        return None

    @material.setter
    def material(self, value):
        if self._shapes:
            self._shapes[0]._material_groups['default'] = value

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

    # ─── Taichi export ────────────────────────────────────────────────────────

    def taichi_export(self):
        """Return a flat dict (or list of dicts) of world-space GPU data."""
        from scene.primitives import Sphere, Plane, Cube
        from scene.mesh import Triangle, Mesh

        shape = self.shape
        if shape is None:
            return []

        M   = self.world_matrix
        inv = self.world_inverse_matrix
        mat = self.material

        if mat is None:
            from scene.materials import Diffuse
            from core import Color
            mat = Diffuse(Color(0.8, 0.8, 0.8))

        albedo    = list(mat._albedo)
        mat_type  = mat.taichi_type_id()
        params    = mat.taichi_params()
        roughness = params[0] if mat_type in (1, 4) else 0.0
        ior       = params[0] if mat_type == 2 else 1.0
        emission  = params[0] if mat_type == 3 else 0.0

        def _mat():
            return {'albedo': albedo, 'mat_type': mat_type,
                    'roughness': roughness, 'ior': ior, 'emission': emission}

        if isinstance(shape, Sphere):
            center = list(M.transform_point(Vec3(0, 0, 0)))
            sx = M.transform_vector(Vec3(1, 0, 0)).length()
            sy = M.transform_vector(Vec3(0, 1, 0)).length()
            sz = M.transform_vector(Vec3(0, 0, 1)).length()
            radius = shape._radius * max(sx, sy, sz)
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
            half = shape._side_length / 2
            sx = M.transform_vector(Vec3(1, 0, 0)).length()
            sy = M.transform_vector(Vec3(0, 1, 0)).length()
            sz = M.transform_vector(Vec3(0, 0, 1)).length()
            return {'type': 2, 'center': center, 'radius': 0.0,
                    'normal': [0.0, 1.0, 0.0], 'offset': 0.0,
                    'extra': [half * sx, half * sy, half * sz],
                    **_mat()}

        def _mat_for(group_name):
            shape_wrapper = self._shapes[0] if self._shapes else None
            tri_mat = shape_wrapper.material_for_group(group_name) if shape_wrapper else None
            if tri_mat is None:
                tri_mat = mat
            tri_albedo   = list(tri_mat._albedo)
            tri_type     = tri_mat.taichi_type_id()
            tri_params   = tri_mat.taichi_params()
            return {
                'albedo':    tri_albedo,
                'mat_type':  tri_type,
                'roughness': tri_params[0] if tri_type in (1, 4) else 0.0,
                'ior':       tri_params[0] if tri_type == 2 else 1.0,
                'emission':  tri_params[0] if tri_type == 3 else 0.0,
            }

        def _tri_slot(tri):
            group = getattr(tri, 'group', 'default')
            return {'type': 4,
                    'center': [0.0, 0.0, 0.0], 'radius': 0.0,
                    'normal': [0.0, 1.0, 0.0], 'offset': 0.0, 'extra': [0.0, 0.0, 0.0],
                    'v0': list(M.transform_point(tri._v0)),
                    'v1': list(M.transform_point(tri._v1)),
                    'v2': list(M.transform_point(tri._v2)),
                    **_mat_for(group)}

        if isinstance(shape, Triangle):
            return _tri_slot(shape)

        if isinstance(shape, Mesh):
            return [_tri_slot(tri) for tri in shape._triangles]

        raise NotImplementedError(f"taichi_export not supported for {type(shape).__name__}")

    # ─── Serialization ────────────────────────────────────────────────────────

    def __repr__(self):
        return f"SceneObject(name={self._name!r}, shapes={len(self._shapes)}, children={len(self._children)})"

    def to_dict(self):
        return {
            "name":       self._name,
            "shapes":     [shape.to_dict() for shape in self._shapes],
            "children":   [child.to_dict() for child in self._children],
            "transform":  self._transform.to_dict(),
            "visible":    self._visible,
            "renderable": self._renderable,
            "selectable": self._selectable,
        }

    @classmethod
    def from_dict(cls, data):
        from scene.primitives import create_primitive_from_dict
        from scene.materials import create_material_from_dict
        from scene.shape import Shape

        shapes = []
        if "shapes" in data:
            shapes = [Shape.from_dict(shape_data) for shape_data in data["shapes"]]
        elif data.get("shape") is not None:
            shapes = [Shape(
                create_primitive_from_dict(data["shape"]),
                {"default": create_material_from_dict(data["material"])} if data.get("material") else {},
                name=data.get("name", ""),
            )]

        obj = cls(
            name=data["name"],
            shapes=shapes,
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
        for child_data in data.get("children", []):
            obj.add_child(cls.from_dict(child_data))
        return obj
