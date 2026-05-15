import numpy as np

from rendering.scene_arrays import flatten_world_triangles


class SceneIntersector:
    """Ray-query interface for scalar and batched renderer backends."""

    def intersect(self, ray):
        raise NotImplementedError

    def occluded(self, ray, max_t):
        raise NotImplementedError

    def intersect_many(self, rays):
        return [self.intersect(ray) for ray in rays]

    def occluded_many(self, rays, max_ts):
        return [
            self.occluded(ray, max_t)
            for ray, max_t in zip(rays, max_ts)
        ]


class WorldIntersector(SceneIntersector):
    """Default adapter around the existing World scalar ray queries."""

    def __init__(self, world):
        self._world = world

    def intersect(self, ray):
        return self._world.intersect(ray)

    def occluded(self, ray, max_t):
        return self._world.occluded(ray, max_t)


class CompositeIntersector(SceneIntersector):
    """Query multiple intersectors and return the closest hit.

    This is mainly useful during the Embree migration: an Embree mesh backend can
    handle triangles while WorldIntersector keeps unsupported primitives working.
    """

    def __init__(self, *intersectors):
        self._intersectors = [i for i in intersectors if i is not None]

    def intersect(self, ray):
        closest = None
        for intersector in self._intersectors:
            hit = intersector.intersect(ray)
            if hit and (closest is None or hit < closest):
                closest = hit
        return closest

    def occluded(self, ray, max_t):
        for intersector in self._intersectors:
            if intersector.occluded(ray, max_t):
                return True
        return False


class TriangleArrayIntersector(SceneIntersector):
    """Reference NumPy triangle intersector over flattened world-space meshes.

    This is not intended to beat the existing BVH. It exists as an executable
    contract for the Embree adapter: same triangle arrays in, same HitRecord out.
    """

    def __init__(self, triangle_scene):
        self._scene = triangle_scene
        vertices = triangle_scene.vertices
        indices = triangle_scene.indices
        self._v0 = vertices[indices[:, 0]] if triangle_scene.triangle_count else vertices
        self._v1 = vertices[indices[:, 1]] if triangle_scene.triangle_count else vertices
        self._v2 = vertices[indices[:, 2]] if triangle_scene.triangle_count else vertices
        self._eps = 1e-8
        self._t_min = 0.001

    @classmethod
    def from_world(cls, world):
        return cls(flatten_world_triangles(world))

    @property
    def triangle_count(self):
        return self._scene.triangle_count

    def intersect(self, ray):
        hit = self._intersect_arrays(ray, max_t=float("inf"), any_hit=False)
        if hit is None:
            return None
        prim_id, t, u, v = hit
        return self._scene.hit_record(ray, prim_id, t, u, v)

    def occluded(self, ray, max_t):
        return self._intersect_arrays(ray, max_t=max_t, any_hit=True) is not None

    def _intersect_arrays(self, ray, max_t, any_hit):
        if self.triangle_count == 0:
            return None

        ro = _vec3_array(ray.origin)
        rd = _vec3_array(ray.direction)
        e1 = self._v1 - self._v0
        e2 = self._v2 - self._v0
        h = np_cross_broadcast(rd, e2)
        a = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(a) > self._eps
        if not valid.any():
            return None

        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = ro - self._v0
        u = f * np.einsum("ij,ij->i", s, h)
        valid &= (u >= 0.0) & (u <= 1.0)
        if not valid.any():
            return None

        q = np.cross(s, e1)
        v = f * np.einsum("j,ij->i", rd, q)
        valid &= (v >= 0.0) & (u + v <= 1.0)
        if not valid.any():
            return None

        t = f * np.einsum("ij,ij->i", e2, q)
        valid &= (t > self._t_min) & (t < max_t)
        if not valid.any():
            return None

        if any_hit:
            prim_id = int(np.nonzero(valid)[0][0])
        else:
            masked_t = np.where(valid, t, np.inf)
            prim_id = int(np.argmin(masked_t))
        return prim_id, float(t[prim_id]), float(u[prim_id]), float(v[prim_id])


class EmbreeUnavailableError(RuntimeError):
    pass


class EmbreeIntersector(SceneIntersector):
    """Placeholder adapter documenting the intended Embree boundary.

    The current environment does not ship an Embree binding. This class fails
    loudly so benchmark scripts can report that dependency gap cleanly.
    """

    def __init__(self, world, *, fallback=None):
        self._triangle_scene = flatten_world_triangles(world)
        self._fallback = fallback
        raise EmbreeUnavailableError(
            "No Embree Python binding is installed. Expected future adapter: "
            "build from TriangleSceneArrays.vertices/indices and return "
            "HitRecord via primID/u/v side tables."
        )

    def intersect(self, ray):
        raise NotImplementedError

    def occluded(self, ray, max_t):
        raise NotImplementedError


def _vec3_array(value):
    return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def np_cross_broadcast(vec, rows):
    return np.cross(np.broadcast_to(vec, rows.shape), rows)
