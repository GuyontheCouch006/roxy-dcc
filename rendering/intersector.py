import importlib
from dataclasses import dataclass

import numpy as np

from core import Point3, Ray, Vec3
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

    def intersect_raw_arrays(self, origins, directions, max_t=None):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        hits = self.intersect_many(_rays_from_arrays(origins, directions))
        return _hit_records_to_raw(hits, max_ts)

    def occluded_raw_arrays(self, origins, directions, max_t):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        return np.asarray(
            self.occluded_many(_rays_from_arrays(origins, directions), max_ts),
            dtype=np.bool_,
        )


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

    def intersect_raw_arrays(self, origins, directions, max_t=None):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        result = _empty_raw_hits(len(origins))

        for i, (ro, rd, ray_max_t) in enumerate(zip(origins, directions, max_ts)):
            hit = self._intersect_vectors(ro, rd, max_t=ray_max_t, any_hit=False)
            if hit is None:
                continue
            prim_id, t, u, v = hit
            result["hit"][i] = True
            result["t"][i] = t
            result["u"][i] = u
            result["v"][i] = v
            result["tri_id"][i] = prim_id

        return result

    def occluded_raw_arrays(self, origins, directions, max_t):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        blocked = np.zeros(len(origins), dtype=np.bool_)

        for i, (ro, rd, ray_max_t) in enumerate(zip(origins, directions, max_ts)):
            blocked[i] = self._intersect_vectors(
                ro,
                rd,
                max_t=ray_max_t,
                any_hit=True,
            ) is not None

        return blocked

    def _intersect_arrays(self, ray, max_t, any_hit):
        return self._intersect_vectors(
            _vec3_array(ray.origin),
            _vec3_array(ray.direction),
            max_t=max_t,
            any_hit=any_hit,
        )

    def _intersect_vectors(self, ro, rd, max_t, any_hit):
        if self.triangle_count == 0:
            return None

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


@dataclass(frozen=True)
class _EmbreeBinding:
    name: str
    scene_cls: object
    mesh_cls: object


class EmbreeIntersector(SceneIntersector):
    """Embree-backed intersector over flattened mesh triangles.

    Embree is deliberately kept behind this optional adapter. The Python package
    owns the acceleration structure while TriangleSceneArrays owns shading data,
    so Embree primID/u/v results can be turned back into regular HitRecords.
    """

    _T_NEAR = 0.001

    def __init__(
        self,
        world=None,
        *,
        triangle_scene=None,
        fallback=None,
        binding_modules=None,
    ):
        if triangle_scene is None:
            if world is None:
                raise ValueError("EmbreeIntersector needs a world or triangle_scene")
            triangle_scene = flatten_world_triangles(world)

        self._triangle_scene = triangle_scene
        self._fallback = fallback
        self._binding = _load_embree_binding(binding_modules)
        self._scene = None
        self._mesh = None
        self._vertices = None
        self._indices = None
        self._run_cache = {}
        self._occlusion_query_failed = False
        self._build_embree_scene()

    @classmethod
    def available(cls, *, binding_modules=None):
        try:
            _load_embree_binding(binding_modules)
        except EmbreeUnavailableError:
            return False
        return True

    @property
    def triangle_count(self):
        return self._triangle_scene.triangle_count

    @property
    def backend_name(self):
        return self._binding.name

    def intersect(self, ray):
        return self.intersect_many([ray])[0]

    def occluded(self, ray, max_t):
        return bool(self.occluded_many([ray], [max_t])[0])

    def intersect_many(self, rays):
        origins, directions = _ray_arrays_from_rays(rays)
        raw = self.intersect_raw_arrays(origins, directions)
        hits = []
        for ray, is_hit, tri_id, t, u, v in zip(
            rays,
            raw["hit"],
            raw["tri_id"],
            raw["t"],
            raw["u"],
            raw["v"],
        ):
            hit = None
            if is_hit:
                hit = self._triangle_scene.hit_record(ray, tri_id, t, u, v)
            if self._fallback is not None:
                fallback_hit = self._fallback.intersect(ray)
                if fallback_hit and (hit is None or fallback_hit < hit):
                    hit = fallback_hit
            hits.append(hit)
        return hits

    def occluded_many(self, rays, max_ts):
        origins, directions = _ray_arrays_from_rays(rays)
        blocked = self.occluded_raw_arrays(origins, directions, max_ts)
        if self._fallback is not None:
            blocked = blocked.copy()
            for i, (ray, max_t) in enumerate(zip(rays, max_ts)):
                blocked[i] = blocked[i] or self._fallback.occluded(ray, max_t)
        return blocked.tolist()

    def intersect_raw_arrays(self, origins, directions, max_t=None):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        if self.triangle_count == 0:
            return _empty_raw_hits(len(origins))

        result = self._run_embree(
            origins,
            directions,
            query="INTERSECT",
            max_ts=max_ts,
            output=True,
        )
        return self._raw_hits_from_embree_result(result, max_ts)

    def occluded_raw_arrays(self, origins, directions, max_t):
        origins, directions = _ray_input_arrays(origins, directions)
        max_ts = _max_t_array(max_t, len(origins))
        if self.triangle_count == 0:
            return np.zeros(len(origins), dtype=np.bool_)

        if not self._occlusion_query_failed:
            try:
                result = self._run_embree(
                    origins,
                    directions,
                    query="OCCLUDED",
                    max_ts=max_ts,
                    output=False,
                )
                return self._occlusion_from_embree_result(result, max_ts)
            except Exception:
                self._occlusion_query_failed = True

        return self.intersect_raw_arrays(origins, directions, max_ts)["hit"]

    def _build_embree_scene(self):
        scene_errors = []
        dtype_candidates = (
            (np.float32, np.uint32),
            (np.float32, np.int32),
            (np.float64, np.int64),
        )

        for vertex_dtype, index_dtype in dtype_candidates:
            vertices = np.ascontiguousarray(
                self._triangle_scene.vertices,
                dtype=vertex_dtype,
            )
            indices = np.ascontiguousarray(
                self._triangle_scene.indices,
                dtype=index_dtype,
            )
            try:
                scene = self._binding.scene_cls()
                mesh = self._binding.mesh_cls(scene, vertices, indices)
                if hasattr(scene, "commit"):
                    scene.commit()
                self._scene = scene
                self._mesh = mesh
                self._vertices = vertices
                self._indices = indices
                return
            except Exception as exc:
                scene_errors.append(
                    f"{vertex_dtype.__name__}/{index_dtype.__name__}: {exc}"
                )

        detail = "; ".join(scene_errors)
        raise EmbreeUnavailableError(
            f"Embree binding '{self._binding.name}' loaded, but scene "
            f"construction failed ({detail})."
        )

    def _run_embree(self, origins, directions, query, max_ts, output):
        cached = self._run_cache.get((query, output))
        if cached is not None:
            return cached(origins, directions, max_ts)

        errors = []
        query_names = (query, query.lower())

        def candidate_call(name, kwargs_factory):
            def call(ro, rd, tfar):
                kwargs = kwargs_factory(tfar)
                return self._scene.run(ro, rd, **kwargs)
            return name, call

        candidates = []
        for query_name in query_names:
            candidates.extend([
                candidate_call(
                    f"{query_name} output/dists",
                    lambda tfar, q=query_name: {
                        "query": q,
                        "output": output,
                        "dists": np.asarray(tfar, dtype=np.float32).copy(),
                    },
                ),
                candidate_call(
                    f"{query_name} output/tfar",
                    lambda tfar, q=query_name: {
                        "query": q,
                        "output": output,
                        "tfar": np.asarray(tfar, dtype=np.float32).copy(),
                    },
                ),
                candidate_call(
                    f"{query_name} output",
                    lambda tfar, q=query_name: {
                        "query": q,
                        "output": output,
                    },
                ),
            ])

        for label, call in candidates:
            try:
                result = call(origins, directions, max_ts)
                self._run_cache[(query, output)] = call
                return result
            except TypeError as exc:
                errors.append(f"{label}: {exc}")

        raise EmbreeUnavailableError(
            f"Embree binding '{self._binding.name}' does not expose a "
            f"compatible {query} ray query ({'; '.join(errors)})."
        )

    def _raw_hits_from_embree_result(self, result, max_ts):
        count = len(max_ts)
        raw = _empty_raw_hits(count)
        prim_ids = _result_array(result, "primID", "prim_id", "primitive_id")
        tfar = _result_array(result, "tfar", "t", "distance")
        if prim_ids is None or tfar is None:
            raise EmbreeUnavailableError(
                "Embree query output did not include primID and tfar fields."
            )

        prim_ids = _as_result_vector(prim_ids, count).astype(np.int64, copy=False)
        tfar = _as_result_vector(tfar, count).astype(np.float32, copy=False)
        u = _as_result_vector(
            _result_array(result, "u", default=np.zeros(count, dtype=np.float32)),
            count,
        ).astype(np.float32, copy=False)
        v = _as_result_vector(
            _result_array(result, "v", default=np.zeros(count, dtype=np.float32)),
            count,
        ).astype(np.float32, copy=False)

        valid = (
            (prim_ids >= 0)
            & (prim_ids < self.triangle_count)
            & np.isfinite(tfar)
            & (tfar > self._T_NEAR)
            & (tfar < max_ts)
        )

        raw["hit"][:] = valid
        raw["t"][valid] = tfar[valid]
        raw["u"][valid] = u[valid]
        raw["v"][valid] = v[valid]
        raw["tri_id"][valid] = prim_ids[valid].astype(np.int32, copy=False)
        geom_ids = _result_array(result, "geomID", "geom_id", default=None)
        if geom_ids is not None:
            geom_ids = _as_result_vector(geom_ids, count).astype(np.int64, copy=False)
            raw["geom_id"][valid] = geom_ids[valid].astype(np.int32, copy=False)
        return raw

    def _occlusion_from_embree_result(self, result, max_ts):
        count = len(max_ts)
        if isinstance(result, np.ndarray) and result.dtype == np.bool_:
            return _as_result_vector(result, count).astype(np.bool_, copy=False)
        if isinstance(result, np.ndarray):
            prim_ids = _as_result_vector(result, count).astype(np.int64, copy=False)
            return (prim_ids >= 0) & (prim_ids < self.triangle_count)
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], bool):
            return np.asarray(result, dtype=np.bool_)
        if isinstance(result, (list, tuple)):
            prim_ids = np.asarray(result, dtype=np.int64).reshape((count,))
            return (prim_ids >= 0) & (prim_ids < self.triangle_count)

        tfar = _result_array(result, "tfar", "t", "distance")
        prim_ids = _result_array(result, "primID", "prim_id", "primitive_id")
        if tfar is not None:
            tfar = _as_result_vector(tfar, count).astype(np.float32, copy=False)
            occluded = np.isfinite(tfar) & (tfar > self._T_NEAR) & (tfar < max_ts)
            occluded |= tfar < 0.0
            if prim_ids is not None:
                prim_ids = _as_result_vector(prim_ids, count).astype(
                    np.int64,
                    copy=False,
                )
                occluded &= (prim_ids >= 0) & (prim_ids < self.triangle_count)
            return occluded
        if prim_ids is not None:
            prim_ids = _as_result_vector(prim_ids, count).astype(np.int64, copy=False)
            return (prim_ids >= 0) & (prim_ids < self.triangle_count)
        raise EmbreeUnavailableError("Embree occlusion output was not understood.")


def _vec3_array(value):
    return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def np_cross_broadcast(vec, rows):
    return np.cross(np.broadcast_to(vec, rows.shape), rows)


def _load_embree_binding(binding_modules=None):
    module_pairs = binding_modules or (
        ("embreex.rtcore_scene", "embreex.mesh_construction"),
        ("pyembree.rtcore_scene", "pyembree.mesh_construction"),
    )
    errors = []

    for scene_module_name, mesh_module_name in module_pairs:
        try:
            scene_module = importlib.import_module(scene_module_name)
            mesh_module = importlib.import_module(mesh_module_name)
        except Exception as exc:
            errors.append(f"{scene_module_name}/{mesh_module_name}: {exc}")
            continue

        scene_cls = getattr(scene_module, "EmbreeScene", None)
        mesh_cls = getattr(mesh_module, "TriangleMesh", None)
        if scene_cls is None or mesh_cls is None:
            errors.append(
                f"{scene_module_name}/{mesh_module_name}: missing "
                "EmbreeScene or TriangleMesh"
            )
            continue

        package_name = scene_module_name.split(".", 1)[0]
        return _EmbreeBinding(package_name, scene_cls, mesh_cls)

    raise EmbreeUnavailableError(
        "No compatible Embree Python binding is installed. Tried: "
        + "; ".join(errors)
    )


def _ray_input_arrays(origins, directions):
    origins = np.ascontiguousarray(origins, dtype=np.float32).reshape((-1, 3))
    directions = np.ascontiguousarray(directions, dtype=np.float32).reshape((-1, 3))
    if len(origins) != len(directions):
        raise ValueError("origins and directions must have the same length")
    return origins, directions


def _ray_arrays_from_rays(rays):
    origins = np.asarray(
        [[ray.origin.x, ray.origin.y, ray.origin.z] for ray in rays],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[ray.direction.x, ray.direction.y, ray.direction.z] for ray in rays],
        dtype=np.float32,
    )
    return _ray_input_arrays(origins, directions)


def _rays_from_arrays(origins, directions):
    return [
        Ray(Point3(float(origin[0]), float(origin[1]), float(origin[2])),
            Vec3(float(direction[0]), float(direction[1]), float(direction[2])))
        for origin, direction in zip(origins, directions)
    ]


def _max_t_array(max_t, count):
    if max_t is None:
        return np.full(count, np.inf, dtype=np.float32)
    max_ts = np.asarray(max_t, dtype=np.float32)
    if max_ts.ndim == 0:
        return np.full(count, float(max_ts), dtype=np.float32)
    return np.ascontiguousarray(max_ts, dtype=np.float32).reshape((count,))


def _empty_raw_hits(count):
    return {
        "hit": np.zeros(count, dtype=np.bool_),
        "t": np.full(count, np.inf, dtype=np.float32),
        "u": np.zeros(count, dtype=np.float32),
        "v": np.zeros(count, dtype=np.float32),
        "tri_id": np.full(count, -1, dtype=np.int32),
        "geom_id": np.full(count, -1, dtype=np.int32),
    }


def _hit_records_to_raw(hits, max_ts):
    raw = _empty_raw_hits(len(hits))
    for i, hit in enumerate(hits):
        if hit is None or not hit or not (hit.t < max_ts[i]):
            continue
        raw["hit"][i] = True
        raw["t"][i] = float(hit.t)
    return raw


def _result_array(result, *keys, default=None):
    if isinstance(result, dict):
        for key in keys:
            if key in result:
                return np.asarray(result[key])
        return default
    if isinstance(result, np.ndarray) and result.dtype.names:
        for key in keys:
            if key in result.dtype.names:
                return np.asarray(result[key])
        return default
    for key in keys:
        if hasattr(result, key):
            return np.asarray(getattr(result, key))
    return default


def _as_result_vector(value, count):
    value = np.asarray(value)
    if value.ndim == 0:
        return np.full(count, value.item())
    return value.reshape((count,))
