import taichi as ti

MAX_OBJECTS   = 10_000_000
MAX_W         = 1920
MAX_H         = 1080

MAX_BVH_NODES = 4_000_000   # 2× road-bike triangle count
MAX_TRIANGLES = 2_000_000   # road-bike quads → ~1.68 M triangles
MAX_MATERIALS = 4_096
MAX_LIGHTS    = 4_096

_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_frame_count  = ti.field(dtype=ti.i32, shape=())
_ray_count    = ti.field(dtype=ti.i32, shape=())

_obj_type   = ti.field(dtype=ti.i32, shape=MAX_OBJECTS)
_obj_center = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_radius = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_offset = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)  # plane: n·p + offset = 0

_obj_extra    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)  # cube: half-extents

_obj_v0 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)  # triangle vertex 0
_obj_v1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)  # triangle vertex 1
_obj_v2 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)  # triangle vertex 2

_obj_mat_type  = ti.field(dtype=ti.i32, shape=MAX_OBJECTS)
_obj_albedo    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_roughness = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_ior       = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_emission  = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)  # emissive intensity, 0 otherwise

_n_objects = ti.field(dtype=ti.i32, shape=())

# ── Explicit light sampling ─────────────────────────────────────────────────
_light_center   = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
_light_radius   = ti.field(dtype=ti.f32, shape=MAX_LIGHTS)
_light_albedo   = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
_light_emission = ti.field(dtype=ti.f32, shape=MAX_LIGHTS)
_n_lights       = ti.field(dtype=ti.i32, shape=())

# ── BVH nodes ──────────────────────────────────────────────────────────────
_bvh_aabb_min  = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BVH_NODES)
_bvh_aabb_max  = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BVH_NODES)
_bvh_left      = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)
_bvh_right     = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)
_bvh_tri_start = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)
_bvh_tri_count = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)
_bvh_n_nodes   = ti.field(dtype=ti.i32, shape=())

# ── BVH triangles (reordered by BVH build) ─────────────────────────────────
_bvh_v0      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_v1      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_v2      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_n0      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_n1      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_n2      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_mat_idx = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
_bvh_n_tris  = ti.field(dtype=ti.i32, shape=())

# ── Material palette ────────────────────────────────────────────────────────
_mat_type      = ti.field(dtype=ti.i32,           shape=MAX_MATERIALS)
_mat_albedo    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MATERIALS)
_mat_roughness = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_ior       = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_emission  = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_n_mats    = ti.field(dtype=ti.i32,           shape=())
