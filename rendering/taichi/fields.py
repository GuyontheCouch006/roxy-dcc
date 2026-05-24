import taichi as ti

MAX_OBJECTS   = 10_000_000
MAX_W         = 1920
MAX_H         = 1080

MAX_BVH_NODES = 4_000_000   # 2× road-bike triangle count
MAX_TRIANGLES = 2_000_000   # road-bike quads → ~1.68 M triangles
MAX_MATERIALS = 4_096
MAX_LIGHTS    = 4_096
MAX_TEXTURES  = 8
MAX_TEXTURE_SIZE = 2_048

_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_normal_accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_albedo_accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_depth_accumulator = ti.field(dtype=ti.f32, shape=(MAX_H, MAX_W))
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
_bvh_uv0     = ti.Vector.field(2, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_uv1     = ti.Vector.field(2, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_uv2     = ti.Vector.field(2, dtype=ti.f32, shape=MAX_TRIANGLES)
_bvh_has_uv  = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
_bvh_mat_idx = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
_bvh_n_tris  = ti.field(dtype=ti.i32, shape=())

# ── Material palette ────────────────────────────────────────────────────────
_mat_type      = ti.field(dtype=ti.i32,           shape=MAX_MATERIALS)
_mat_albedo    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MATERIALS)
_mat_roughness = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_ior       = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_emission  = ti.field(dtype=ti.f32,           shape=MAX_MATERIALS)
_mat_texture   = ti.field(dtype=ti.i32,           shape=MAX_MATERIALS)
_mat_n_mats    = ti.field(dtype=ti.i32,           shape=())

# ── Texture cache ───────────────────────────────────────────────────────────
_tex_pixels = ti.Vector.field(
    3, dtype=ti.u8, shape=(MAX_TEXTURES, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE)
)
_tex_width   = ti.field(dtype=ti.i32, shape=MAX_TEXTURES)
_tex_height  = ti.field(dtype=ti.i32, shape=MAX_TEXTURES)
_tex_flip_v  = ti.field(dtype=ti.i32, shape=MAX_TEXTURES)
_tex_n       = ti.field(dtype=ti.i32, shape=())

# ── Wavefront path-tracing queues ───────────────────────────────────────────
MAX_RAYS = MAX_W * MAX_H   # 2,073,600 at 1920×1080

# Per-ray state (updated each bounce)
_wf_ro         = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_rd         = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_throughput = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_ior        = ti.field(dtype=ti.f32,           shape=MAX_RAYS)
_wf_active     = ti.field(dtype=ti.i32,           shape=MAX_RAYS)
_wf_color      = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_ray_count  = ti.field(dtype=ti.i32,           shape=MAX_RAYS)

# First-hit data for denoising accumulators (written once per sample)
_wf_first_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_first_albedo = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
_wf_first_depth  = ti.field(dtype=ti.f32,           shape=MAX_RAYS)

# Hit results written by wf_traverse, consumed by wf_shade
_wf_hit_t      = ti.field(dtype=ti.f32, shape=MAX_RAYS)
_wf_hit_u      = ti.field(dtype=ti.f32, shape=MAX_RAYS)
_wf_hit_v      = ti.field(dtype=ti.f32, shape=MAX_RAYS)
_wf_hit_tri    = ti.field(dtype=ti.i32, shape=MAX_RAYS)
_wf_hit_is_bvh = ti.field(dtype=ti.i32, shape=MAX_RAYS)
_wf_hit_bvh_mat= ti.field(dtype=ti.i32, shape=MAX_RAYS)
_wf_hit_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RAYS)
