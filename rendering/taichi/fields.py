import taichi as ti

MAX_OBJECTS = 1024
MAX_W = 1920
MAX_H = 1080

_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_H, MAX_W))
_frame_count  = ti.field(dtype=ti.i32, shape=())

_obj_type   = ti.field(dtype=ti.i32, shape=MAX_OBJECTS)
_obj_center = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_radius = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_offset = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)  # plane: n·p + offset = 0

_obj_extra    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)  # cube: half-extents

_obj_mat_type  = ti.field(dtype=ti.i32, shape=MAX_OBJECTS)
_obj_albedo    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_OBJECTS)
_obj_roughness = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_ior       = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)
_obj_emission  = ti.field(dtype=ti.f32, shape=MAX_OBJECTS)  # emissive intensity, 0 otherwise

_n_objects = ti.field(dtype=ti.i32, shape=())

