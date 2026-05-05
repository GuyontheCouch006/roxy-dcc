from rendering.taichi.fields import (
    _obj_type, _obj_center, _obj_radius,
    _obj_normal, _obj_offset, _obj_extra,
    _obj_mat_type, _obj_albedo,
    _obj_roughness, _obj_ior, _obj_emission,
    _n_objects, MAX_OBJECTS,
)


def extract_scene(world):
    objects = [obj for obj in world.objects if obj.renderable]
    n = len(objects)
    assert n <= MAX_OBJECTS, f"Scene has {n} objects but MAX_OBJECTS={MAX_OBJECTS}"

    _n_objects[None] = n
    for i, obj in enumerate(objects):
        data = obj.taichi_export()
        _obj_type[i]     = data['type']
        _obj_center[i]   = data['center']
        _obj_radius[i]   = data['radius']
        _obj_normal[i]   = data['normal']
        _obj_offset[i]   = data['offset']
        _obj_extra[i]    = data['extra']
        _obj_albedo[i]    = data['albedo']
        _obj_mat_type[i]  = data['mat_type']
        _obj_roughness[i] = data['roughness']
        _obj_ior[i]       = data['ior']
        _obj_emission[i]  = data['emission']
