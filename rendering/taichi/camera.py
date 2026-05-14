import taichi as ti


@ti.func
def _fract(value):
    return value - ti.floor(value)


@ti.func
def _hash_float(x, y, salt):
    seed = ti.cast(x, ti.f32) * 127.1 + ti.cast(y, ti.f32) * 311.7 + salt * 74.7
    return _fract(ti.sin(seed) * 43758.5453123)


@ti.func
def _pixel_sample_offset(px, py, frame):
    i = ti.cast(frame + 1, ti.f32)
    jitter_x = _fract(_hash_float(px, py, 0.0) + i * 0.7548776662466927)
    jitter_y = _fract(_hash_float(px, py, 1.0) + i * 0.5698402909980532)
    return jitter_x, jitter_y


@ti.func
def get_ray_direction(px, py, W, H, frame, fov_tan, aspect, cam_fwd, cam_right, cam_up):
    jitter_x, jitter_y = _pixel_sample_offset(px, py, frame)
    u = (px + jitter_x) / W * 2.0 - 1.0
    v = 1.0 - (py + jitter_y) / H * 2.0
    ndc_x = u * aspect * fov_tan
    ndc_y = v * fov_tan
    return (cam_fwd + cam_right * ndc_x + cam_up * ndc_y).normalized()
