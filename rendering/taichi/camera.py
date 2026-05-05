import taichi as ti


@ti.func
def get_ray_direction(px, py, W, H, fov_tan, aspect, cam_fwd, cam_right, cam_up):
    u = (px + ti.random()) / W * 2.0 - 1.0
    v = 1.0 - (py + ti.random()) / H * 2.0
    ndc_x = u * aspect * fov_tan
    ndc_y = v * fov_tan
    return (cam_fwd + cam_right * ndc_x + cam_up * ndc_y).normalized()
