import taichi as ti


@ti.func
def sky_color(rd):
    t = 0.5 * (rd[1] + 1.0)
    return ti.Vector([1.0, 1.0, 1.0]) * (1.0 - t) + ti.Vector([0.5, 0.7, 1.0]) * t
