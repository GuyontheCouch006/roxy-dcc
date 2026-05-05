import math

import taichi as ti


@ti.func
def random_unit_vector():
    theta = ti.acos(1.0 - 2.0 * ti.random())
    phi = 2.0 * math.pi * ti.random()
    sin_theta = ti.sin(theta)
    return ti.Vector([sin_theta * ti.cos(phi), sin_theta * ti.sin(phi), ti.cos(theta)])


@ti.func
def diffuse_scatter(normal):
    s = normal + random_unit_vector()
    if s.dot(s) < 1e-8:
        s = normal
    return s.normalized()

@ti.func
def reflect(rd, normal):
    return rd - 2*rd.dot(normal)*normal

@ti.func
def refract(rd, normal, eta):
    cos_theta = min(-rd.dot(normal), 1.0)
    r_out_perp = eta * (rd + cos_theta * normal)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * normal
    return r_out_perp + r_out_parallel

@ti.func
def metal_scatter(rd, normal, roughness):
    reflected = reflect(rd, normal)
    scattered = (reflected + random_unit_vector() * roughness).normalized()
    return scattered

@ti.func
def schlick(cosine, eta):
    r0 = ((1.0 - eta) / (1.0 + eta)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cosine) ** 5

@ti.func
def dielectric_scatter(rd, normal, eta):  # eta = n1/n2, already computed
    cos_theta = min(-rd.dot(normal), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    cannot_refract = eta * sin_theta > 1.0
    result = ti.Vector([0.0, 0.0, 0.0])
    if cannot_refract or schlick(cos_theta, eta) > ti.random():
        result = reflect(rd, normal)
    else:
        result = refract(rd, normal, eta)
    return result