# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Abstract Material base class and surface material implementations
#              (Diffuse, etc.). Materials define how a surface scatters incoming rays.
# ============================================

from abc import ABC, abstractmethod
from math import sqrt
import random

from core.utils import random_unit_vector, random_cosine_hemisphere
from core import Ray, Color
from scene.textures import create_texture_from_dict

def schlick(cos_theta, ior):
    r0 = ((1 - ior) / (1 + ior)) ** 2
    return r0 + (1 - r0) * (1 - cos_theta) ** 5

def reflect(v, n):
    return v - 2 * v.dot(n) * n

def refract(uv, normal, eta):  # eta = n1/n2
    cos_theta = min(-uv.dot(normal), 1.0)
    r_perp = eta * (uv + cos_theta * normal)
    r_parallel = -sqrt(abs(1 - r_perp.length_sq())) * normal
    return r_perp + r_parallel

class Material(ABC):
    """Abstract base for all surface materials."""

    def __init__(self, albedo=None, albedo_texture=None):
        self._albedo = albedo or Color(1, 1, 1)  # Default to white if no albedo provided.
        self._albedo_texture = albedo_texture

    @abstractmethod
    def scatter(self, ray_in, hit_record): ...

    def albedo_at(self, hit_record):
        if self._albedo_texture is not None and getattr(hit_record, 'uv', None) is not None:
            return self._albedo_texture.sample(hit_record.uv)
        return self._albedo

    def _texture_dict(self):
        if self._albedo_texture is None:
            return None
        return self._albedo_texture.to_dict()

    def taichi_type_id(self): return 0
    def taichi_params(self): return []


class Diffuse(Material):
    """Lambertian diffuse material — scatters rays in a random hemisphere direction."""

    def __init__(self, albedo, albedo_texture=None):
        super().__init__(albedo, albedo_texture=albedo_texture)

    def taichi_type_id(self): return 0
    def taichi_params(self): return []

    def scatter(self, ray_in, hit_record):
        scatter_direction = random_cosine_hemisphere(hit_record.normal)
        if scatter_direction.length_sq() < 1e-8:
            scatter_direction = hit_record.normal
        scattered_ray = Ray(hit_record.point, scatter_direction)
        return scattered_ray, self.albedo_at(hit_record)
    
    def __repr__(self):
        return f"Diffuse(albedo={self._albedo})"

    def to_dict(self):
        data = {"type": "diffuse", "albedo": self._albedo.to_dict()}
        texture = self._texture_dict()
        if texture:
            data["albedo_texture"] = texture
        return data
    

class Metal(Material):
    """Metallic material — reflects rays with some fuzziness."""

    def __init__(self, albedo, roughness=0.0, albedo_texture=None):
        super().__init__(albedo, albedo_texture=albedo_texture)
        self._roughness = min(max(roughness, 0.0), 1.0)

    def taichi_type_id(self): return 1
    def taichi_params(self): return [self._roughness]

    def scatter(self, ray_in, hit_record):
        rd = ray_in.direction
        normal = hit_record.normal
        reflected = rd - 2*rd.dot(normal)*normal
        scattered = (reflected + random_unit_vector() * self._roughness).normalize()
        if scattered.dot(normal) >= 0:
            return Ray(hit_record.point, scattered), self.albedo_at(hit_record)
        return None
    
    def __repr__(self):
        return f"Metal(albedo={self._albedo}, roughness={self._roughness})"

    def to_dict(self):
        data = {"type": "metal", "albedo": self._albedo.to_dict(), "roughness": self._roughness}
        texture = self._texture_dict()
        if texture:
            data["albedo_texture"] = texture
        return data
    
class Dielectric(Material):
    """Dielectric material — refracts rays based on a given index of refraction."""

    def __init__(self, albedo, ior=1.5, albedo_texture=None):
        super().__init__(albedo, albedo_texture=albedo_texture)
        self._ior = ior

    def taichi_type_id(self): return 2
    def taichi_params(self): return [self._ior]
    
    
    def scatter(self, ray_in, hit):
        ray_in_ior = ray_in.current_ior
        eta = ray_in_ior/self._ior if hit.front_face else self._ior/ray_in_ior
        
        cos_theta = min(-ray_in.direction.dot(hit.normal), 1.0)
        sin_theta = sqrt(1 - cos_theta**2)
        
        cannot_refract = eta * sin_theta > 1.0
        
        if cannot_refract or schlick(cos_theta, eta) > random.random():
            direction = reflect(ray_in.direction, hit.normal)
            new_ior_stack = ray_in.ior_stack  # reflection — same medium, stack unchanged
        else:
            direction = refract(ray_in.direction, hit.normal, eta)
            if hit.front_face:
                new_ior_stack = ray_in.ior_stack + [self._ior]  # entering — push
            else:
                new_ior_stack = ray_in.ior_stack[:-1] or [1.0]  # exiting — pop

        return Ray(hit.point, direction, new_ior_stack), self.albedo_at(hit)
    
    def __repr__(self):
        return f"Dielectric(albedo={self._albedo}, ior={self._ior})"

    def to_dict(self):
        data = {"type": "dielectric", "albedo": self._albedo.to_dict(), "ior": self._ior}
        texture = self._texture_dict()
        if texture:
            data["albedo_texture"] = texture
        return data

class Emissive(Material):
    def __init__(self, color, intensity=1.0, albedo_texture=None):
        super().__init__(color, albedo_texture=albedo_texture)
        self._intensity = intensity

    def scatter(self, ray_in, hit_record):
        return None  # absorbed — no scatter

    def emitted(self):
        return self._albedo * self._intensity

    def taichi_type_id(self): return 3
    def taichi_params(self): return [self._intensity]

    def __repr__(self):
        return f"Emissive(color={self._albedo}, intensity={self._intensity})"

    def to_dict(self):
        data = {"type": "emissive", "albedo": self._albedo.to_dict(), "intensity": self._intensity}
        texture = self._texture_dict()
        if texture:
            data["albedo_texture"] = texture
        return data

class Glossy(Material):
    """Glossy material — a mix of diffuse and metal based on roughness."""

    def __init__(self, albedo, roughness=0.5, albedo_texture=None):
        super().__init__(albedo, albedo_texture=albedo_texture)
        self._roughness = min(max(roughness, 0.0), 1.0)

    def taichi_type_id(self): return 4
    def taichi_params(self): return [self._roughness]

    def scatter(self, ray_in, hit_record):
        albedo = self.albedo_at(hit_record)
        diffuse_part = Diffuse(albedo).scatter(ray_in, hit_record)
        metal_part = Metal(albedo, self._roughness).scatter(ray_in, hit_record)
        if diffuse_part and metal_part:
            if random.random() < self._roughness:
                return diffuse_part
            else:
                return metal_part
        return diffuse_part or metal_part

    def __repr__(self):
        return f"Glossy(albedo={self._albedo}, roughness={self._roughness})"

    def to_dict(self):
        data = {"type": "glossy", "albedo": self._albedo.to_dict(), "roughness": self._roughness}
        texture = self._texture_dict()
        if texture:
            data["albedo_texture"] = texture
        return data

def create_material_from_dict(data):
    mat_type = data["type"]
    albedo = Color.from_dict(data["albedo"])
    texture = create_texture_from_dict(data.get("albedo_texture"))
    if mat_type == "diffuse":
        return Diffuse(albedo, albedo_texture=texture)
    elif mat_type == "metal":
        return Metal(albedo, data["roughness"], albedo_texture=texture)
    elif mat_type == "dielectric":
        return Dielectric(albedo, data["ior"], albedo_texture=texture)
    elif mat_type == "emissive":
        return Emissive(albedo, data["intensity"], albedo_texture=texture)
    elif mat_type == "glossy":
        return Glossy(albedo, data["roughness"], albedo_texture=texture)
    else:
        raise ValueError(f"Unknown material type: {mat_type}")
