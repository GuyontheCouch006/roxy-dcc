
from dataclasses import dataclass
from pathlib import Path
from core import Color
from scene.materials import Diffuse, Metal, Dielectric, Emissive, Glossy
from scene.textures import ImageTexture

@dataclass
class MTLMaterial:
    name: str
    ka: Color = None    #ambient      
    kd: Color = None    # diffuse
    ks: Color = None    # specular  
    ke: Color = None    # emissive
    tf: Color = None    # transmission filter
    ns: float = 10.0    # shininess
    ni: float = 1.0     # IOR
    d: float = 1.0      # opacity
    tr: float = 0.0     # transparency (inverse of d)
    illum: int   = 1      # illumination model
    map_kd: str   = None   # diffuse texture path (store but don't load yet)

    def to_material(self):
        texture = ImageTexture(self.map_kd) if self.map_kd else None
        if self.ke and (self.ke.r > 0 or self.ke.g > 0 or self.ke.b > 0):
            return Emissive(self.ke, intensity=1.0, albedo_texture=texture)
        if self.d < 0.99:
            albedo = self.tf if self.tf else Color(1, 1, 1)
            return Dielectric(albedo, ior=self.ni, albedo_texture=texture)
        if self.illum in (3, 4, 5):
            roughness = max(0.0, 1.0 - self.ns / 1000.0)
            return Metal(self.kd or Color(1,1,1), roughness=roughness,
                         albedo_texture=texture)
        if self.ks and (self.ks.r > 0 or self.ks.g > 0 or self.ks.b > 0):
            roughness = max(0.0, 1.0 - self.ns / 1000.0)
            return Glossy(self.kd or Color(1,1,1), roughness=roughness,
                          albedo_texture=texture)
        return Diffuse(self.kd or Color(0.8, 0.8, 0.8), albedo_texture=texture)
    
class MTLLoader:
    @staticmethod
    def load(path) -> dict:
        """Parse a .mtl file and return dict of material name → Material."""
        base_dir = Path(path).parent
        materials = []
        current_material = None
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                line = line.rsplit('#')[0]
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == 'newmtl':
                    if current_material:
                        materials.append(current_material)  # save previous
                    current_material = MTLMaterial(name=parts[1])
                if parts[0] == 'Ns':
                    current_material.ns = float(parts[1])
                if parts[0] == 'Ni':
                    current_material.ni = float(parts[1])
                if parts[0] == 'd':
                    current_material.d = float(parts[1])
                if parts[0] == 'Tr':
                    current_material.tr = float(parts[1])
                    current_material.d  = 1.0 - current_material.tr  # ← add this
                if parts[0] == 'illum':
                    current_material.illum = int(parts[1])
                if parts[0] in ('map_kd', 'map_Kd'):
                    current_material.map_kd = str(base_dir / parts[-1])
                if parts[0] == 'Ka':
                    r,g,b = parts[1:]
                    color = Color(float(r), float(g), float(b))
                    current_material.ka = color
                if parts[0] == 'Kd':
                    r,g,b = parts[1:]
                    color = Color(float(r), float(g), float(b))
                    current_material.kd = color
                if parts[0] == 'Ks':
                    r,g,b = parts[1:]
                    color = Color(float(r), float(g), float(b))
                    current_material.ks = color
                if parts[0] == 'Ke':
                    r,g,b = parts[1:]
                    color = Color(float(r), float(g), float(b))
                    current_material.ke = color
                if parts[0] == 'Tf':
                    r,g,b = parts[1:]
                    color = Color(float(r), float(g), float(b))
                    current_material.tf = color
        if current_material:
            materials.append(current_material)

        return {mat.name: mat for mat in materials}
                   
