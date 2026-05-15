USE_TAICHI  = True
DEBUG_LEVEL = 0   # 0=off  1=milestone timers  2=timers + cProfile

import random
import core.timing as timing
timing.LEVEL = DEBUG_LEVEL

from core import Vec3, Color, Point3
from rendering.taichi_renderer import TaichiRenderer
from rendering.ray_tracer import RayTracer
from scene import Sphere, Plane, Cube, Diffuse, SceneObject, World, Camera
from scene.materials import Emissive, Metal, Glossy
from rendering import Image, GLViewport
from scene.io import load_scene
from scene.io.obj_reader import OBJReader
from rendering.startup_progress import StartupProgress


def build_rabbit():
    world = World(use_sky=False)

    # ── Materials ─────────────────────────────────────────────────────────────
    fur    = Diffuse(Color(0.88, 0.85, 0.82))
    belly  = Diffuse(Color(0.95, 0.93, 0.90))
    pink   = Diffuse(Color(0.92, 0.60, 0.66))
    eye_m  = Metal(Color(0.04, 0.03, 0.03), roughness=0.04)
    ground = Diffuse(Color(0.50, 0.42, 0.28))
    orange = Diffuse(Color(0.96, 0.44, 0.06))
    cgreen = Diffuse(Color(0.12, 0.52, 0.10))

    def grass_mat():
        g = random.uniform(0.40, 0.58)
        r = random.uniform(0.10, 0.20)
        return Diffuse(Color(r, g, 0.08))

    # ── Ground ────────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),
        material=ground, name="ground",
    ))

    # ── Body ──────────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Sphere(1.0), material=fur,
        translation=Vec3(0, 1.10, 0),
        scale=Vec3(0.88, 0.96, 0.82),
        name="body",
    ))
    world.add_object(SceneObject(                        # belly patch
        shape=Sphere(1.0), material=belly,
        translation=Vec3(0, 0.88, 0.52),
        scale=Vec3(0.52, 0.58, 0.38),
        name="belly",
    ))

    # ── Head ──────────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Sphere(1.0), material=fur,
        translation=Vec3(0, 2.15, 0.32),
        scale=Vec3(0.58, 0.58, 0.58),
        name="head",
    ))
    world.add_object(SceneObject(                        # muzzle
        shape=Sphere(1.0), material=belly,
        translation=Vec3(0, 1.98, 0.80),
        scale=Vec3(0.29, 0.22, 0.22),
        name="snout",
    ))
    world.add_object(SceneObject(                        # nose
        shape=Sphere(1.0), material=pink,
        translation=Vec3(0, 2.02, 0.98),
        scale=Vec3(0.056, 0.046, 0.046),
        name="nose",
    ))

    # ── Eyes ──────────────────────────────────────────────────────────────────
    for side, ex in (("L", -0.22), ("R", 0.22)):
        world.add_object(SceneObject(
            shape=Sphere(1.0), material=eye_m,
            translation=Vec3(ex, 2.28, 0.66),
            scale=Vec3(0.088, 0.088, 0.088),
            name=f"eye_{side}",
        ))
        world.add_object(SceneObject(                    # specular highlight dot
            shape=Sphere(1.0), material=belly,
            translation=Vec3(ex + 0.028, 2.31, 0.72),
            scale=Vec3(0.026, 0.026, 0.026),
            name=f"eye_hi_{side}",
        ))

    # ── Ears ──────────────────────────────────────────────────────────────────
    for side, ex, ry in (("L", -0.24, -10), ("R", 0.24, 10)):
        world.add_object(SceneObject(
            shape=Cube(1.0), material=fur,
            translation=Vec3(ex, 3.05, 0.14),
            rotation=Vec3(4, ry, 0),
            scale=Vec3(0.14, 0.88, 0.09),
            name=f"ear_{side}",
        ))
        world.add_object(SceneObject(                    # inner ear
            shape=Cube(1.0), material=pink,
            translation=Vec3(ex, 3.05, 0.20),
            rotation=Vec3(4, ry, 0),
            scale=Vec3(0.075, 0.74, 0.030),
            name=f"ear_inner_{side}",
        ))

    # ── Tail ──────────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Sphere(1.0), material=belly,
        translation=Vec3(0, 1.02, -0.86),
        scale=Vec3(0.29, 0.26, 0.23),
        name="tail",
    ))

    # ── Back feet ─────────────────────────────────────────────────────────────
    for side, ex in (("L", -0.44), ("R", 0.44)):
        world.add_object(SceneObject(
            shape=Sphere(1.0), material=fur,
            translation=Vec3(ex, 0.19, 0.30),
            scale=Vec3(0.28, 0.14, 0.50),
            name=f"foot_{side}",
        ))

    # ── Front paws ────────────────────────────────────────────────────────────
    world.add_object(SceneObject(                        # left paw (resting)
        shape=Sphere(1.0), material=fur,
        translation=Vec3(-0.52, 0.26, 0.66),
        scale=Vec3(0.20, 0.13, 0.28),
        name="paw_L",
    ))
    world.add_object(SceneObject(                        # right arm (raised)
        shape=Sphere(1.0), material=fur,
        translation=Vec3(0.50, 0.76, 0.52),
        scale=Vec3(0.16, 0.32, 0.16),
        name="arm_R",
    ))
    world.add_object(SceneObject(                        # right paw (holding carrot)
        shape=Sphere(1.0), material=fur,
        translation=Vec3(0.54, 0.44, 0.70),
        scale=Vec3(0.18, 0.13, 0.22),
        name="paw_R",
    ))

    # ── Carrot ────────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Cube(1.0), material=orange,
        translation=Vec3(0.74, 0.86, 0.64),
        rotation=Vec3(15, 12, 28),
        scale=Vec3(0.10, 0.44, 0.10),
        name="carrot",
    ))
    world.add_object(SceneObject(                        # pointed tip
        shape=Sphere(1.0), material=orange,
        translation=Vec3(0.82, 0.56, 0.74),
        scale=Vec3(0.072, 0.072, 0.072),
        name="carrot_tip",
    ))
    for i, (ox, oz, rx, rz) in enumerate([
        ( 0.00,  0.00, -30,  0),
        (-0.04,  0.03, -25, 18),
        ( 0.04, -0.02, -28,-15),
        ( 0.01,  0.05, -20, 30),
    ]):
        world.add_object(SceneObject(
            shape=Cube(1.0), material=cgreen,
            translation=Vec3(0.74 + ox, 1.14, 0.64 + oz),
            rotation=Vec3(rx, 0, rz),
            scale=Vec3(0.030, 0.20, 0.028),
            name=f"carrot_green_{i}",
        ))

    # ── Grass patch ───────────────────────────────────────────────────────────
    random.seed(7)
    for i in range(28):
        x  = random.uniform(-1.6, 1.6)
        z  = random.uniform(-1.0, 1.2)
        h  = random.uniform(0.14, 0.38)
        tx = random.uniform(-14, 14)
        tz = random.uniform(-14, 14)
        ry = random.uniform(0, 180)
        world.add_object(SceneObject(
            shape=Cube(1.0), material=grass_mat(),
            translation=Vec3(x, h * 0.5, z),
            rotation=Vec3(tx, ry, tz),
            scale=Vec3(0.032, h, 0.032),
            name=f"grass_{i}",
        ))

    # ── 3-point lighting (emissive spheres) ───────────────────────────────────
    world.add_object(SceneObject(                        # key — warm, front-left-high
        shape=Sphere(1.0),
        material=Emissive(Color(1.00, 0.92, 0.76), intensity=42.0),
        translation=Vec3(-4.5, 7.0, 5.5),
        scale=Vec3(0.5, 0.5, 0.5),
        name="key_light",
    ))
    world.add_object(SceneObject(                        # fill — cool, front-right
        shape=Sphere(1.0),
        material=Emissive(Color(0.72, 0.84, 1.00), intensity=14.0),
        translation=Vec3(5.0, 3.5, 4.5),
        scale=Vec3(0.4, 0.4, 0.4),
        name="fill_light",
    ))
    world.add_object(SceneObject(                        # rim — violet, back-high
        shape=Sphere(1.0),
        material=Emissive(Color(0.88, 0.72, 1.00), intensity=22.0),
        translation=Vec3(0.5, 6.0, -5.5),
        scale=Vec3(0.35, 0.35, 0.35),
        name="rim_light",
    ))

    return world


def debug_scene_object(node, indent=0):
    """Print a SceneObject hierarchy with shapes and materials."""
    pad = "  " * indent
    tris = ""
    if node.shapes:
        geo = node.shapes[0].geometry
        tris = f"  [{type(geo).__name__}"
        if hasattr(geo, '_triangles'):
            tris += f" × {len(geo._triangles)}"
        tris += "]"
    print(f"{pad}▸ {node.name!r}{tris}")
    for shape in node.shapes:
        for group_name, mat in shape.material_groups.items():
            albedo = getattr(mat, '_albedo', None)
            color_str = f"  rgb({albedo.r:.2f}, {albedo.g:.2f}, {albedo.b:.2f})" if albedo else ""
            print(f"{pad}    mat[{group_name!r}] = {type(mat).__name__}{color_str}")
    for child in node.children:
        debug_scene_object(child, indent + 1)


def build_cornell_from_obj(W, H):
    OBJ = "sample_scenes/CornellBox/CornellBox-Glossy.obj"

    root = OBJReader.load(OBJ)
    debug_scene_object(root)

    world = World(use_sky=False)
    world.add_object(root)

    camera = Camera(
        position=Point3(0, .795, 3.85),
        forward=Vec3(0, 0, -1),
        fov=37,
        width=W,
        height=H,
    )
    world.add_camera(camera)
    return world


def build_glossy_test(W, H):
    world = World(use_sky=True)

    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),
        material=Glossy(Color(0.55, 0.55, 0.55), roughness=0.08),
        name="ground",
    ))

    gold = Color(0.85, 0.65, 0.12)
    roughnesses = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i, roughness in enumerate(roughnesses):
        world.add_object(SceneObject(
            shape=Sphere(1.0),
            material=Glossy(gold, roughness=roughness),
            translation=Vec3((i - 2) * 2.5, 1.0, 0.0),
            name=f"sphere_r{int(roughness * 100):03d}",
        ))

    camera = Camera(
        position=Point3(0, 2.5, 10),
        forward=Vec3(0, -0.15, -1),
        fov=55,
        width=W,
        height=H,
    )
    world.add_camera(camera)
    return world

def build_dragon(W, H):
    OBJ = "sample_scenes/dragon.obj"
    root = OBJReader.load(OBJ)
    debug_scene_object(root)
    root.rotation = Vec3(0,90,0)
    root.scale = Vec3(2,2,2)
    aabb = root.world_aabb
    y = -aabb.min.y
    root.translation = Vec3(0, y, 0)
    root.children[0].material = Diffuse(Color(.5, .2, .14))
    world = World(use_sky=False)
    world.add_object(root)

    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),
        material=Glossy(Color(0.45, 0.45, 0.45), roughness=0.05),
        name="ground",
    ))

def build_bicycle(W, H):
    OBJ = "sample_scenes/roadBike/roadBike.obj"
    root = OBJReader.load(OBJ)
    debug_scene_object(root)
    root.rotation = Vec3(0,-10,0)
    world = World(use_sky=True)
    world.add_object(root)

    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),
        material=Glossy(Color(0.45, 0.45, 0.45), roughness=0.05),
        name="ground",
    ))

    world.add_object(SceneObject(                        # key — warm, front-left-high
        shape=Sphere(1.0),
        material=Emissive(Color(1.00, 0.92, 0.76), intensity=50.0),
        translation=Vec3(-3.0, 4.5, 4.0),
        scale=Vec3(0.6, 0.6, 0.6),
        name="key_light",
    ))
    world.add_object(SceneObject(                        # fill — cool, front-right
        shape=Sphere(1.0),
        material=Emissive(Color(0.72, 0.84, 1.00), intensity=20.0),
        translation=Vec3(4.0, 2.5, 3.0),
        scale=Vec3(0.5, 0.5, 0.5),
        name="fill_light",
    ))
    world.add_object(SceneObject(                        # rim — violet, back-high
        shape=Sphere(1.0),
        material=Emissive(Color(0.88, 0.72, 1.00), intensity=30.0),
        translation=Vec3(0.5, 5.0, -4.0),
        scale=Vec3(0.4, 0.4, 0.4),
        name="rim_light",
    ))

    camera = Camera(
        position=Point3(1.2, 0.8, 2.0),
        forward=Vec3(-2.0, -0.5, -3.1),
        fov=50,
        width=W,
        height=H,
    )
    world.add_camera(camera)
    return world


def build_gallery(W, H, progress=None):
    OBJ = "sample_scenes/gallery/gallery.obj"
    if progress:
        progress.step("Loading gallery mesh", OBJ)
    root = OBJReader.load(
        OBJ,
        progress=progress.update if progress else None,
    )
    if progress:
        progress.step("Inspecting gallery hierarchy")
    debug_scene_object(root)

    if progress:
        progress.step("Building scene objects", "Adding gallery geometry and ceiling lights.")
    world = World(use_sky=False)
    world.add_object(root)

    # Gallery bounding box: X -6.2→5.0, Y 0.06→6.26, Z -14.0→11.4
    # Warm overhead strip lights along the central axis just below the ceiling
    warm_white = Color(1.00, 0.96, 0.88)
    for z in range(-12, 10, 4):
        world.add_object(SceneObject(
            shape=Sphere(1.0),
            material=Emissive(warm_white, intensity=25.0),
            translation=Vec3(-0.6, 5.7, float(z)),
            scale=Vec3(0.25, 0.25, 0.25),
            name=f"ceiling_light_{z}",
        ))

    camera = Camera(
        position=Point3(-0.9, 1.7, 9.0),
        forward=Vec3(0.0, -0.05, -1.0),
        fov=70,
        width=W,
        height=H,
    )
    world.add_camera(camera)
    return world


def main():
    W, H = int(1920), int(1080)
    progress = StartupProgress(
        "Roxy startup",
        total_steps=9 if USE_TAICHI else 8,
    )

    progress.step("Starting Roxy", f"Preparing {W} x {H} render.")
    with timing.timed("build_gallery", tag="scene"):
        world = build_gallery(W, H, progress)

    progress.step("Allocating image buffer")
    image    = Image(W, H)
    progress.step("Opening viewport", "Creating the OpenGL preview window.")
    viewport = GLViewport(W, H, "Picture Gallery – Hallwyl Museum")

    progress.step(
        "Preparing renderer",
        "Taichi GPU renderer" if USE_TAICHI else "Python path tracer",
    )
    if USE_TAICHI:
        tracer = TaichiRenderer(
            world, image, viewport,
            samples=2000,
            max_depth=5,
            direct_light_max_depth=1,
            direct_light_mode="one",
            denoise=True,
            startup_progress=progress,
        )
    else:
        tracer = RayTracer(
            world, image, viewport,
            samples=64,
            max_depth=8,
            direct_light_mode="all",
            denoise=True,
            startup_progress=progress,
        )

    try:
        with timing.profile("render"):
            tracer.render()
    finally:
        progress.close()

    while not viewport.should_close:
        viewport.poll_events()
        viewport.update(image)

    viewport.close()


if __name__ == "__main__":
    main()
