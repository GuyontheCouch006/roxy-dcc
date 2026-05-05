USE_TAICHI = True
DEBUG = False

if DEBUG:
    import cProfile
    import pstats

from core import Vec3, Color, Point3
from rendering.taichi_renderer import TaichiRenderer
from scene import Plane, Cube, Diffuse, SceneObject, World, Camera, save_scene
from scene.materials import Emissive
from rendering import Image, GLViewport


def build_cornell_box():
    world = World(use_sky=False)

    white = Diffuse(Color(0.73, 0.73, 0.73))
    red   = Diffuse(Color(0.65, 0.05, 0.05))
    green = Diffuse(Color(0.12, 0.45, 0.15))
    # ── Room walls ────────────────────────────────────────────────────────────
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),   # floor
        material=white,
        translation=Vec3(0, 0, 0),
        name="floor",
    ))
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, -1, 0)),  # ceiling
        material=white,
        translation=Vec3(0, 5.5, 0),
        name="ceiling",
    ))
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 0, 1)),   # back wall
        material=white,
        translation=Vec3(0, 0, -8),
        name="back wall",
    ))
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(1, 0, 0)),   # left wall (red)
        material=red,
        translation=Vec3(-2.75, 0, 0),
        name="left wall",
    ))
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(-1, 0, 0)),  # right wall (green)
        material=green,
        translation=Vec3(2.75, 0, 0),
        name="right wall",
    ))

    # ── Area light (flat panel flush with ceiling) ────────────────────────────
    world.add_object(SceneObject(
        shape=Cube(side_length=1.0),
        material=Emissive(Color(1.0, 0.95, 0.85), intensity=25.0),
        translation=Vec3(0, 5.48, -4.5),
        scale=Vec3(1.5, 0.05, 1.5),
        name="light",
    ))

    # ── Boxes ─────────────────────────────────────────────────────────────────
    # Tall box — left side
    world.add_object(SceneObject(
        shape=Cube(side_length=1.5),
        material=white,
        translation=Vec3(-1.1, 1.5, -5.5),
        scale=Vec3(1.0, 2.0, 1.0),
        name="tall box",
    ))
    # Short box — right side
    world.add_object(SceneObject(
        shape=Cube(side_length=1.5),
        material=white,
        translation=Vec3(1.1, 0.75, -4.0),
        name="short box",
    ))

    return world


def main():
    W, H = 768, 768  # square — classic Cornell box aspect

    world = build_cornell_box()

    camera = Camera(
        position=Point3(0, 2.75, 3.5),
        forward=Vec3(0, -0.05, -1),
        fov=45,
        width=W,
        height=H,
    )
    world.add_camera(camera)

    save_scene(world, "sample_scenes/cornell_box.json")

    image    = Image(W, H)
    viewport = GLViewport(W, H, "Roxy — Cornell Box")
    tracer   = TaichiRenderer(world, image, viewport, samples=6000, max_depth=12)

    if DEBUG:
        with cProfile.Profile() as pr:
            tracer.render()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
    else:
        tracer.render()

    while not viewport.should_close:
        viewport.poll_events()
        viewport.update(image)

    viewport.close()


if __name__ == "__main__":
    main()
