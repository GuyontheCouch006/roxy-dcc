"""Smoke benchmark Embree raw, preview, and wavefront render paths.

Examples:
    python bench_embree_wavefront.py --scene gallery --resolution 64x36
    python bench_embree_wavefront.py --scene bicycle --resolution 96x54 --backend all
"""

import argparse
import contextlib
import importlib
import inspect
import io
import os
import time
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TI_LOG_LEVEL", "error")

Camera = None
Color = None
EmbreeIntersector = None
EmbreePreviewRenderer = None
EmbreeUnavailableError = None
Emissive = None
Glossy = None
Image = None
OBJReader = None
Plane = None
Point3 = None
SceneObject = None
Sphere = None
Vec3 = None
World = None


@dataclass
class BenchResult:
    backend: str
    setup_seconds: float
    load_seconds: float
    build_seconds: float
    render_seconds: float
    rays_cast: int
    nonzero_pixels: int
    detail: str = ""

    @property
    def rays_per_second(self):
        if self.render_seconds <= 0.0:
            return 0.0
        return self.rays_cast / self.render_seconds


def _parse_resolution(value):
    try:
        width, height = value.lower().split("x", 1)
        width = int(width)
        height = int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must look like 64x36") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def _load_repo_symbols(quiet=True):
    stream = io.StringIO()
    stdout_context = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    stderr_context = contextlib.redirect_stderr(stream) if quiet else contextlib.nullcontext()
    with stdout_context, stderr_context:
        from core import Color as CoreColor
        from core import Point3 as CorePoint3
        from core import Vec3 as CoreVec3
        from rendering.embree_preview_renderer import (
            EmbreePreviewRenderer as PreviewRenderer,
        )
        from rendering.image import Image as RenderImage
        from rendering.intersector import (
            EmbreeIntersector as Intersector,
            EmbreeUnavailableError as UnavailableError,
        )
        from scene import Camera as SceneCamera
        from scene import Glossy as SceneGlossy
        from scene import Plane as ScenePlane
        from scene import SceneObject as SceneObjectClass
        from scene import Sphere as SceneSphere
        from scene import World as SceneWorld
        from scene.io.obj_reader import OBJReader as Reader
        from scene.materials import Emissive as EmissiveMaterial

    globals().update({
        "Camera": SceneCamera,
        "Color": CoreColor,
        "EmbreeIntersector": Intersector,
        "EmbreePreviewRenderer": PreviewRenderer,
        "EmbreeUnavailableError": UnavailableError,
        "Emissive": EmissiveMaterial,
        "Glossy": SceneGlossy,
        "Image": RenderImage,
        "OBJReader": Reader,
        "Plane": ScenePlane,
        "Point3": CorePoint3,
        "SceneObject": SceneObjectClass,
        "Sphere": SceneSphere,
        "Vec3": CoreVec3,
        "World": SceneWorld,
    })


def _build_gallery(width, height):
    root = OBJReader.load("sample_scenes/gallery/gallery.obj", indexed=True)
    world = World(use_sky=False)
    world.add_object(root)

    warm_white = Color(1.00, 0.96, 0.88)
    for z in range(-12, 10, 4):
        world.add_object(SceneObject(
            shape=Sphere(1.0),
            material=Emissive(warm_white, intensity=25.0),
            translation=Vec3(-0.6, 5.7, float(z)),
            scale=Vec3(0.25, 0.25, 0.25),
            name=f"ceiling_light_{z}",
        ))

    world.add_camera(Camera(
        position=Point3(-0.9, 1.7, 9.0),
        forward=Vec3(0.0, -0.05, -1.0),
        fov=70,
        width=width,
        height=height,
    ))
    return world


def _build_bicycle(width, height):
    root = OBJReader.load("sample_scenes/roadBike/roadBike.obj", indexed=True)
    root.rotation = Vec3(0, -10, 0)

    world = World(use_sky=True)
    world.add_object(root)
    world.add_object(SceneObject(
        shape=Plane(normal=Vec3(0, 1, 0)),
        material=Glossy(Color(0.45, 0.45, 0.45), roughness=0.05),
        name="ground",
    ))
    world.add_object(SceneObject(
        shape=Sphere(1.0),
        material=Emissive(Color(1.00, 0.92, 0.76), intensity=50.0),
        translation=Vec3(-3.0, 4.5, 4.0),
        scale=Vec3(0.6, 0.6, 0.6),
        name="key_light",
    ))
    world.add_object(SceneObject(
        shape=Sphere(1.0),
        material=Emissive(Color(0.72, 0.84, 1.00), intensity=20.0),
        translation=Vec3(4.0, 2.5, 3.0),
        scale=Vec3(0.5, 0.5, 0.5),
        name="fill_light",
    ))
    world.add_object(SceneObject(
        shape=Sphere(1.0),
        material=Emissive(Color(0.88, 0.72, 1.00), intensity=30.0),
        translation=Vec3(0.5, 5.0, -4.0),
        scale=Vec3(0.4, 0.4, 0.4),
        name="rim_light",
    ))
    world.add_camera(Camera(
        position=Point3(1.2, 0.8, 2.0),
        forward=Vec3(-2.0, -0.5, -3.1),
        fov=50,
        width=width,
        height=height,
    ))
    return world


def _primary_rays(camera, width, height):
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    u = xs / width * 2.0 - 1.0
    v = 1.0 - ys / height * 2.0
    half_fov = np.tan(np.radians(camera.fov) / 2.0)
    ndc_x = u * camera.aspect_ratio * half_fov
    ndc_y = v * half_fov

    forward = np.asarray(list(camera.forward), dtype=np.float32)
    right = np.asarray(list(camera.right), dtype=np.float32)
    up = np.asarray(list(camera.up), dtype=np.float32)
    directions = (
        forward[None, None, :]
        + right[None, None, :] * ndc_x[:, :, None]
        + up[None, None, :] * ndc_y[:, :, None]
    )
    directions = directions.reshape((-1, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    origins = np.repeat(
        np.asarray([list(camera.position)], dtype=np.float32),
        len(directions),
        axis=0,
    )
    return origins, directions.astype(np.float32, copy=False)


def _load_world(scene, width, height):
    start = time.perf_counter()
    if scene == "gallery":
        world = _build_gallery(width, height)
    elif scene == "bicycle":
        world = _build_bicycle(width, height)
    else:
        raise ValueError(f"Unknown scene: {scene}")
    return world, time.perf_counter() - start


def _build_intersector(world):
    start = time.perf_counter()
    intersector = EmbreeIntersector(world)
    return intersector, time.perf_counter() - start


def _run_raw(world, width, height, load_seconds):
    setup_start = time.perf_counter()
    origins, directions = _primary_rays(world.active_camera, width, height)
    setup_seconds = time.perf_counter() - setup_start
    intersector, build_seconds = _build_intersector(world)

    render_start = time.perf_counter()
    raw = intersector.intersect_raw_arrays(origins, directions)
    render_seconds = time.perf_counter() - render_start
    return BenchResult(
        backend="embree-raw",
        setup_seconds=setup_seconds,
        load_seconds=load_seconds,
        build_seconds=build_seconds,
        render_seconds=render_seconds,
        rays_cast=len(origins),
        nonzero_pixels=int(np.count_nonzero(raw["hit"])),
        detail=f"{intersector.triangle_count:,} tris via {intersector.backend_name}",
    )


def _run_renderer(backend, renderer_cls, world, width, height, load_seconds,
                  quiet=True):
    setup_start = time.perf_counter()
    image = Image(width, height)
    setup_seconds = time.perf_counter() - setup_start
    intersector, build_seconds = _build_intersector(world)
    renderer = _instantiate_renderer(renderer_cls, world, image, intersector)
    if hasattr(renderer, "preload_textures"):
        preload_start = time.perf_counter()
        renderer.preload_textures()
        setup_seconds += time.perf_counter() - preload_start

    render_start = time.perf_counter()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            renderer.render()
    else:
        renderer.render()
    render_seconds = time.perf_counter() - render_start

    stats = getattr(renderer, "last_stats", None)
    rays_cast = int(getattr(stats, "rays_cast", 0) or getattr(renderer, "last_ray_count", 0))
    nonzero_pixels = int(np.count_nonzero(np.any(image.pixels > 1e-6, axis=2)))
    return BenchResult(
        backend=backend,
        setup_seconds=setup_seconds,
        load_seconds=load_seconds,
        build_seconds=build_seconds,
        render_seconds=render_seconds,
        rays_cast=rays_cast,
        nonzero_pixels=nonzero_pixels,
        detail=f"{intersector.triangle_count:,} tris via {intersector.backend_name}",
    )


def _instantiate_renderer(renderer_cls, world, image, intersector):
    signature = inspect.signature(renderer_cls)
    kwargs = {}
    if "intersector" in signature.parameters:
        kwargs["intersector"] = intersector

    attempts = (
        (world, image, None),
        (world, image),
    )
    errors = []
    for args in attempts:
        try:
            return renderer_cls(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
    raise TypeError(
        f"Could not instantiate {renderer_cls.__name__}; tried viewport=None "
        f"and headless signatures. Last errors: {'; '.join(errors)}"
    )


def _load_wavefront_renderer():
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = importlib.import_module("rendering.embree_wavefront_renderer")
    except ModuleNotFoundError as exc:
        if exc.name == "rendering.embree_wavefront_renderer":
            return None, (
                "Embree wavefront renderer is not present yet. Expected "
                "rendering.embree_wavefront_renderer.EmbreeWavefrontRenderer."
            )
        raise

    renderer_cls = getattr(module, "EmbreeWavefrontRenderer", None)
    if renderer_cls is None:
        return None, (
            "Embree wavefront renderer module loaded, but class "
            "EmbreeWavefrontRenderer was not found."
        )
    return renderer_cls, None


def _print_results(args, results, skipped):
    print("Embree renderer smoke benchmark")
    print(f"  scene:       {args.scene}")
    print(f"  resolution:  {args.resolution[0]}x{args.resolution[1]}")
    print()
    print(
        "backend             setup     load    build   render    rays_cast"
        "       rays/s  nonzero_pixels  detail"
    )
    print("-" * 115)
    for result in results:
        print(
            f"{result.backend:<17}"
            f"{result.setup_seconds:>8.3f}s"
            f"{result.load_seconds:>8.3f}s"
            f"{result.build_seconds:>8.3f}s"
            f"{result.render_seconds:>8.3f}s"
            f"{result.rays_cast:>13,}"
            f"{result.rays_per_second:>13,.0f}"
            f"{result.nonzero_pixels:>16,}  "
            f"{result.detail}"
        )
    for name, reason in skipped:
        print(f"\nskipped {name}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("gallery", "bicycle"), default="gallery")
    parser.add_argument("--resolution", type=_parse_resolution, default=(64, 36))
    parser.add_argument(
        "--backend",
        choices=("all", "raw", "preview", "wavefront"),
        default="all",
        help="backend to run; all runs raw, preview, and wavefront if present",
    )
    parser.add_argument(
        "--show-render-output",
        action="store_true",
        help="let renderer-specific reports print during timed render calls",
    )
    args = parser.parse_args()

    _load_repo_symbols(quiet=not args.show_render_output)

    width, height = args.resolution
    try:
        world, load_seconds = _load_world(args.scene, width, height)
    except FileNotFoundError as exc:
        print(f"Scene asset missing: {exc}")
        return 2

    selected = {"raw", "preview", "wavefront"} if args.backend == "all" else {args.backend}
    results = []
    skipped = []

    try:
        if "raw" in selected:
            results.append(_run_raw(world, width, height, load_seconds))
        if "preview" in selected:
            results.append(_run_renderer(
                "embree-preview",
                EmbreePreviewRenderer,
                world,
                width,
                height,
                load_seconds,
                quiet=not args.show_render_output,
            ))
        if "wavefront" in selected:
            renderer_cls, reason = _load_wavefront_renderer()
            if renderer_cls is None:
                skipped.append(("embree-wavefront", reason))
            else:
                results.append(_run_renderer(
                    "embree-wavefront",
                    renderer_cls,
                    world,
                    width,
                    height,
                    load_seconds,
                    quiet=not args.show_render_output,
                ))
    except EmbreeUnavailableError as exc:
        print(f"Embree unavailable: {exc}")
        return 2

    _print_results(args, results, skipped)
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
