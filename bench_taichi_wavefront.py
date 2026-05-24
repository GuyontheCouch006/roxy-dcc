"""Benchmark Taichi mega-kernel vs wavefront renderer.

Examples:
    python bench_taichi_wavefront.py --scene gallery --resolution 320x180
    python bench_taichi_wavefront.py --scene bicycle --resolution 160x90 --samples 32
    python bench_taichi_wavefront.py --backend wavefront --scene gallery --resolution 640x360
"""

import argparse
import contextlib
import io
import os
import time
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TI_LOG_LEVEL", "error")


@dataclass
class BenchResult:
    backend: str
    load_seconds: float
    extract_seconds: float
    jit_seconds: float
    steady_seconds: float
    total_seconds: float
    rays_cast: int
    nonzero_pixels: int
    mean_luminance: float
    frames: int
    steady_frames: int

    @property
    def rays_per_second(self):
        if self.total_seconds <= 0.0:
            return 0.0
        return self.rays_cast / self.total_seconds

    @property
    def ms_per_frame(self):
        if self.steady_frames <= 0:
            return 0.0
        return self.steady_seconds / self.steady_frames * 1000


def _parse_resolution(value):
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must be WxH, e.g. 320x180") from exc


def _load_world(scene, width, height):
    from core import Color, Point3, Vec3
    from scene import Camera, Plane, SceneObject, Sphere, World
    from scene.io.obj_reader import OBJReader
    from scene.materials import Emissive
    from scene import Glossy

    start = time.perf_counter()
    if scene == "gallery":
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
    elif scene == "bicycle":
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
    else:
        raise ValueError(f"Unknown scene: {scene}")

    load_seconds = time.perf_counter() - start
    return world, load_seconds


def _run_taichi(renderer_cls, backend_name, world, width, height,
                load_seconds, samples, max_depth):
    from rendering.image import Image
    image = Image(width, height)

    renderer = renderer_cls(
        world, image, viewport=None,
        samples=samples,
        max_depth=max_depth,
        direct_light_mode="one",
        sample_clamp=10.0,
        direct_light_max_depth=1,
    )

    # Timed render — suppress all output
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        renderer.render()
    total_seconds = time.perf_counter() - t0

    stats = renderer.last_stats
    timing = getattr(renderer, "last_timing", {}) or {}

    extract_seconds = float(timing.get("extract_seconds", 0.0))
    jit_seconds = float(timing.get("jit_seconds", 0.0))
    steady_seconds = float(timing.get("steady_seconds", 0.0))
    steady_frames = int(timing.get("steady_frames", max(0, samples - 1)))
    measured_total = float(timing.get(
        "total_seconds",
        stats.elapsed_seconds if stats is not None else total_seconds,
    ))

    pixels = image.pixels
    nonzero = int(np.count_nonzero(np.any(pixels > 1e-6, axis=2)))
    mean_lum = float(np.mean(pixels))

    return BenchResult(
        backend=backend_name,
        load_seconds=load_seconds,
        extract_seconds=extract_seconds,
        jit_seconds=jit_seconds,
        steady_seconds=steady_seconds,
        total_seconds=measured_total,
        rays_cast=renderer.last_ray_count,
        nonzero_pixels=nonzero,
        mean_luminance=mean_lum,
        frames=stats.samples_rendered if stats is not None else samples,
        steady_frames=steady_frames,
    )


def _print_results(args, results):
    width, height = args.resolution
    print(f"Taichi wavefront benchmark  scene={args.scene}  "
          f"res={width}x{height}  samples={args.samples}  depth={args.max_depth}")
    print()
    hdr = (
        f"{'backend':<20} {'load':>7} {'extract':>8} {'jit':>7} "
        f"{'steady':>8} {'ms/frame':>10} {'total':>8} "
        f"{'rays_cast':>13} {'rays/s':>13} {'nonzero':>8} {'mean_lum':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.backend:<20} "
            f"{r.load_seconds:>6.2f}s "
            f"{r.extract_seconds:>7.2f}s "
            f"{r.jit_seconds:>6.2f}s "
            f"{r.steady_seconds:>7.2f}s "
            f"{r.ms_per_frame:>9.1f}ms "
            f"{r.total_seconds:>7.2f}s "
            f"{r.rays_cast:>13,} "
            f"{r.rays_per_second:>13,.0f} "
            f"{r.nonzero_pixels:>8,} "
            f"{r.mean_luminance:>9.4f}"
        )

    if len(results) == 2:
        baseline = results[0].steady_seconds or results[0].total_seconds
        candidate = results[1].steady_seconds or results[1].total_seconds
        speedup = baseline / candidate if candidate > 0.0 else 0.0
        print(f"\n  wavefront steady-state speedup vs mega-kernel: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("gallery", "bicycle"), default="gallery")
    parser.add_argument("--resolution", type=_parse_resolution, default=(320, 180))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("both", "mega", "wavefront"),
        default="both",
        help="'both' runs mega-kernel then wavefront (default)",
    )
    args = parser.parse_args()

    width, height = args.resolution

    print("Loading scene …")
    try:
        world, load_seconds = _load_world(args.scene, width, height)
    except FileNotFoundError as exc:
        print(f"Scene asset missing: {exc}")
        return 2

    print(f"Scene loaded in {load_seconds:.2f}s")
    print("Importing Taichi (this triggers JIT init) …")

    from rendering.taichi_renderer import TaichiRenderer
    from rendering.taichi_wavefront_renderer import TaichiWavefrontRenderer

    results = []
    selected = {"mega", "wavefront"} if args.backend == "both" else {args.backend}

    if "mega" in selected:
        print(f"\n[mega-kernel]  {width}x{height} @ {args.samples} samples …")
        r = _run_taichi(TaichiRenderer, "taichi-mega", world, width, height,
                        load_seconds, args.samples, args.max_depth)
        results.append(r)
        print(f"  done: total {r.total_seconds:.2f}s  steady {r.ms_per_frame:.1f} ms/frame  "
              f"{r.rays_per_second:,.0f} rays/s total")

    if "wavefront" in selected:
        print(f"\n[wavefront]  {width}x{height} @ {args.samples} samples …")
        r = _run_taichi(TaichiWavefrontRenderer, "taichi-wavefront", world, width, height,
                        load_seconds, args.samples, args.max_depth)
        results.append(r)
        print(f"  done: total {r.total_seconds:.2f}s  steady {r.ms_per_frame:.1f} ms/frame  "
              f"{r.rays_per_second:,.0f} rays/s total")

    print()
    _print_results(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
