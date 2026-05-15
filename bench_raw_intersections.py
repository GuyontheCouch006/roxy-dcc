import argparse
import time

import numpy as np

from core import Point3, Ray, Vec3
from rendering.intersector import (
    EmbreeIntersector,
    EmbreeUnavailableError,
    TriangleArrayIntersector,
    WorldIntersector,
)
from rendering.scene_arrays import flatten_world_triangles
from scene import Camera, World
from scene.io.obj_reader import OBJReader


def _parse_resolution(value):
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def _build_gallery(width, height):
    root = OBJReader.load("sample_scenes/gallery/gallery.obj", indexed=True)
    world = World(use_sky=False)
    world.add_object(root)
    world.add_camera(Camera(
        position=Point3(-0.6, 1.7, 9.0),
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


def _rays_from_arrays(origins, directions):
    return [
        Ray(Point3(*origin), Vec3(*direction))
        for origin, direction in zip(origins, directions)
    ]


def _make_intersector(kind, world, triangle_scene):
    if kind == "world":
        return WorldIntersector(world)
    if kind == "triangle-array":
        return TriangleArrayIntersector(triangle_scene)
    if kind == "embree":
        return EmbreeIntersector(triangle_scene=triangle_scene)
    raise ValueError(f"Unknown backend: {kind}")


def _use_raw_arrays(kind):
    return kind in ("triangle-array", "embree")


def _time_closest(intersector, rays, origins, directions, repeats, use_raw):
    best_hits = None
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        if use_raw:
            best_hits = intersector.intersect_raw_arrays(origins, directions)
        else:
            best_hits = intersector.intersect_many(rays)
        samples.append(time.perf_counter() - start)
    if use_raw:
        hit_mask = best_hits["hit"]
        checksum = float(np.sum(best_hits["t"][hit_mask])) if hit_mask.any() else 0.0
        return np.asarray(samples), int(np.count_nonzero(hit_mask)), checksum
    finite = [hit.t for hit in best_hits if hit is not None]
    checksum = float(np.sum(finite)) if finite else 0.0
    return np.asarray(samples), len(finite), checksum


def _time_occluded(intersector, rays, origins, directions, repeats, max_t, use_raw):
    blocked = None
    samples = []
    ray_count = len(origins) if use_raw else len(rays)
    max_ts = np.full(ray_count, max_t, dtype=np.float32)
    for _ in range(repeats):
        start = time.perf_counter()
        if use_raw:
            blocked = intersector.occluded_raw_arrays(origins, directions, max_ts)
        else:
            blocked = intersector.occluded_many(rays, max_ts)
        samples.append(time.perf_counter() - start)
    checksum = int(np.count_nonzero(blocked))
    return np.asarray(samples), checksum, float(checksum)


def _print_result(args, ray_count, samples, hit_count, checksum, triangle_scene):
    median = float(np.median(samples))
    mean = float(np.mean(samples))
    p95 = float(np.percentile(samples, 95))
    mrays = ray_count / median / 1_000_000 if median > 0 else 0.0
    print()
    print("Raw intersection benchmark")
    print(f"  scene:          {args.scene}")
    print(f"  backend:        {args.backend}")
    print(f"  query:          {args.query}")
    print(f"  resolution:     {args.resolution}")
    print(f"  rays:           {ray_count:,}")
    print(f"  triangles:      {triangle_scene.triangle_count:,}")
    print(f"  vertices:       {triangle_scene.vertex_count:,}")
    print(f"  materials:      {len(triangle_scene.materials):,}")
    print(f"  skipped prims:  {triangle_scene.skipped_primitives:,}")
    print(f"  hit/blocked:    {hit_count:,}")
    print(f"  checksum:       {checksum:.6f}")
    print(f"  median:         {median * 1000.0:.3f} ms")
    print(f"  mean:           {mean * 1000.0:.3f} ms")
    print(f"  p95:            {p95 * 1000.0:.3f} ms")
    print(f"  throughput:     {mrays:.3f} Mrays/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("gallery", "bicycle"), default="gallery")
    parser.add_argument(
        "--backend",
        choices=("world", "triangle-array", "embree"),
        default="triangle-array",
    )
    parser.add_argument("--query", choices=("closest", "occluded"), default="closest")
    parser.add_argument("--resolution", default="64x36")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-rays", type=int, default=4096)
    parser.add_argument("--max-t", type=float, default=1e9)
    args = parser.parse_args()

    width, height = _parse_resolution(args.resolution)
    start = time.perf_counter()
    world = _build_gallery(width, height) if args.scene == "gallery" else _build_bicycle(width, height)
    load_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    triangle_scene = flatten_world_triangles(world)
    flatten_elapsed = time.perf_counter() - start
    print(f"setup: load {load_elapsed:.3f}s, flatten {flatten_elapsed:.3f}s")

    origins, directions = _primary_rays(world.active_camera, width, height)
    if args.max_rays and len(origins) > args.max_rays:
        origins = origins[:args.max_rays]
        directions = directions[:args.max_rays]
    use_raw = _use_raw_arrays(args.backend)
    rays = _rays_from_arrays(origins, directions) if not use_raw else []

    try:
        intersector = _make_intersector(args.backend, world, triangle_scene)
    except EmbreeUnavailableError as exc:
        print(f"Embree unavailable: {exc}")
        return 2

    for _ in range(args.warmups):
        if args.query == "closest":
            if use_raw:
                intersector.intersect_raw_arrays(origins, directions)
            else:
                intersector.intersect_many(rays)
        else:
            max_ts = np.full(len(origins), args.max_t, dtype=np.float32)
            if use_raw:
                intersector.occluded_raw_arrays(origins, directions, max_ts)
            else:
                intersector.occluded_many(rays, max_ts)

    if args.query == "closest":
        samples, hit_count, checksum = _time_closest(
            intersector,
            rays,
            origins,
            directions,
            args.repeats,
            use_raw,
        )
    else:
        samples, hit_count, checksum = _time_occluded(
            intersector,
            rays,
            origins,
            directions,
            args.repeats,
            args.max_t,
            use_raw,
        )

    _print_result(args, len(origins), samples, hit_count, checksum, triangle_scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
