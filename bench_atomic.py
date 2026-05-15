"""
Benchmark: compare per-frame time with and without the ray-count atomics.
Run with: python bench_atomic.py
"""
import math
import time

import taichi as ti

import core.timing as timing
from core import Vec3, Color, Point3
from scene import Sphere, SceneObject, World, Camera
from scene.materials import Emissive
from scene.io.obj_reader import OBJReader
from rendering import Image
from rendering.taichi import render_kernel, extract_scene
from rendering.taichi.fields import _frame_count, _ray_count

W, H = 480, 270
MAX_DEPTH = 5
DIRECT_LIGHT_MODE = 0   # "one"
DIRECT_LIGHT_MAX_DEPTH = 1
SAMPLE_CLAMP = 10.0
WARMUP_FRAMES = 5
BENCH_FRAMES = 40


def build_scene():
    root = OBJReader.load("sample_scenes/gallery/gallery.obj")
    world = World(use_sky=False)
    world.add_object(root)
    warm_white = Color(1.00, 0.96, 0.88)
    for z in range(-12, 10, 4):
        world.add_object(SceneObject(
            shape=Sphere(1.0),
            material=Emissive(warm_white, intensity=25.0),
            translation=Vec3(-0.6, 5.7, float(z)),
            scale=Vec3(0.25, 0.25, 0.25),
        ))
    world.add_camera(Camera(
        position=Point3(-0.9, 1.7, 9.0),
        forward=Vec3(0.0, -0.05, -1.0),
        fov=70, width=W, height=H,
    ))
    return world


print(f"Building scene and extracting to GPU...")
world = build_scene()
image = Image(W, H)

extract_scene(world)

cam = world.active_camera
fov_tan = math.tan(math.radians(cam.fov) / 2)
aspect  = cam.aspect_ratio
cam_pos   = ti.Vector(list(cam.position))
cam_fwd   = ti.Vector(list(cam.forward))
cam_right = ti.Vector(list(cam.right))
cam_up    = ti.Vector(list(cam.up))
use_sky   = int(world._use_sky)
bg_color  = list(world._background_color)


def run_bench(count_rays, n_frames):
    _frame_count[None] = 0
    times = []
    for i in range(n_frames):
        _frame_count[None] = i
        t0 = time.perf_counter()
        render_kernel(W, H, fov_tan, aspect, MAX_DEPTH, use_sky,
                      DIRECT_LIGHT_MODE, DIRECT_LIGHT_MAX_DEPTH,
                      SAMPLE_CLAMP, count_rays, bg_color,
                      cam_pos, cam_fwd, cam_right, cam_up)
        ti.sync()
        times.append(time.perf_counter() - t0)
    return times


print(f"JIT warm-up ({WARMUP_FRAMES} frames)...")
run_bench(1, WARMUP_FRAMES)

print(f"Benchmarking count_rays=1 ({BENCH_FRAMES} frames)...")
t_atomic = run_bench(1, BENCH_FRAMES)

print(f"Benchmarking count_rays=0 ({BENCH_FRAMES} frames)...")
t_no_atomic = run_bench(0, BENCH_FRAMES)

avg_atomic    = sum(t_atomic)    / len(t_atomic)    * 1000
avg_no_atomic = sum(t_no_atomic) / len(t_no_atomic) * 1000
speedup = avg_atomic / avg_no_atomic

print(f"\nResults — {W}x{H}, depth={MAX_DEPTH}, {BENCH_FRAMES} frames each")
print(f"  with atomics    (count_rays=1): {avg_atomic:.1f} ms/frame")
print(f"  without atomics (count_rays=0): {avg_no_atomic:.1f} ms/frame")
print(f"  difference: {avg_atomic - avg_no_atomic:+.1f} ms  ({speedup:.3f}x)")
