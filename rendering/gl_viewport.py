# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: OpenGL viewport using moderngl + pygame.
#              Displays ray-traced image buffers and interactive scene geometry.
# ============================================

from dataclasses import dataclass
import math

import numpy as np

from core import AABB, Mat4x4, Ray, Vec3
from scene.mesh import IndexedMesh, Mesh, Triangle

try:
    import pygame
    import moderngl
except ImportError:  # pragma: no cover - tested via importable pure helpers.
    pygame = None
    moderngl = None


IMAGE_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

IMAGE_FRAGMENT_SHADER = """
#version 330

uniform sampler2D texture0;
in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = texture(texture0, v_uv);
}
"""

SCENE_VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 v_normal;
out vec3 v_color;

void main() {
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

SCENE_FRAGMENT_SHADER = """
#version 330

in vec3 v_normal;
in vec3 v_color;

uniform vec3 light_direction;

out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(light_direction);
    float diffuse = max(dot(n, l), 0.0);
    vec3 color = v_color * (0.22 + diffuse * 0.78);
    out_color = vec4(color, 1.0);
}
"""

GRID_VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

GRID_FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
out vec4 out_color;

void main() {
    out_color = vec4(v_color, 1.0);
}
"""

QUAD_VERTICES = np.array([
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
     1.0,  1.0,  1.0, 0.0,
    -1.0, -1.0,  0.0, 1.0,
     1.0,  1.0,  1.0, 0.0,
    -1.0,  1.0,  0.0, 0.0,
], dtype="f4")

TRACKPAD_PAN_PIXELS = 56.0
TRACKPAD_ORBIT_PIXELS = 24.0
TRACKPAD_PINCH_DOLLY_PIXELS = 1800.0


@dataclass
class ObjectVertexSpan:
    scene_object: object
    start: int
    count: int
    bounds: AABB | None


@dataclass
class PickResult:
    scene_object: object
    shape: object
    t: float
    point: Vec3


@dataclass
class MoveGizmoDrag:
    scene_object: object
    axis_name: str
    axis: np.ndarray
    origin: np.ndarray
    size: float
    start_mouse: np.ndarray
    start_translation: Vec3
    start_matrix: Mat4x4 | None


@dataclass
class SceneViewportBuffers:
    vertices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    bounds: AABB | None
    triangle_count: int
    shape_count: int
    object_spans: tuple = ()

    @property
    def vertex_count(self):
        return int(len(self.vertices))

    @property
    def is_empty(self):
        return self.vertex_count == 0

    def span_for(self, scene_object):
        for span in self.object_spans:
            if span.scene_object is scene_object:
                return span
        return None


class ViewportCamera:
    """Small orbit camera for the interactive OpenGL scene viewport."""

    def __init__(
        self,
        target=None,
        distance=10.0,
        yaw=0.0,
        pitch=-20.0,
        fov=45.0,
        near=0.001,
        far=100000.0,
    ):
        self.target = _as_np3(target if target is not None else (0.0, 0.0, 0.0))
        self.distance = max(float(distance), 0.001)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self._clip_radius = max(self.distance, 1.0)

    @property
    def eye(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        cp = math.cos(pitch)
        offset = np.array([
            math.sin(yaw) * cp,
            math.sin(pitch),
            math.cos(yaw) * cp,
        ], dtype=np.float32)
        return self.target + offset * self.distance

    @property
    def forward(self):
        return _normalize(self.target - self.eye)

    @property
    def right(self):
        return _normalize(np.cross(self.forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))

    @property
    def up(self):
        return _normalize(np.cross(self.right, self.forward))

    def orbit(self, dx, dy, sensitivity=0.25):
        self.yaw -= float(dx) * sensitivity
        self.pitch = _clamp(self.pitch + float(dy) * sensitivity, -89.0, 89.0)

    def pan(self, dx, dy, width, height):
        if height <= 0:
            return
        world_per_pixel = (
            2.0 * self.distance * math.tan(math.radians(self.fov) * 0.5)
            / float(height)
        )
        self.target += self.right * (-float(dx) * world_per_pixel)
        self.target += self.up * (float(dy) * world_per_pixel)

    def dolly(self, delta, sensitivity=0.0015):
        self.distance *= math.exp(float(delta) * sensitivity)
        self.distance = max(self.distance, 0.001)
        self._update_clip_planes()

    def frame_bounds(self, bounds, padding=1.35):
        if bounds is None:
            return
        center = _vec3_to_np((bounds.min + bounds.max) * 0.5)
        extent = _vec3_to_np(bounds.max - bounds.min)
        radius = max(float(np.linalg.norm(extent) * 0.5), 0.001)
        self.target = center
        self.distance = max(
            radius * padding / math.tan(math.radians(self.fov) * 0.5),
            0.001,
        )
        self._update_clip_planes(radius)

    def _update_clip_planes(self, radius=None):
        if radius is not None:
            self._clip_radius = max(float(radius), 0.001)
        radius = max(self._clip_radius, 0.001)
        self.near = max(min(self.distance * 0.02, radius * 0.05), 0.001)
        self.far = max(
            self.distance + radius * 8.0,
            radius * 10.0,
            self.near + 100.0,
        )

    def view_matrix(self):
        eye = self.eye.astype(np.float32)
        target = self.target.astype(np.float32)
        forward = _normalize(target - eye)
        right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        up = np.cross(right, forward)

        return np.array([
            [right[0], right[1], right[2], -float(np.dot(right, eye))],
            [up[0], up[1], up[2], -float(np.dot(up, eye))],
            [-forward[0], -forward[1], -forward[2], float(np.dot(forward, eye))],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def projection_matrix(self, aspect):
        aspect = max(float(aspect), 1e-6)
        f = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        near = max(self.near, 0.001)
        far = max(self.far, near + 0.001)
        return np.array([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)

    def project_point(self, point, width, height):
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        point = _as_np3(point)
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        view_point = self.view_matrix() @ p
        clip = self.projection_matrix(width / height) @ view_point
        if abs(float(clip[3])) <= 1e-8:
            return None
        ndc = clip[:3] / clip[3]
        return np.array([
            (float(ndc[0]) + 1.0) * 0.5 * width,
            (1.0 - float(ndc[1])) * 0.5 * height,
            float(ndc[2]),
        ], dtype=np.float32)

    def apply_to_camera(self, camera):
        eye = self.eye
        camera.position = Vec3(float(eye[0]), float(eye[1]), float(eye[2]))
        fwd = self.forward
        up = self.up
        camera.forward = Vec3(float(fwd[0]), float(fwd[1]), float(fwd[2]))
        camera.up = Vec3(float(up[0]), float(up[1]), float(up[2]))
        camera.fov = self.fov

    def screen_ray(self, x, y, width, height):
        aspect = width / height if height else 1.0
        half_fov = math.tan(math.radians(self.fov) * 0.5)
        ndc_x = ((float(x) / float(width)) * 2.0 - 1.0) * aspect * half_fov
        ndc_y = (1.0 - (float(y) / float(height)) * 2.0) * half_fov
        origin = self.eye
        direction = self.forward + self.right * ndc_x + self.up * ndc_y
        return Ray(
            Vec3(float(origin[0]), float(origin[1]), float(origin[2])),
            Vec3(float(direction[0]), float(direction[1]), float(direction[2])),
        )


def build_scene_viewport_buffers(world, default_color=(0.62, 0.66, 0.70)):
    """Flatten renderable world geometry into GPU-friendly viewport arrays."""
    vertices = []
    normals = []
    colors = []
    object_spans = []
    bounds = None
    triangle_count = 0
    shape_count = 0
    vertex_cursor = 0

    for obj in _walk_objects(world.objects):
        if not getattr(obj, "visible", True):
            continue

        obj_vertices = []
        obj_normals = []
        obj_colors = []
        obj_bounds = None

        for shape in obj.shapes:
            geometry = shape.geometry
            extracted = _extract_shape_arrays(obj, shape, default_color)
            if extracted is None:
                continue

            shape_vertices, shape_normals, shape_colors = extracted
            if len(shape_vertices) == 0:
                continue

            obj_vertices.append(shape_vertices)
            obj_normals.append(shape_normals)
            obj_colors.append(shape_colors)
            triangle_count += len(shape_vertices) // 3
            shape_count += 1

            local_bounds = geometry.local_bounds()
            if local_bounds is not None:
                world_bounds = local_bounds.transform(obj.world_matrix)
                bounds = world_bounds if bounds is None else bounds.union(world_bounds)
                obj_bounds = world_bounds if obj_bounds is None else obj_bounds.union(world_bounds)

        if obj_vertices:
            combined_vertices = np.concatenate(obj_vertices, axis=0).astype(np.float32, copy=False)
            combined_normals = np.concatenate(obj_normals, axis=0).astype(np.float32, copy=False)
            combined_colors = np.concatenate(obj_colors, axis=0).astype(np.float32, copy=False)

            vertices.append(combined_vertices)
            normals.append(combined_normals)
            colors.append(combined_colors)
            object_spans.append(
                ObjectVertexSpan(
                    scene_object=obj,
                    start=vertex_cursor,
                    count=len(combined_vertices),
                    bounds=obj_bounds,
                )
            )
            vertex_cursor += len(combined_vertices)

    if not vertices:
        empty_vec3 = np.zeros((0, 3), dtype=np.float32)
        return SceneViewportBuffers(empty_vec3, empty_vec3, empty_vec3, None, 0, 0)

    return SceneViewportBuffers(
        np.concatenate(vertices, axis=0).astype(np.float32, copy=False),
        np.concatenate(normals, axis=0).astype(np.float32, copy=False),
        np.concatenate(colors, axis=0).astype(np.float32, copy=False),
        bounds,
        triangle_count,
        shape_count,
        tuple(object_spans),
    )


class GLViewport:
    """OpenGL window for path-traced images and direct scene previews."""

    def __init__(self, width, height, title="Roxy", world=None, sync_camera=False):
        if pygame is None or moderngl is None:
            raise RuntimeError("pygame and moderngl are required for GLViewport")

        self._width = int(width)
        self._height = int(height)
        self._texture_width = int(width)
        self._texture_height = int(height)
        self._should_close = False
        self._mode = "scene" if world is not None else "image"
        self._world = None
        self._sync_camera = bool(sync_camera)
        self._scene_buffers = None
        self._scene_vbo = None
        self._scene_vao = None
        self._selection_vbo = None
        self._selection_vao = None
        self._selection_vertex_count = 0
        self._grid_vbo = None
        self._grid_vao = None
        self._grid_vertex_count = 0
        self._gizmo_vbo = None
        self._gizmo_vao = None
        self._gizmo_vertex_count = 0
        self._drag_button = None
        self._mouse_down_pos = None
        self._mouse_dragged = False
        self._move_drag = None
        self._wireframe = False
        self._render_requested = False
        self._selected_object = None
        self._gizmo_mode = "select"
        self._camera = ViewportCamera()

        pygame.init()
        pygame.display.set_caption(title)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        pygame.display.set_mode(
            (self._width, self._height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )

        self._ctx = moderngl.create_context()
        self._ctx.viewport = (0, 0, self._width, self._height)
        self._ctx.enable(moderngl.DEPTH_TEST)

        self._init_image_resources()
        self._init_scene_resources()
        self._init_grid_resources()

        if world is not None:
            self.set_world(world)
        else:
            self._ctx.clear(0.0, 0.0, 0.0)

    @property
    def should_close(self):
        return self._should_close

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def camera(self):
        return self._camera

    @property
    def render_requested(self):
        return self._render_requested

    @property
    def selected_object(self):
        return self._selected_object

    @property
    def gizmo_mode(self):
        return self._gizmo_mode

    def consume_render_requested(self):
        requested = self._render_requested
        self._render_requested = False
        return requested

    def set_world(self, world, frame=True):
        self._world = world
        self._scene_buffers = build_scene_viewport_buffers(world)
        self._upload_scene_buffers()
        if frame and self._scene_buffers.bounds is not None:
            self._camera.frame_bounds(self._scene_buffers.bounds)
            self._sync_world_camera()
        self.set_selected_object(None)
        self._mode = "scene"

    def set_selected_object(self, scene_object):
        if scene_object is self._selected_object:
            return
        self._selected_object = scene_object
        self._upload_selection_buffers()
        self._upload_gizmo_buffers()

    def set_gizmo_mode(self, mode):
        if mode not in ("select", "move", "rotate", "scale"):
            raise ValueError("gizmo mode must be 'select', 'move', 'rotate', or 'scale'")
        if mode == self._gizmo_mode:
            return
        self._gizmo_mode = mode
        self._upload_gizmo_buffers()

    def pick_object(self, screen_x, screen_y):
        if self._world is None:
            return None
        result = pick_scene_object(
            self._world,
            self._camera,
            screen_x,
            screen_y,
            self._width,
            self._height,
        )
        self.set_selected_object(result.scene_object if result else None)
        return result

    def begin_move_gizmo_drag(self, screen_x, screen_y):
        if self._selected_object is None or self._gizmo_mode != "move":
            return False
        hit = pick_move_gizmo_axis(
            self._selected_object,
            self._camera,
            screen_x,
            screen_y,
            self._width,
            self._height,
        )
        if hit is None:
            return False

        axis_name, axis, origin, size = hit
        self._move_drag = MoveGizmoDrag(
            scene_object=self._selected_object,
            axis_name=axis_name,
            axis=axis,
            origin=origin,
            size=size,
            start_mouse=np.asarray([screen_x, screen_y], dtype=np.float32),
            start_translation=self._selected_object.translation,
            start_matrix=_copy_matrix(self._selected_object.local_matrix)
            if self._selected_object.matrix_mode
            else None,
        )
        return True

    def drag_move_gizmo(self, screen_x, screen_y):
        if self._move_drag is None:
            return False
        delta = move_gizmo_drag_delta(
            self._camera,
            self._move_drag.origin,
            self._move_drag.axis,
            self._move_drag.size,
            self._move_drag.start_mouse,
            np.asarray([screen_x, screen_y], dtype=np.float32),
            self._width,
            self._height,
        )
        _apply_world_translation(
            self._move_drag.scene_object,
            delta,
            start_translation=self._move_drag.start_translation,
            start_matrix=self._move_drag.start_matrix,
        )
        self._refresh_scene_geometry()
        return True

    def end_move_gizmo_drag(self):
        self._move_drag = None

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_close = True
            elif event.type == pygame.VIDEORESIZE:
                self._resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 2, 3):
                    self._drag_button = event.button
                    self._mouse_down_pos = event.pos
                    self._mouse_dragged = (
                        event.button == 1
                        and self.begin_move_gizmo_drag(*event.pos)
                    )
                elif event.button == 4:
                    self._camera.dolly(-120)
                    self._sync_world_camera()
                elif event.button == 5:
                    self._camera.dolly(120)
                    self._sync_world_camera()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == self._drag_button:
                    if event.button == 1 and not self._mouse_dragged and self._mode == "scene":
                        self.pick_object(*event.pos)
                    self._drag_button = None
                    self._mouse_down_pos = None
                    self.end_move_gizmo_drag()
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_wheel(event)
            elif (
                getattr(pygame, "MULTIGESTURE", None) is not None
                and event.type == pygame.MULTIGESTURE
            ):
                self._handle_multigesture(event)
            elif event.type == pygame.MOUSEMOTION and self._mode == "scene":
                if self._mouse_down_pos is not None:
                    dx = event.pos[0] - self._mouse_down_pos[0]
                    dy = event.pos[1] - self._mouse_down_pos[1]
                    if dx * dx + dy * dy > 16:
                        self._mouse_dragged = True
                if self._move_drag is not None:
                    self.drag_move_gizmo(*event.pos)
                else:
                    self._handle_mouse_drag(event.rel)

    def update(self, image):
        self._mode = "image"
        self._ensure_image_texture(image.width, image.height)
        self._texture.write(np.ascontiguousarray(image.pixels).tobytes())
        self._draw_image()

    def update_scanline(self, image, y):
        self._mode = "image"
        self._ensure_image_texture(image.width, image.height)
        row_data = np.ascontiguousarray(image.pixels[y:y + 1])
        self._texture.write(row_data.tobytes(), viewport=(0, y, image.width, 1))
        self._draw_image()

    def draw_scene(self):
        self._mode = "scene"
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.035, 0.038, 0.044, 1.0)
        self._write_scene_matrices(self._scene_program)
        self._write_scene_matrices(self._grid_program)

        if self._grid_vao is not None:
            self._grid_vao.render(moderngl.LINES, vertices=self._grid_vertex_count)

        if self._scene_vao is not None and self._scene_buffers is not None:
            self._scene_program["light_direction"].value = (0.35, 0.82, 0.44)
            self._ctx.wireframe = self._wireframe
            self._scene_vao.render(moderngl.TRIANGLES, vertices=self._scene_buffers.vertex_count)
            self._ctx.wireframe = False

        if self._selection_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._ctx.wireframe = True
            self._selection_vao.render(moderngl.TRIANGLES, vertices=self._selection_vertex_count)
            self._ctx.wireframe = False
            self._ctx.enable(moderngl.DEPTH_TEST)

        if self._gizmo_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._gizmo_vao.render(moderngl.LINES, vertices=self._gizmo_vertex_count)
            self._ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()

    def refresh(self):
        if self._mode == "scene":
            self.draw_scene()

    def close(self):
        for resource in (
            self._image_vao,
            self._image_vbo,
            self._texture,
            self._image_program,
            self._scene_vao,
            self._scene_vbo,
            self._scene_program,
            self._selection_vao,
            self._selection_vbo,
            self._grid_vao,
            self._grid_vbo,
            self._grid_program,
            self._gizmo_vao,
            self._gizmo_vbo,
        ):
            if resource is not None:
                resource.release()
        self._ctx.release()
        pygame.quit()

    def _init_image_resources(self):
        self._image_program = self._ctx.program(
            vertex_shader=IMAGE_VERTEX_SHADER,
            fragment_shader=IMAGE_FRAGMENT_SHADER,
        )
        self._image_vbo = self._ctx.buffer(QUAD_VERTICES.tobytes())
        self._image_vao = self._ctx.vertex_array(
            self._image_program,
            [(self._image_vbo, "2f 2f", "in_position", "in_uv")],
        )
        self._texture = self._ctx.texture(
            (self._texture_width, self._texture_height),
            3,
            dtype="f4",
        )
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.use(0)
        self._image_program["texture0"] = 0

    def _init_scene_resources(self):
        self._scene_program = self._ctx.program(
            vertex_shader=SCENE_VERTEX_SHADER,
            fragment_shader=SCENE_FRAGMENT_SHADER,
        )

    def _init_grid_resources(self):
        # Viewport-only DCC grid: never part of World, picking, or rendering.
        self._grid_program = self._ctx.program(
            vertex_shader=GRID_VERTEX_SHADER,
            fragment_shader=GRID_FRAGMENT_SHADER,
        )
        grid = _build_grid_vertices()
        self._grid_vertex_count = len(grid)
        self._grid_vbo = self._ctx.buffer(grid.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._grid_program,
            [(self._grid_vbo, "3f 3f", "in_position", "in_color")],
        )

    def _upload_scene_buffers(self):
        if self._scene_vao is not None:
            self._scene_vao.release()
            self._scene_vao = None
        if self._scene_vbo is not None:
            self._scene_vbo.release()
            self._scene_vbo = None

        if self._scene_buffers is None or self._scene_buffers.is_empty:
            return

        interleaved = np.concatenate(
            [
                self._scene_buffers.vertices,
                self._scene_buffers.normals,
                self._scene_buffers.colors,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        self._scene_vbo = self._ctx.buffer(interleaved.tobytes())
        self._scene_vao = self._ctx.vertex_array(
            self._scene_program,
            [(self._scene_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

    def _refresh_scene_geometry(self):
        if self._world is None:
            return
        self._scene_buffers = build_scene_viewport_buffers(self._world)
        self._upload_scene_buffers()
        self._upload_selection_buffers()
        self._upload_gizmo_buffers()

    def _upload_selection_buffers(self):
        if self._selection_vao is not None:
            self._selection_vao.release()
            self._selection_vao = None
        if self._selection_vbo is not None:
            self._selection_vbo.release()
            self._selection_vbo = None
        self._selection_vertex_count = 0

        if (
            self._scene_buffers is None
            or self._selected_object is None
            or self._scene_buffers.is_empty
        ):
            return

        span = self._scene_buffers.span_for(self._selected_object)
        if span is None or span.count <= 0:
            return

        start = span.start
        end = start + span.count
        vertices = self._scene_buffers.vertices[start:end]
        normals = self._scene_buffers.normals[start:end]
        colors = np.repeat(
            np.asarray([[1.0, 0.63, 0.12]], dtype=np.float32),
            span.count,
            axis=0,
        )
        interleaved = np.concatenate([vertices, normals, colors], axis=1).astype(
            np.float32,
            copy=False,
        )
        self._selection_vertex_count = span.count
        self._selection_vbo = self._ctx.buffer(interleaved.tobytes())
        self._selection_vao = self._ctx.vertex_array(
            self._scene_program,
            [(self._selection_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

    def _upload_gizmo_buffers(self):
        if self._gizmo_vao is not None:
            self._gizmo_vao.release()
            self._gizmo_vao = None
        if self._gizmo_vbo is not None:
            self._gizmo_vbo.release()
            self._gizmo_vbo = None
        self._gizmo_vertex_count = 0

        if self._selected_object is None or self._gizmo_mode == "select":
            return

        origin = _object_gizmo_origin(self._selected_object)
        size = _object_gizmo_size(self._selected_object)
        vertices = _build_gizmo_vertices(origin, size, self._gizmo_mode)
        if len(vertices) == 0:
            return

        self._gizmo_vertex_count = len(vertices)
        self._gizmo_vbo = self._ctx.buffer(vertices.tobytes())
        self._gizmo_vao = self._ctx.vertex_array(
            self._grid_program,
            [(self._gizmo_vbo, "3f 3f", "in_position", "in_color")],
        )

    def _draw_image(self):
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.0, 0.0, 0.0)
        self._image_vao.render(moderngl.TRIANGLES)
        pygame.display.flip()

    def _ensure_image_texture(self, width, height):
        width = int(width)
        height = int(height)
        if width == self._texture_width and height == self._texture_height:
            return
        self._texture.release()
        self._texture_width = width
        self._texture_height = height
        self._texture = self._ctx.texture((width, height), 3, dtype="f4")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.use(0)

    def _resize(self, width, height):
        self._width = max(int(width), 1)
        self._height = max(int(height), 1)
        self._ctx.viewport = (0, 0, self._width, self._height)

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self._should_close = True
        elif key == pygame.K_f and self._scene_buffers is not None:
            if self._selected_object is not None:
                span = self._scene_buffers.span_for(self._selected_object)
                self._camera.frame_bounds(span.bounds if span else self._scene_buffers.bounds)
            else:
                self._camera.frame_bounds(self._scene_buffers.bounds)
            self._sync_world_camera()
        elif key == pygame.K_q:
            self.set_gizmo_mode("select")
        elif key == pygame.K_w:
            self.set_gizmo_mode("move")
        elif key == pygame.K_e:
            self.set_gizmo_mode("rotate")
        elif key == pygame.K_r:
            self.set_gizmo_mode("scale")
        elif key == pygame.K_4:
            self._wireframe = not self._wireframe
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._render_requested = True

    def _handle_mouse_drag(self, rel):
        if self._drag_button is None:
            return
        dx, dy = rel
        if self._drag_button == 1:
            self._camera.orbit(dx, dy)
        elif self._drag_button == 2:
            self._camera.pan(dx, dy, self._width, self._height)
        elif self._drag_button == 3:
            self._camera.dolly(dy)
        self._sync_world_camera()

    def _handle_wheel(self, event):
        wheel_x, wheel_y = _wheel_delta(event)
        mods = pygame.key.get_mods()
        action = _scroll_wheel_view_action(
            wheel_x,
            wheel_y,
            shift=bool(mods & pygame.KMOD_SHIFT),
            ctrl=bool(mods & pygame.KMOD_CTRL),
            alt=bool(mods & pygame.KMOD_ALT),
            meta=bool(mods & pygame.KMOD_META),
        )
        self._apply_view_action(action)

    def _handle_multigesture(self, event):
        action = _pinch_view_action(getattr(event, "pinched", 0.0))
        self._apply_view_action(action)

    def _apply_view_action(self, action):
        moved = False
        if action[0] == "pan":
            self._camera.pan(action[1], action[2], self._width, self._height)
            moved = True
        elif action[0] == "dolly":
            self._camera.dolly(action[1])
            moved = True
        elif action[0] == "orbit":
            self._camera.orbit(action[1], action[2])
            moved = True
        if moved:
            self._sync_world_camera()

    def _write_scene_matrices(self, program):
        aspect = self._width / self._height if self._height else 1.0
        program["view"].write(_gl_matrix_bytes(self._camera.view_matrix()))
        program["projection"].write(_gl_matrix_bytes(self._camera.projection_matrix(aspect)))

    def _sync_world_camera(self):
        if not self._sync_camera or self._world is None or self._world.active_camera is None:
            return
        self._camera.apply_to_camera(self._world.active_camera)

    def __repr__(self):
        return f"GLViewport({self._width}x{self._height}, mode={self._mode!r})"


def pick_scene_object(world, viewport_camera, screen_x, screen_y, width, height):
    """Pick the nearest selectable scene object under a viewport pixel."""
    ray = viewport_camera.screen_ray(screen_x, screen_y, width, height)
    closest = None

    for obj in _walk_objects(world.objects):
        if not getattr(obj, "visible", True) or not getattr(obj, "selectable", True):
            continue
        result = _pick_object_shapes(obj, ray)
        if result is not None and (closest is None or result.t < closest.t):
            closest = result

    return closest


def _scroll_wheel_view_action(wheel_x, wheel_y, shift=False, ctrl=False, alt=False, meta=False):
    wheel_x = float(wheel_x)
    wheel_y = float(wheel_y)
    if alt:
        return (
            "orbit",
            -wheel_x * TRACKPAD_ORBIT_PIXELS,
            wheel_y * TRACKPAD_ORBIT_PIXELS,
        )
    if shift or ctrl or meta:
        return ("none",)
    if abs(wheel_x) <= 1e-8:
        return ("none",)
    return (
        "pan",
        -wheel_x * TRACKPAD_PAN_PIXELS,
        0.0,
    )


def _pinch_view_action(pinched):
    pinched = float(pinched)
    if abs(pinched) <= 1e-8:
        return ("none",)
    return ("dolly", -pinched * TRACKPAD_PINCH_DOLLY_PIXELS)


def _wheel_delta(event):
    return (
        float(getattr(event, "precise_x", getattr(event, "x", 0.0))),
        float(getattr(event, "precise_y", getattr(event, "y", 0.0))),
    )


def _pick_object_shapes(obj, world_ray):
    if not obj.shapes:
        return None

    if getattr(obj, "_can_use_aabb_early_out", None) and obj._can_use_aabb_early_out():
        if obj.world_aabb.intersect(world_ray) is None:
            return None

    inv = obj.world_inverse_matrix
    local_origin = inv.transform_point(world_ray.origin)
    local_direction = inv.transform_vector(world_ray.direction)
    local_ray = Ray(local_origin, local_direction)
    closest = None

    for shape in obj.shapes:
        hit = shape.intersect(local_ray)
        if hit is None:
            continue

        world_point = obj.world_matrix.transform_point(hit.point)
        world_t = (world_point - world_ray.origin).dot(world_ray.direction)
        if world_t <= 0.001:
            continue

        result = PickResult(obj, shape, world_t, world_point)
        if closest is None or result.t < closest.t:
            closest = result

    return closest


def pick_move_gizmo_axis(scene_object, viewport_camera, screen_x, screen_y, width, height, threshold=12.0):
    origin = _object_gizmo_origin(scene_object)
    size = _object_gizmo_size(scene_object)
    mouse = np.asarray([screen_x, screen_y], dtype=np.float32)
    best = None
    best_distance = float(threshold)

    for axis_name, axis in _gizmo_axes().items():
        start = viewport_camera.project_point(origin, width, height)
        end = viewport_camera.project_point(origin + axis * size, width, height)
        if start is None or end is None:
            continue
        distance = _screen_segment_distance(mouse, start[:2], end[:2])
        if distance <= best_distance:
            best_distance = distance
            best = (axis_name, axis, origin, size)

    return best


def pick_rotate_gizmo_axis(scene_object, viewport_camera, screen_x, screen_y, width, height, threshold=12.0):
    origin = _object_gizmo_origin(scene_object)
    size = _object_gizmo_size(scene_object)
    mouse = np.asarray([screen_x, screen_y], dtype=np.float32)
    best = None
    best_distance = float(threshold)

    for axis_name, axis in _gizmo_axes().items():
        ring = _ring_points(origin, axis, size, segments=64)
        projected = [viewport_camera.project_point(point, width, height) for point in ring]
        for i, start in enumerate(projected):
            end = projected[(i + 1) % len(projected)]
            if start is None or end is None:
                continue
            distance = _screen_segment_distance(mouse, start[:2], end[:2])
            if distance <= best_distance:
                best_distance = distance
                best = (axis_name, axis, origin, size)

    return best


def move_gizmo_drag_delta(viewport_camera, origin, axis, size, start_mouse, current_mouse, width, height):
    origin = _as_np3(origin)
    axis = _normalize(axis)
    start_mouse = np.asarray(start_mouse, dtype=np.float32)
    current_mouse = np.asarray(current_mouse, dtype=np.float32)
    start_screen = viewport_camera.project_point(origin, width, height)
    end_screen = viewport_camera.project_point(origin + axis * float(size), width, height)
    if start_screen is None or end_screen is None:
        return np.zeros(3, dtype=np.float32)

    screen_axis = end_screen[:2] - start_screen[:2]
    screen_length = float(np.linalg.norm(screen_axis))
    if screen_length <= 1e-5:
        return np.zeros(3, dtype=np.float32)

    screen_axis /= screen_length
    scalar_pixels = float(np.dot(current_mouse - start_mouse, screen_axis))
    world_units = scalar_pixels * (float(size) / screen_length)
    return axis * world_units


def scale_gizmo_drag_factor(viewport_camera, origin, axis, size, start_mouse, current_mouse, width, height):
    delta = move_gizmo_drag_delta(
        viewport_camera,
        origin,
        axis,
        size,
        start_mouse,
        current_mouse,
        width,
        height,
    )
    units = float(np.dot(delta, _normalize(axis)))
    return max(0.01, 1.0 + units / max(float(size), 1e-6))


def rotate_gizmo_drag_degrees(viewport_camera, origin, axis, start_mouse, current_mouse, width, height):
    origin = _as_np3(origin)
    axis = _normalize(axis)
    start_hit = _screen_ray_plane_hit(viewport_camera, start_mouse, width, height, origin, axis)
    current_hit = _screen_ray_plane_hit(viewport_camera, current_mouse, width, height, origin, axis)
    if start_hit is None or current_hit is None:
        return 0.0

    v0 = _normalize(start_hit - origin)
    v1 = _normalize(current_hit - origin)
    if float(np.linalg.norm(v0)) <= 1e-8 or float(np.linalg.norm(v1)) <= 1e-8:
        return 0.0
    signed = math.atan2(float(np.dot(axis, np.cross(v0, v1))), float(np.dot(v0, v1)))
    return math.degrees(signed)


def gizmo_axis_rotation_matrix(axis_name, degrees):
    if axis_name == "x":
        return Mat4x4.rotation_x(degrees)
    if axis_name == "y":
        return Mat4x4.rotation_y(degrees)
    if axis_name == "z":
        return Mat4x4.rotation_z(degrees)
    raise ValueError("axis_name must be 'x', 'y', or 'z'")


def gizmo_axis_scale_matrix(axis_name, factor):
    if axis_name == "x":
        return Mat4x4.scale(factor, 1.0, 1.0)
    if axis_name == "y":
        return Mat4x4.scale(1.0, factor, 1.0)
    if axis_name == "z":
        return Mat4x4.scale(1.0, 1.0, factor)
    raise ValueError("axis_name must be 'x', 'y', or 'z'")


def _apply_world_translation(scene_object, world_delta, start_translation=None, start_matrix=None):
    local_delta = _world_delta_to_parent_local(scene_object, world_delta)
    if start_matrix is not None:
        scene_object.local_matrix = (
            Mat4x4.translation(local_delta.x, local_delta.y, local_delta.z)
            * start_matrix
        )
        return

    base = start_translation if start_translation is not None else scene_object.translation
    scene_object.translation = base + local_delta


def _world_delta_to_parent_local(scene_object, world_delta):
    delta = _np_to_vec3(world_delta)
    if scene_object.parent is not None:
        return scene_object.parent.world_inverse_matrix.transform_vector(delta)
    return delta


def _copy_matrix(matrix):
    return Mat4x4([row[:] for row in matrix.rows])


def _gizmo_axes():
    return {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }


def _screen_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = _clamp(float(np.dot(point - start, segment) / length_sq), 0.0, 1.0)
    closest = start + segment * t
    return float(np.linalg.norm(point - closest))


def _screen_ray_plane_hit(viewport_camera, screen, width, height, plane_origin, plane_normal):
    screen = np.asarray(screen, dtype=np.float32)
    ray = viewport_camera.screen_ray(float(screen[0]), float(screen[1]), width, height)
    ray_origin = _vec3_to_np(ray.origin)
    ray_direction = _vec3_to_np(ray.direction)
    plane_origin = _as_np3(plane_origin)
    plane_normal = _normalize(plane_normal)
    denom = float(np.dot(ray_direction, plane_normal))
    if abs(denom) <= 1e-8:
        return None
    t = float(np.dot(plane_origin - ray_origin, plane_normal) / denom)
    if t <= 0.0:
        return None
    return ray_origin + ray_direction * t


def _extract_shape_arrays(obj, shape, default_color):
    geometry = shape.geometry
    if isinstance(geometry, IndexedMesh):
        arrays = geometry.indexed_triangle_arrays(
            matrix=obj.world_matrix,
            normal_matrix=obj.world_inverse_transpose_matrix,
            dtype=np.float32,
        )
        vertices = np.stack([arrays["v0"], arrays["v1"], arrays["v2"]], axis=1).reshape((-1, 3))
        normals = np.stack([arrays["n0"], arrays["n1"], arrays["n2"]], axis=1).reshape((-1, 3))
        tri_colors = _colors_for_groups(shape, arrays["groups"], arrays["group_idx"], default_color)
        colors = np.repeat(tri_colors, 3, axis=0)
        return vertices, normals, colors

    if isinstance(geometry, Mesh):
        return _extract_triangles(
            obj,
            shape,
            geometry._triangles,
            default_color,
        )

    if isinstance(geometry, Triangle):
        return _extract_triangles(obj, shape, [geometry], default_color)

    local_bounds = geometry.local_bounds()
    if local_bounds is None:
        return None
    return _extract_box_proxy(obj, shape, local_bounds, default_color)


def _extract_triangles(obj, shape, triangles, default_color):
    vertices = []
    normals = []
    colors = []
    matrix = obj.world_matrix
    normal_matrix = obj.world_inverse_transpose_matrix

    for tri in triangles:
        pts = [matrix.transform_point(p) for p in (tri._v0, tri._v1, tri._v2)]
        n = normal_matrix.transform_vector(tri._normal).normalize()
        color = _material_color(shape.material_for_group(tri.group), default_color)
        vertices.extend(_vec3_to_np(p) for p in pts)
        normals.extend(_vec3_to_np(n) for _ in range(3))
        colors.extend(color for _ in range(3))

    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        np.asarray(normals, dtype=np.float32).reshape((-1, 3)),
        np.asarray(colors, dtype=np.float32).reshape((-1, 3)),
    )


def _extract_box_proxy(obj, shape, bounds, default_color):
    mn, mx = bounds.min, bounds.max
    corners = [
        Vec3(mn.x, mn.y, mn.z),
        Vec3(mx.x, mn.y, mn.z),
        Vec3(mx.x, mx.y, mn.z),
        Vec3(mn.x, mx.y, mn.z),
        Vec3(mn.x, mn.y, mx.z),
        Vec3(mx.x, mn.y, mx.z),
        Vec3(mx.x, mx.y, mx.z),
        Vec3(mn.x, mx.y, mx.z),
    ]
    faces = [
        (0, 1, 2, 3, Vec3(0, 0, -1)),
        (5, 4, 7, 6, Vec3(0, 0, 1)),
        (4, 0, 3, 7, Vec3(-1, 0, 0)),
        (1, 5, 6, 2, Vec3(1, 0, 0)),
        (3, 2, 6, 7, Vec3(0, 1, 0)),
        (4, 5, 1, 0, Vec3(0, -1, 0)),
    ]
    matrix = obj.world_matrix
    normal_matrix = obj.world_inverse_transpose_matrix
    color = _material_color(shape.material_for_group("default"), default_color)
    vertices = []
    normals = []
    colors = []

    for a, b, c, d, normal in faces:
        transformed_normal = normal_matrix.transform_vector(normal).normalize()
        for idx in (a, b, c, a, c, d):
            vertices.append(_vec3_to_np(matrix.transform_point(corners[idx])))
            normals.append(_vec3_to_np(transformed_normal))
            colors.append(color)

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
    )


def _colors_for_groups(shape, group_names, group_indices, default_color):
    palette = np.asarray([
        _material_color(shape.material_for_group(group), default_color)
        for group in group_names
    ], dtype=np.float32)
    if len(palette) == 0:
        palette = np.asarray([default_color], dtype=np.float32)
    safe_indices = np.clip(group_indices.astype(np.int32, copy=False), 0, len(palette) - 1)
    return palette[safe_indices]


def _material_color(material, default_color):
    if material is None:
        return np.asarray(default_color, dtype=np.float32)
    albedo = getattr(material, "_albedo", None)
    if albedo is None:
        return np.asarray(default_color, dtype=np.float32)
    intensity = float(getattr(material, "_intensity", 1.0))
    scale = min(max(intensity, 1.0), 4.0)
    color = np.asarray([albedo.r, albedo.g, albedo.b], dtype=np.float32) * scale
    return np.clip(color, 0.0, 1.0)


def _walk_objects(objects):
    for obj in objects:
        yield obj
        for child in obj.children:
            yield from _walk_objects([child])


def _object_gizmo_origin(scene_object):
    pivot = getattr(scene_object, "pivot", Vec3(0, 0, 0))
    try:
        return _vec3_to_np(scene_object.world_matrix.transform_point(pivot))
    except Exception:
        return _vec3_to_np(scene_object.world_matrix.transform_point(Vec3(0, 0, 0)))


def _object_gizmo_size(scene_object):
    try:
        bounds = scene_object.world_aabb
    except Exception:
        bounds = None
    if bounds is None:
        return 1.0
    extent = _vec3_to_np(bounds.max - bounds.min)
    radius = float(np.linalg.norm(extent) * 0.5)
    return max(min(radius * 0.65, 10.0), 0.5)


def _build_gizmo_vertices(origin, size, mode):
    origin = _as_np3(origin)
    size = float(size)
    lines = []
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axes = (
        (x_axis, np.array([1.0, 0.18, 0.16], dtype=np.float32)),
        (y_axis, np.array([0.22, 0.86, 0.28], dtype=np.float32)),
        (z_axis, np.array([0.22, 0.42, 1.0], dtype=np.float32)),
    )

    if mode == "move":
        for axis, color in axes:
            end = origin + axis * size
            _append_line(lines, origin, end, color)
            _append_arrowhead(lines, end, axis, color, size * 0.14)
    elif mode == "scale":
        for axis, color in axes:
            end = origin + axis * size
            _append_line(lines, origin, end, color)
            _append_scale_handle(lines, end, axis, color, size * 0.11)
    elif mode == "rotate":
        _append_ring(lines, origin, x_axis, np.array([1.0, 0.18, 0.16], dtype=np.float32), size)
        _append_ring(lines, origin, y_axis, np.array([0.22, 0.86, 0.28], dtype=np.float32), size)
        _append_ring(lines, origin, z_axis, np.array([0.22, 0.42, 1.0], dtype=np.float32), size)
    else:
        raise ValueError("gizmo mode must be 'move', 'rotate', or 'scale'")

    return np.asarray(lines, dtype=np.float32).reshape((-1, 6))


def _append_line(lines, p0, p1, color):
    lines.append([*p0, *color])
    lines.append([*p1, *color])


def _append_arrowhead(lines, end, axis, color, size):
    basis_a, basis_b = _axis_perpendiculars(axis)
    back = end - axis * size
    for side in (basis_a, -basis_a, basis_b, -basis_b):
        _append_line(lines, end, back + side * size * 0.45, color)


def _append_scale_handle(lines, end, axis, color, size):
    basis_a, basis_b = _axis_perpendiculars(axis)
    corners = [
        end + basis_a * size + basis_b * size,
        end - basis_a * size + basis_b * size,
        end - basis_a * size - basis_b * size,
        end + basis_a * size - basis_b * size,
    ]
    for i in range(4):
        _append_line(lines, corners[i], corners[(i + 1) % 4], color)
    _append_line(lines, end, end + axis * size * 0.65, color)


def _append_ring(lines, origin, normal, color, radius, segments=64):
    points = _ring_points(origin, normal, radius, segments=segments)
    for i in range(segments):
        _append_line(lines, points[i], points[(i + 1) % segments], color)


def _ring_points(origin, normal, radius, segments=64):
    origin = _as_np3(origin)
    basis_a, basis_b = _axis_perpendiculars(normal)
    points = []
    for i in range(segments):
        angle = (i / segments) * math.tau
        points.append(origin + (basis_a * math.cos(angle) + basis_b * math.sin(angle)) * radius)
    return points


def _axis_perpendiculars(axis):
    axis = _normalize(axis)
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    basis_a = _normalize(np.cross(axis, helper))
    basis_b = _normalize(np.cross(axis, basis_a))
    return basis_a, basis_b


def _build_grid_vertices(size=10.0, divisions=20):
    lines = []
    color = np.array([0.23, 0.25, 0.28], dtype=np.float32)
    axis_x = np.array([0.45, 0.18, 0.18], dtype=np.float32)
    axis_z = np.array([0.18, 0.24, 0.45], dtype=np.float32)
    step = (size * 2.0) / divisions
    for i in range(divisions + 1):
        p = -size + i * step
        line_color = axis_z if abs(p) < 1e-6 else color
        lines.append([-size, 0.0, p, *line_color])
        lines.append([size, 0.0, p, *line_color])
        line_color = axis_x if abs(p) < 1e-6 else color
        lines.append([p, 0.0, -size, *line_color])
        lines.append([p, 0.0, size, *line_color])
    return np.asarray(lines, dtype=np.float32)


def _gl_matrix_bytes(matrix):
    return np.asarray(matrix, dtype=np.float32).T.tobytes()


def _as_np3(value):
    if isinstance(value, Vec3):
        return _vec3_to_np(value)
    return np.asarray(value, dtype=np.float32).reshape((3,))


def _vec3_to_np(value):
    return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def _np_to_vec3(value):
    value = _as_np3(value)
    return Vec3(float(value[0]), float(value[1]), float(value[2]))


def _normalize(value):
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(value / length, dtype=np.float32)


def _clamp(value, low, high):
    return max(low, min(high, value))
