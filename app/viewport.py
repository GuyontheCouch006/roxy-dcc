from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets

import moderngl
from rendering.gl_viewport import (
    GRID_FRAGMENT_SHADER,
    GRID_VERTEX_SHADER,
    SCENE_FRAGMENT_SHADER,
    SCENE_VERTEX_SHADER,
    MoveGizmoDrag,
    ViewportCamera,
    _apply_world_translation,
    _build_gizmo_vertices,
    _build_grid_vertices,
    _copy_matrix,
    _gl_matrix_bytes,
    _object_gizmo_origin,
    _object_gizmo_size,
    _pinch_view_action,
    _scroll_wheel_view_action,
    build_scene_viewport_buffers,
    move_gizmo_drag_delta,
    pick_move_gizmo_axis,
    pick_scene_object,
)
from scene import SceneObject


class QtGLViewport(QtOpenGLWidgets.QOpenGLWidget):
    objectSelected = QtCore.Signal(object)
    renderRequested = QtCore.Signal()
    sceneChanged = QtCore.Signal()

    def __init__(self, world=None, sync_camera=False, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 280)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        surface_format = QtGui.QSurfaceFormat()
        surface_format.setVersion(3, 3)
        surface_format.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface_format.setDepthBufferSize(24)
        self.setFormat(surface_format)

        self._world = None
        self._sync_camera = bool(sync_camera)
        self._camera = ViewportCamera()
        self._scene_buffers = None
        self._selected_object = None
        self._gizmo_mode = "select"
        self._wireframe = False

        self._ctx = None
        self._qt_framebuffer = None
        self._qt_framebuffer_id = None
        self._scene_program = None
        self._grid_program = None
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

        self._drag_button = QtCore.Qt.MouseButton.NoButton
        self._mouse_down_pos = None
        self._last_mouse_pos = None
        self._mouse_dragged = False
        self._move_drag = None

        if world is not None:
            self.set_world(world)

    @property
    def world(self):
        return self._world

    @property
    def camera(self):
        return self._camera

    @property
    def selected_object(self):
        return self._selected_object

    @property
    def gizmo_mode(self):
        return self._gizmo_mode

    @property
    def scene_buffers(self):
        return self._scene_buffers

    def set_world(self, world, frame=True):
        self._world = world
        self._scene_buffers = (
            build_scene_viewport_buffers(world) if world is not None else None
        )
        if (
            frame
            and self._scene_buffers is not None
            and self._scene_buffers.bounds is not None
        ):
            self._camera.frame_bounds(self._scene_buffers.bounds)
            self._sync_world_camera()
        self.set_selected_object(None)
        self._upload_all_if_ready()
        self.update()

    def refresh_scene_geometry(self):
        if self._world is None:
            return
        self._scene_buffers = build_scene_viewport_buffers(self._world)
        self._upload_all_if_ready()
        self.sceneChanged.emit()
        self.update()

    def set_selected_object(self, scene_object, emit=False):
        if scene_object is not None and not isinstance(scene_object, SceneObject):
            scene_object = None
        if scene_object is self._selected_object:
            return
        self._selected_object = scene_object
        self._upload_selection_if_ready()
        self._upload_gizmo_if_ready()
        self.update()
        if emit:
            self.objectSelected.emit(scene_object)

    def set_gizmo_mode(self, mode):
        if mode not in ("select", "move", "rotate", "scale"):
            raise ValueError("gizmo mode must be 'select', 'move', 'rotate', or 'scale'")
        if mode == self._gizmo_mode:
            return
        self._gizmo_mode = mode
        self._upload_gizmo_if_ready()
        self.update()

    def frame_selection_or_scene(self):
        if self._scene_buffers is None:
            return
        bounds = self._scene_buffers.bounds
        if self._selected_object is not None:
            span = self._scene_buffers.span_for(self._selected_object)
            if span is not None and span.bounds is not None:
                bounds = span.bounds
        if bounds is not None:
            self._camera.frame_bounds(bounds)
            self._sync_world_camera()
            self.update()

    def pick_object(self, screen_x, screen_y):
        if self._world is None:
            self.set_selected_object(None, emit=True)
            return None
        result = pick_scene_object(
            self._world,
            self._camera,
            screen_x,
            screen_y,
            max(self.width(), 1),
            max(self.height(), 1),
        )
        self.set_selected_object(result.scene_object if result else None, emit=True)
        return result

    def begin_move_gizmo_drag(self, screen_x, screen_y):
        if self._selected_object is None or self._gizmo_mode != "move":
            return False
        hit = pick_move_gizmo_axis(
            self._selected_object,
            self._camera,
            screen_x,
            screen_y,
            max(self.width(), 1),
            max(self.height(), 1),
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
            max(self.width(), 1),
            max(self.height(), 1),
        )
        _apply_world_translation(
            self._move_drag.scene_object,
            delta,
            start_translation=self._move_drag.start_translation,
            start_matrix=self._move_drag.start_matrix,
        )
        self.refresh_scene_geometry()
        return True

    def end_move_gizmo_drag(self):
        self._move_drag = None

    def initializeGL(self):
        self._ctx = moderngl.create_context(require=330)
        self._use_qt_framebuffer()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._scene_program = self._ctx.program(
            vertex_shader=SCENE_VERTEX_SHADER,
            fragment_shader=SCENE_FRAGMENT_SHADER,
        )
        self._grid_program = self._ctx.program(
            vertex_shader=GRID_VERTEX_SHADER,
            fragment_shader=GRID_FRAGMENT_SHADER,
        )
        self._upload_grid_buffers()
        self._upload_scene_buffers()
        self._upload_selection_buffers()
        self._upload_gizmo_buffers()

    def resizeGL(self, width, height):
        if self._ctx is None:
            return
        self._qt_framebuffer = None
        self._qt_framebuffer_id = None
        self._use_qt_framebuffer(max(int(width), 1), max(int(height), 1))

    def paintGL(self):
        if self._ctx is None:
            return
        target = self._use_qt_framebuffer(max(self.width(), 1), max(self.height(), 1))
        if target is None:
            return
        self._ctx.enable(moderngl.DEPTH_TEST)
        target.clear(0.035, 0.038, 0.044, 1.0)
        self._write_scene_matrices(self._scene_program)
        self._write_scene_matrices(self._grid_program)

        if self._grid_vao is not None:
            self._grid_vao.render(moderngl.LINES, vertices=self._grid_vertex_count)

        if self._scene_vao is not None and self._scene_buffers is not None:
            self._scene_program["light_direction"].value = (0.35, 0.82, 0.44)
            self._ctx.wireframe = self._wireframe
            self._scene_vao.render(
                moderngl.TRIANGLES,
                vertices=self._scene_buffers.vertex_count,
            )
            self._ctx.wireframe = False

        if self._selection_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._ctx.wireframe = True
            self._selection_vao.render(
                moderngl.TRIANGLES,
                vertices=self._selection_vertex_count,
            )
            self._ctx.wireframe = False
            self._ctx.enable(moderngl.DEPTH_TEST)

        if self._gizmo_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._gizmo_vao.render(moderngl.LINES, vertices=self._gizmo_vertex_count)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            pos = event.position()
            self._drag_button = event.button()
            self._mouse_down_pos = pos
            self._last_mouse_pos = pos
            self._mouse_dragged = (
                event.button() == QtCore.Qt.MouseButton.LeftButton
                and self.begin_move_gizmo_drag(pos.x(), pos.y())
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_button == QtCore.Qt.MouseButton.NoButton:
            super().mouseMoveEvent(event)
            return

        pos = event.position()
        last = self._last_mouse_pos or pos
        rel_x = pos.x() - last.x()
        rel_y = pos.y() - last.y()
        self._last_mouse_pos = pos

        if self._mouse_down_pos is not None:
            dx = pos.x() - self._mouse_down_pos.x()
            dy = pos.y() - self._mouse_down_pos.y()
            if dx * dx + dy * dy > 16.0:
                self._mouse_dragged = True

        if self._move_drag is not None:
            self.drag_move_gizmo(pos.x(), pos.y())
        else:
            self._handle_mouse_drag(rel_x, rel_y)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == self._drag_button:
            pos = event.position()
            if (
                event.button() == QtCore.Qt.MouseButton.LeftButton
                and not self._mouse_dragged
            ):
                self.pick_object(pos.x(), pos.y())
            self._drag_button = QtCore.Qt.MouseButton.NoButton
            self._mouse_down_pos = None
            self._last_mouse_pos = None
            self._mouse_dragged = False
            self.end_move_gizmo_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        wheel_x, wheel_y = _qt_wheel_delta(event)
        modifiers = event.modifiers()
        action = _scroll_wheel_view_action(
            wheel_x,
            wheel_y,
            shift=bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier),
            ctrl=bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier),
            alt=bool(modifiers & QtCore.Qt.KeyboardModifier.AltModifier),
            meta=bool(modifiers & QtCore.Qt.KeyboardModifier.MetaModifier),
        )
        if self._apply_view_action(action):
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_F:
            self.frame_selection_or_scene()
        elif key == QtCore.Qt.Key.Key_Q:
            self.set_gizmo_mode("select")
        elif key == QtCore.Qt.Key.Key_W:
            self.set_gizmo_mode("move")
        elif key == QtCore.Qt.Key.Key_E:
            self.set_gizmo_mode("rotate")
        elif key == QtCore.Qt.Key.Key_R:
            self.set_gizmo_mode("scale")
        elif key == QtCore.Qt.Key.Key_4:
            self._wireframe = not self._wireframe
            self.update()
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.renderRequested.emit()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def event(self, event):
        if event.type() == QtCore.QEvent.Type.NativeGesture:
            gesture_type = event.gestureType()
            if gesture_type == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                if self._apply_view_action(_pinch_view_action(event.value())):
                    event.accept()
                    return True
        return super().event(event)

    def closeEvent(self, event):
        self._release_gl_resources()
        super().closeEvent(event)

    def _handle_mouse_drag(self, dx, dy):
        moved = False
        if self._drag_button == QtCore.Qt.MouseButton.LeftButton:
            self._camera.orbit(dx, dy)
            moved = True
        elif self._drag_button == QtCore.Qt.MouseButton.MiddleButton:
            self._camera.pan(dx, dy, max(self.width(), 1), max(self.height(), 1))
            moved = True
        elif self._drag_button == QtCore.Qt.MouseButton.RightButton:
            self._camera.dolly(dy)
            moved = True
        if moved:
            self._sync_world_camera()
            self.update()

    def _apply_view_action(self, action):
        moved = False
        if action[0] == "pan":
            self._camera.pan(
                action[1],
                action[2],
                max(self.width(), 1),
                max(self.height(), 1),
            )
            moved = True
        elif action[0] == "dolly":
            self._camera.dolly(action[1])
            moved = True
        elif action[0] == "orbit":
            self._camera.orbit(action[1], action[2])
            moved = True
        if moved:
            self._sync_world_camera()
            self.update()
        return moved

    def _sync_world_camera(self):
        if not self._sync_camera or self._world is None or self._world.active_camera is None:
            return
        self._camera.apply_to_camera(self._world.active_camera)

    def _write_scene_matrices(self, program):
        aspect = self.width() / self.height() if self.height() else 1.0
        program["view"].write(_gl_matrix_bytes(self._camera.view_matrix()))
        program["projection"].write(_gl_matrix_bytes(self._camera.projection_matrix(aspect)))

    def _use_qt_framebuffer(self, width=None, height=None):
        if self._ctx is None:
            return None
        framebuffer_id = int(self.defaultFramebufferObject())
        if (
            self._qt_framebuffer is None
            or self._qt_framebuffer_id != framebuffer_id
        ):
            self._qt_framebuffer = self._ctx.detect_framebuffer(framebuffer_id)
            self._qt_framebuffer_id = framebuffer_id

        width = max(int(width if width is not None else self.width()), 1)
        height = max(int(height if height is not None else self.height()), 1)
        viewport = (0, 0, width, height)
        self._qt_framebuffer.use()
        self._qt_framebuffer.viewport = viewport
        self._ctx.viewport = viewport
        return self._qt_framebuffer

    def _upload_all_if_ready(self):
        if self._ctx is None:
            return
        self.makeCurrent()
        try:
            self._upload_scene_buffers()
            self._upload_selection_buffers()
            self._upload_gizmo_buffers()
        finally:
            self.doneCurrent()

    def _upload_selection_if_ready(self):
        if self._ctx is None:
            return
        self.makeCurrent()
        try:
            self._upload_selection_buffers()
        finally:
            self.doneCurrent()

    def _upload_gizmo_if_ready(self):
        if self._ctx is None:
            return
        self.makeCurrent()
        try:
            self._upload_gizmo_buffers()
        finally:
            self.doneCurrent()

    def _upload_grid_buffers(self):
        self._release(self._grid_vao)
        self._release(self._grid_vbo)
        self._grid_vao = None
        self._grid_vbo = None
        self._grid_vertex_count = 0

        grid = _build_grid_vertices()
        self._grid_vertex_count = len(grid)
        self._grid_vbo = self._ctx.buffer(grid.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._grid_program,
            [(self._grid_vbo, "3f 3f", "in_position", "in_color")],
        )

    def _upload_scene_buffers(self):
        self._release(self._scene_vao)
        self._release(self._scene_vbo)
        self._scene_vao = None
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

    def _upload_selection_buffers(self):
        self._release(self._selection_vao)
        self._release(self._selection_vbo)
        self._selection_vao = None
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
        self._release(self._gizmo_vao)
        self._release(self._gizmo_vbo)
        self._gizmo_vao = None
        self._gizmo_vbo = None
        self._gizmo_vertex_count = 0

        if (
            self._selected_object is None
            or self._gizmo_mode == "select"
            or self._scene_buffers is None
            or self._scene_buffers.span_for(self._selected_object) is None
        ):
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

    def _release_gl_resources(self):
        if self._ctx is None:
            return
        self.makeCurrent()
        try:
            for resource in (
                self._scene_vao,
                self._scene_vbo,
                self._selection_vao,
                self._selection_vbo,
                self._grid_vao,
                self._grid_vbo,
                self._gizmo_vao,
                self._gizmo_vbo,
                self._scene_program,
                self._grid_program,
            ):
                self._release(resource)
        finally:
            self.doneCurrent()
        self._ctx = None

    @staticmethod
    def _release(resource):
        if resource is None:
            return
        try:
            resource.release()
        except Exception:
            pass


def _qt_wheel_delta(event):
    pixel_delta = event.pixelDelta()
    if not pixel_delta.isNull():
        return pixel_delta.x() / 120.0, pixel_delta.y() / 120.0
    angle_delta = event.angleDelta()
    return angle_delta.x() / 120.0, angle_delta.y() / 120.0
