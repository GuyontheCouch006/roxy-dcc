from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGL, QtOpenGLWidgets, QtWidgets

from rendering.gl_viewport import (
    GRID_FRAGMENT_SHADER,
    GRID_VERTEX_SHADER,
    SCENE_FRAGMENT_SHADER,
    SCENE_VERTEX_SHADER,
    ViewportCamera,
    _build_gizmo_vertices,
    _build_grid_vertices,
    _object_gizmo_axes,
    _object_gizmo_origin,
    _object_gizmo_size,
    _pinch_view_action,
    _scroll_wheel_view_action,
    build_scene_viewport_buffers,
    gizmo_axis_rotation_matrix,
    gizmo_axis_scale_matrix,
    move_gizmo_drag_delta,
    pick_move_gizmo_axis,
    pick_rotate_gizmo_axis,
    pick_scene_object,
    rotate_gizmo_drag_degrees,
    scale_gizmo_drag_factor,
)
from scene import SceneObject, SceneSession


GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_SCISSOR_TEST = 0x0C11
GL_STENCIL_TEST = 0x0B90
GL_BLEND = 0x0BE2
GL_FLOAT = 0x1406
GL_FRONT_AND_BACK = 0x0408
GL_LINE = 0x1B01
GL_FILL = 0x1B02
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004


def configure_default_gl_format():
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(surface_format)


configure_default_gl_format()


@dataclass
class TransformGizmoDrag:
    mode: str
    axis_name: str
    axis: np.ndarray
    origin: np.ndarray
    size: float
    start_mouse: np.ndarray


class QtGLViewport(QtOpenGLWidgets.QOpenGLWidget):
    objectSelected = QtCore.Signal(object)
    renderRequested = QtCore.Signal()
    sceneChanged = QtCore.Signal()

    def __init__(self, world=None, sync_camera=False, parent=None, session=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 280)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self._world = None
        self._session = None
        self._owns_session = False
        self._sync_camera = bool(sync_camera)
        self._camera = ViewportCamera()
        self._scene_buffers = None
        self._selected_object = None
        self._selected_objects = ()
        self._highlight_objects = ()
        self._gizmo_mode = "select"
        self._wireframe = False

        self._gl = None
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
        self._transform_drag = None

        if session is not None:
            self.set_session(session)
        elif world is not None:
            self.set_session(SceneSession(world), owns_session=True)

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
    def selected_objects(self):
        return self._selected_objects

    @property
    def highlighted_objects(self):
        return self._highlight_objects

    @property
    def gizmo_mode(self):
        return self._gizmo_mode

    @property
    def scene_buffers(self):
        return self._scene_buffers

    @property
    def session(self):
        return self._session

    def set_session(self, session, owns_session=False):
        if self._session is session:
            return
        if self._session is not None:
            self._session.remove_selection_listener(self._on_session_selection_changed)
            self._session.remove_scene_listener(self._on_session_scene_changed)
            self._session.remove_world_listener(self._on_session_world_changed)
        self._session = session
        self._owns_session = bool(owns_session)
        if session is None:
            self.set_world(None)
            return
        session.add_selection_listener(self._on_session_selection_changed)
        session.add_scene_listener(self._on_session_scene_changed)
        session.add_world_listener(self._on_session_world_changed)
        self.set_world(session.world, clear_selection=False)
        self._sync_selection_from_session()

    def _on_session_selection_changed(self, session):
        del session
        self._sync_selection_from_session()

    def _on_session_scene_changed(self, session):
        del session
        self.refresh_scene_geometry()

    def _on_session_world_changed(self, session):
        self.set_world(session.world, clear_selection=False)
        self._sync_selection_from_session()

    def _sync_selection_from_session(self):
        if self._session is None:
            return
        self._set_selected_objects_local(
            self._session.selected_scene_objects(),
            active_object=self._session.active_scene_object(),
            highlight_objects=self._session.highlighted_scene_objects(),
        )

    def set_world(self, world, frame=True, clear_selection=True):
        if self._session is None and world is not None:
            self.set_session(SceneSession(world), owns_session=True)
            return
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
        if clear_selection:
            self._set_selected_objects_local((), active_object=None)
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
        if self._session is not None:
            if scene_object is None:
                self._session.clear_selection()
            else:
                self._session.replace_selection(scene_object)
            if emit:
                self.objectSelected.emit(scene_object)
            return
        self.set_selected_objects(
            (scene_object,) if scene_object is not None else (),
            active_object=scene_object,
            emit=emit,
        )

    def set_selected_objects(self, scene_objects, active_object=None, emit=False):
        if self._session is not None:
            self._session.set_selection(scene_objects, active=active_object)
            if emit:
                self.objectSelected.emit(active_object)
            return
        self._set_selected_objects_local(scene_objects, active_object=active_object, emit=emit)

    def _set_selected_objects_local(self, scene_objects, active_object=None, emit=False, highlight_objects=None):
        selected = _unique_scene_objects(scene_objects)
        highlight = _unique_scene_objects(highlight_objects if highlight_objects is not None else selected)
        if active_object is not None and active_object not in selected:
            active_object = None
        if active_object is None and len(selected) == 1:
            active_object = selected[0]

        if (
            selected == self._selected_objects
            and active_object is self._selected_object
            and highlight == self._highlight_objects
        ):
            return
        self._selected_objects = selected
        self._selected_object = active_object
        self._highlight_objects = highlight
        self._upload_selection_if_ready()
        self._upload_gizmo_if_ready()
        self.update()
        if emit:
            self.objectSelected.emit(active_object)

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
        selected_bounds = None
        for scene_object in self._highlight_objects:
            span = self._scene_buffers.span_for(scene_object)
            if span is not None and span.bounds is not None:
                selected_bounds = (
                    span.bounds
                    if selected_bounds is None
                    else selected_bounds.union(span.bounds)
                )
        if selected_bounds is not None:
            bounds = selected_bounds
        if bounds is not None:
            self._camera.frame_bounds(bounds)
            self._sync_world_camera()
            self.update()

    def pick_object(self, screen_x, screen_y, toggle=False):
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
        if self._session is not None:
            if result is not None:
                if toggle:
                    self._session.toggle_selection(result.scene_object)
                else:
                    self._session.replace_selection(result.scene_object)
            elif not toggle:
                self._session.clear_selection()
            self.objectSelected.emit(self._session.active_scene_object())
        else:
            self.set_selected_object(result.scene_object if result else None, emit=True)
        return result

    def begin_move_gizmo_drag(self, screen_x, screen_y):
        if self._gizmo_mode != "move":
            return False
        return self.begin_transform_gizmo_drag(screen_x, screen_y)

    def begin_transform_gizmo_drag(self, screen_x, screen_y):
        if self._selected_object is None or self._gizmo_mode not in ("move", "rotate", "scale"):
            return False
        picker = pick_rotate_gizmo_axis if self._gizmo_mode == "rotate" else pick_move_gizmo_axis
        axes = _object_gizmo_axes(self._selected_object)
        hit = picker(
            self._selected_object,
            self._camera,
            screen_x,
            screen_y,
            max(self.width(), 1),
            max(self.height(), 1),
            axes=axes,
        )
        if hit is None:
            return False

        axis_name, axis, origin, size = hit
        if self._session is None or not self._session.begin_transform(
            label=f"{self._gizmo_mode.title()} Drag",
        ):
            return False
        self._transform_drag = TransformGizmoDrag(
            mode=self._gizmo_mode,
            axis_name=axis_name,
            axis=axis,
            origin=origin,
            size=size,
            start_mouse=np.asarray([screen_x, screen_y], dtype=np.float32),
        )
        return True

    def drag_move_gizmo(self, screen_x, screen_y):
        return self.drag_transform_gizmo(screen_x, screen_y)

    def drag_transform_gizmo(self, screen_x, screen_y):
        if self._transform_drag is None or self._session is None:
            return False
        drag = self._transform_drag
        current = np.asarray([screen_x, screen_y], dtype=np.float32)
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        pivot = None
        if drag.mode == "move":
            delta = move_gizmo_drag_delta(
                self._camera,
                drag.origin,
                drag.axis,
                drag.size,
                drag.start_mouse,
                current,
                width,
                height,
            )
            matrix = gizmo_axis_scale_matrix("x", 1.0)
            matrix.rows[0][3] = float(delta[0])
            matrix.rows[1][3] = float(delta[1])
            matrix.rows[2][3] = float(delta[2])
        elif drag.mode == "rotate":
            degrees = rotate_gizmo_drag_degrees(
                self._camera,
                drag.origin,
                drag.axis,
                drag.start_mouse,
                current,
                width,
                height,
            )
            matrix = gizmo_axis_rotation_matrix(drag.axis, degrees)
            pivot = drag.origin
        else:
            factor = scale_gizmo_drag_factor(
                self._camera,
                drag.origin,
                drag.axis,
                drag.size,
                drag.start_mouse,
                current,
                width,
                height,
            )
            matrix = gizmo_axis_scale_matrix(drag.axis, factor)
            pivot = drag.origin
        self._session.preview_transform(matrix, pivot=pivot)
        return True

    def end_move_gizmo_drag(self):
        self.end_transform_gizmo_drag()

    def end_transform_gizmo_drag(self):
        if self._transform_drag is not None and self._session is not None:
            self._session.finish_transform()
        self._transform_drag = None

    def initializeGL(self):
        self._gl = self.context().functions()
        self._gl.initializeOpenGLFunctions()
        self._gl.glEnable(GL_DEPTH_TEST)
        self._scene_program = self._create_program(
            SCENE_VERTEX_SHADER,
            SCENE_FRAGMENT_SHADER,
        )
        self._grid_program = self._create_program(
            GRID_VERTEX_SHADER,
            GRID_FRAGMENT_SHADER,
        )
        self._upload_grid_buffers()
        self._upload_scene_buffers()
        self._upload_selection_buffers()
        self._upload_gizmo_buffers()

    def resizeGL(self, width, height):
        if self._gl is None:
            return
        del width, height
        self.update()

    def paintGL(self):
        if self._gl is None:
            return
        width, height = self._physical_viewport_size()
        self._gl.glViewport(0, 0, width, height)
        self._gl.glDisable(GL_SCISSOR_TEST)
        self._gl.glDisable(GL_STENCIL_TEST)
        self._gl.glDisable(GL_BLEND)
        self._gl.glEnable(GL_DEPTH_TEST)
        self._gl.glClearColor(0.035, 0.038, 0.044, 1.0)
        self._gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._write_scene_matrices(self._scene_program)
        self._write_scene_matrices(self._grid_program)

        if self._grid_vao is not None:
            self._render_vertices(
                self._grid_program,
                self._grid_vao,
                GL_LINES,
                self._grid_vertex_count,
            )

        if self._scene_vao is not None and self._scene_buffers is not None:
            self._scene_program.bind()
            self._scene_program.setUniformValue(
                "light_direction",
                QtGui.QVector3D(0.35, 0.82, 0.44),
            )
            self._scene_program.release()
            self._set_wireframe(self._wireframe)
            self._render_vertices(
                self._scene_program,
                self._scene_vao,
                GL_TRIANGLES,
                self._scene_buffers.vertex_count,
            )
            self._set_wireframe(False)

        if self._selection_vao is not None:
            self._gl.glDisable(GL_DEPTH_TEST)
            self._set_wireframe(True)
            self._render_vertices(
                self._scene_program,
                self._selection_vao,
                GL_TRIANGLES,
                self._selection_vertex_count,
            )
            self._set_wireframe(False)
            self._gl.glEnable(GL_DEPTH_TEST)

        if self._gizmo_vao is not None:
            self._gl.glDisable(GL_DEPTH_TEST)
            self._render_vertices(
                self._grid_program,
                self._gizmo_vao,
                GL_LINES,
                self._gizmo_vertex_count,
            )
            self._gl.glEnable(GL_DEPTH_TEST)

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
                and self.begin_transform_gizmo_drag(pos.x(), pos.y())
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

        if self._transform_drag is not None:
            self.drag_transform_gizmo(pos.x(), pos.y())
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
                self.pick_object(
                    pos.x(),
                    pos.y(),
                    toggle=bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier),
                )
            self._drag_button = QtCore.Qt.MouseButton.NoButton
            self._mouse_down_pos = None
            self._last_mouse_pos = None
            self._mouse_dragged = False
            self.end_transform_gizmo_drag()
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
        modifiers = event.modifiers()
        if (
            key == QtCore.Qt.Key.Key_Z
            and modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
            and self._session is not None
        ):
            self._session.undo()
        elif key == QtCore.Qt.Key.Key_F:
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
        event_type = event.type() if callable(getattr(event, "type", None)) else None
        if event_type == QtCore.QEvent.Type.NativeGesture:
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
        if program is None:
            return
        width, height = self._physical_viewport_size()
        aspect = width / height if height else 1.0
        program.bind()
        program.setUniformValue("view", _qt_matrix(self._camera.view_matrix()))
        program.setUniformValue(
            "projection",
            _qt_matrix(self._camera.projection_matrix(aspect)),
        )
        program.release()

    def _create_program(self, vertex_shader, fragment_shader):
        program = QtOpenGL.QOpenGLShaderProgram()
        if not program.addShaderFromSourceCode(
            QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex,
            vertex_shader,
        ):
            raise RuntimeError(f"Vertex shader compile failed: {program.log()}")
        if not program.addShaderFromSourceCode(
            QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment,
            fragment_shader,
        ):
            raise RuntimeError(f"Fragment shader compile failed: {program.log()}")
        if not program.link():
            raise RuntimeError(f"Shader link failed: {program.log()}")
        return program

    def _render_vertices(self, program, vao, mode, vertex_count):
        if self._gl is None or program is None or vao is None or vertex_count <= 0:
            return
        program.bind()
        vao.bind()
        self._gl.glDrawArrays(mode, 0, int(vertex_count))
        vao.release()
        program.release()

    def _set_wireframe(self, enabled):
        if self._gl is None or not hasattr(self._gl, "glPolygonMode"):
            return
        self._gl.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if enabled else GL_FILL)

    def _physical_viewport_size(self):
        ratio = max(float(self.devicePixelRatioF()), 1.0)
        return (
            max(int(round(self.width() * ratio)), 1),
            max(int(round(self.height() * ratio)), 1),
        )

    def _create_vertex_array(self, program, vertices, attributes):
        vao = QtOpenGL.QOpenGLVertexArrayObject()
        if not vao.create():
            raise RuntimeError("Could not create OpenGL vertex array object")
        vao.bind()

        vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        if not vbo.create():
            vao.release()
            vao.destroy()
            raise RuntimeError("Could not create OpenGL vertex buffer")
        vbo.bind()
        vbo.allocate(vertices.tobytes(), vertices.nbytes)

        program.bind()
        for name, component_count, offset, stride in attributes:
            location = program.attributeLocation(name)
            if location < 0:
                continue
            program.enableAttributeArray(location)
            program.setAttributeBuffer(
                location,
                GL_FLOAT,
                int(offset),
                int(component_count),
                int(stride),
            )
        program.release()
        vbo.release()
        vao.release()
        return vao, vbo

    def _upload_all_if_ready(self):
        if self._gl is None:
            return
        self.makeCurrent()
        try:
            self._upload_scene_buffers()
            self._upload_selection_buffers()
            self._upload_gizmo_buffers()
        finally:
            self.doneCurrent()

    def _upload_selection_if_ready(self):
        if self._gl is None:
            return
        self.makeCurrent()
        try:
            self._upload_selection_buffers()
        finally:
            self.doneCurrent()

    def _upload_gizmo_if_ready(self):
        if self._gl is None:
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
        self._grid_vao, self._grid_vbo = self._create_vertex_array(
            self._grid_program,
            grid.astype(np.float32, copy=False),
            (
                ("in_position", 3, 0, 6 * 4),
                ("in_color", 3, 3 * 4, 6 * 4),
            ),
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
        self._scene_vao, self._scene_vbo = self._create_vertex_array(
            self._scene_program,
            interleaved,
            (
                ("in_position", 3, 0, 9 * 4),
                ("in_normal", 3, 3 * 4, 9 * 4),
                ("in_color", 3, 6 * 4, 9 * 4),
            ),
        )

    def _upload_selection_buffers(self):
        self._release(self._selection_vao)
        self._release(self._selection_vbo)
        self._selection_vao = None
        self._selection_vbo = None
        self._selection_vertex_count = 0

        if (
            self._scene_buffers is None
            or not self._highlight_objects
            or self._scene_buffers.is_empty
        ):
            return

        spans = [
            span
            for scene_object in self._highlight_objects
            for span in (self._scene_buffers.span_for(scene_object),)
            if span is not None and span.count > 0
        ]
        if not spans:
            return

        vertices = np.concatenate(
            [
                self._scene_buffers.vertices[span.start:span.start + span.count]
                for span in spans
            ],
            axis=0,
        )
        normals = np.concatenate(
            [
                self._scene_buffers.normals[span.start:span.start + span.count]
                for span in spans
            ],
            axis=0,
        )
        vertex_count = len(vertices)
        colors = np.repeat(
            np.asarray([[1.0, 0.63, 0.12]], dtype=np.float32),
            vertex_count,
            axis=0,
        )
        interleaved = np.concatenate([vertices, normals, colors], axis=1).astype(
            np.float32,
            copy=False,
        )
        self._selection_vertex_count = vertex_count
        self._selection_vao, self._selection_vbo = self._create_vertex_array(
            self._scene_program,
            interleaved,
            (
                ("in_position", 3, 0, 9 * 4),
                ("in_normal", 3, 3 * 4, 9 * 4),
                ("in_color", 3, 6 * 4, 9 * 4),
            ),
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
        ):
            return

        origin = _object_gizmo_origin(self._selected_object)
        size = _object_gizmo_size(self._selected_object)
        axes = _object_gizmo_axes(self._selected_object)
        vertices = _build_gizmo_vertices(origin, size, self._gizmo_mode, axes=axes)
        if len(vertices) == 0:
            return

        self._gizmo_vertex_count = len(vertices)
        self._gizmo_vao, self._gizmo_vbo = self._create_vertex_array(
            self._grid_program,
            vertices.astype(np.float32, copy=False),
            (
                ("in_position", 3, 0, 6 * 4),
                ("in_color", 3, 3 * 4, 6 * 4),
            ),
        )

    def _release_gl_resources(self):
        if self._gl is None:
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
        self._gl = None

    @staticmethod
    def _release(resource):
        if resource is None:
            return
        try:
            if hasattr(resource, "destroy"):
                resource.destroy()
            elif hasattr(resource, "removeAllShaders"):
                resource.removeAllShaders()
            elif hasattr(resource, "release"):
                resource.release()
        except Exception:
            pass


def _qt_wheel_delta(event):
    pixel_delta = event.pixelDelta()
    if not pixel_delta.isNull():
        return pixel_delta.x() / 120.0, pixel_delta.y() / 120.0
    angle_delta = event.angleDelta()
    return angle_delta.x() / 120.0, angle_delta.y() / 120.0


def _qt_matrix(matrix):
    values = np.asarray(matrix, dtype=np.float32).reshape((4, 4)).reshape(-1)
    return QtGui.QMatrix4x4(*(float(value) for value in values))


def _unique_scene_objects(scene_objects):
    selected = []
    seen = set()
    for scene_object in scene_objects or ():
        if not isinstance(scene_object, SceneObject):
            continue
        key = id(scene_object)
        if key in seen:
            continue
        seen.add(key)
        selected.append(scene_object)
    return tuple(selected)
