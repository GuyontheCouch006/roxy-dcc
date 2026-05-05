# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: OpenGL viewport using moderngl + pygame.
#              Displays the Image pixel buffer as a fullscreen quad texture.
#              Supports progressive scanline updates for live ray tracer preview.
# ============================================

import pygame
import moderngl
import numpy as np


# ── Shaders ──────────────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """
#version 330

uniform sampler2D texture0;
in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = texture(texture0, v_uv);
}
"""

# Fullscreen quad — two triangles covering NDC [-1,1]
# Each vertex: x, y, u, v
QUAD_VERTICES = np.array([
    -1.0, -1.0,  0.0, 1.0,   # bottom-left   (image Y flipped: v=1 at bottom)
     1.0, -1.0,  1.0, 1.0,   # bottom-right
     1.0,  1.0,  1.0, 0.0,   # top-right
    -1.0, -1.0,  0.0, 1.0,   # bottom-left
     1.0,  1.0,  1.0, 0.0,   # top-right
    -1.0,  1.0,  0.0, 0.0,   # top-left
], dtype='f4')


class GLViewport:
    """Minimal OpenGL window for displaying a ray traced Image buffer.

    Usage:
        viewport = GLViewport(800, 400, "Roxy")
        while not viewport.should_close:
            viewport.poll_events()
            # ... trace scanline into image ...
            viewport.update(image)
        viewport.close()
    """

    def __init__(self, width, height, title="Roxy"):
        self._width = width
        self._height = height
        self._should_close = False

        # ── pygame window with OpenGL context ────────────────────────────────
        pygame.init()
        pygame.display.set_caption(title)
        
        # macOS requires explicit OpenGL 3.3 core profile request
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        pygame.display.set_mode(
            (width, height),
            pygame.OPENGL | pygame.DOUBLEBUF
        )

        # ── moderngl context ─────────────────────────────────────────────────
        self._ctx = moderngl.create_context()
        self._ctx.viewport = (0, 0, width, height)

        # ── Shader program ───────────────────────────────────────────────────
        self._program = self._ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        # ── Fullscreen quad VAO ──────────────────────────────────────────────
        self._vbo = self._ctx.buffer(QUAD_VERTICES.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # ── Texture (float32, RGB) ────────────────────────────────────────────
        # dtype='f4' matches Image._pixels (np.float32)
        self._texture = self._ctx.texture(
            (width, height), 3, dtype='f4'
        )
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.use(0)
        self._program['texture0'] = 0

        # Clear to black initially
        self._ctx.clear(0.0, 0.0, 0.0)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def should_close(self):
        return self._should_close

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    # ── Public interface ──────────────────────────────────────────────────────

    def poll_events(self):
        """Process pygame events — call once per frame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_close = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_close = True

    def update(self, image):
        """Upload Image pixel buffer to GPU texture and draw the fullscreen quad."""
        # Upload numpy array to texture
        # contiguous() ensures the array is a flat C-order block in memory
        self._texture.write(image.pixels.tobytes())

        # Draw
        self._ctx.clear(0.0, 0.0, 0.0)
        self._vao.render(moderngl.TRIANGLES)
        pygame.display.flip()

    def update_scanline(self, image, y):
        """Upload only a single scanline — more efficient for progressive rendering.
        
        Writes one row of pixels to the texture at the correct vertical offset.
        """
        row_data = np.ascontiguousarray(image.pixels[y:y+1])
        self._texture.write(row_data.tobytes(), viewport=(0, y, self._width, 1))
        self._ctx.clear(0.0, 0.0, 0.0)
        self._vao.render(moderngl.TRIANGLES)
        pygame.display.flip()

    def close(self):
        """Clean up GL resources and quit pygame."""
        self._vao.release()
        self._vbo.release()
        self._texture.release()
        self._program.release()
        self._ctx.release()
        pygame.quit()

    def __repr__(self):
        return f"GLViewport({self._width}x{self._height})"