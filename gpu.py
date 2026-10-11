# gpu.py
import pygame
import logging
from OpenGL.GL import *
import numpy as np

class PowerVRGPU:
    def __init__(self, width=320, height=480, uc_engine=None):
        self.width = width
        self.height = height
        self.uc = uc_engine
        self.initialized = False
        
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        self.viewport = (0, 0, width, height)
        self.current_matrix_mode = 0x1700
        
        self.modelview_stack = [np.identity(4, dtype=np.float32)]
        self.projection_stack = [np.identity(4, dtype=np.float32)]
        self.texture_stack = [np.identity(4, dtype=np.float32)]
        
        self.client_states = {
            "vertex_array": False,
            "normal_array": False,
            "color_array": False,
            "texcoord_array": False
        }
        
        self.vertex_pointer = None
        self.color_pointer = None
        self.texcoord_pointer = None
        self._persistent_client_buffers = []
        
        self.active_texture_unit = 0
        self.bound_textures = {0: 0}
        self.texture_cache = {}
        
        self.frame_count = 0
        self.context_active = False
        self.active_draw_flag = False

    def set_uc_engine(self, uc_engine):
        self.uc = uc_engine

    def init_gl_context(self):
        # NOTE: GL_CONTEXT_* attributes (major/minor version, compatibility
        # profile) are NOT set here anymore - by the time eagl_create_context()
        # runs, the pygame window (and its GL context) already exists, so
        # gl_set_attribute() calls here would be silent no-ops. They're now
        # set in main.py's set_display_mode("APP") BEFORE pygame.display.set_mode()
        # creates the window, which is the only point SDL actually reads them.
        try:
            glClearColor(*self.clear_color)
            glViewport(*self.viewport)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            self.initialized = True
            logging.info(f"[PowerVR SGX] Full Desktop GL Compatibility Pipeline Initialized ({self.width}x{self.height})")
        except Exception as e:
            logging.error(f"[PowerVR SGX Error] Failed to bind hardware context: {e}")
            self.initialized = False

    def eagl_create_context(self):
        self.context_active = True
        if not self.initialized:
            self.init_gl_context()
        logging.info("[PowerVR SGX] EAGLContext bound. Framebuffer active.")
        return 0xCAFEBABE

    def gl_matrix_mode(self, mode):
        self.current_matrix_mode = mode

    def gl_load_identity(self):
        identity = np.identity(4, dtype=np.float32)
        if self.current_matrix_mode == 0x1700:
            self.modelview_stack[-1] = identity
        elif self.current_matrix_mode == 0x1701:
            self.projection_stack[-1] = identity
        elif self.current_matrix_mode == 0x1702:
            self.texture_stack[-1] = identity

    def gl_clear_color(self, r, g, b, a):
        self.clear_color = (r, g, b, a)
        if self.initialized:
            glClearColor(r, g, b, a)

    def gl_clear(self, mask):
        if self.initialized:
            gl_mask = 0
            if mask & 0x4000: gl_mask |= GL_COLOR_BUFFER_BIT
            if mask & 0x0100: gl_mask |= GL_DEPTH_BUFFER_BIT
            if mask & 0x0200: gl_mask |= GL_STENCIL_BUFFER_BIT
            glClear(gl_mask if gl_mask else GL_COLOR_BUFFER_BIT)

    def gl_viewport(self, x, y, w, h):
        self.viewport = (x, y, w, h)
        if self.initialized:
            glViewport(x, y, w, h)

    def gl_enable_client_state(self, array_enum):
        mapping = {0x8074: "vertex_array", 0x8075: "normal_array", 0x8076: "color_array", 0x8078: "texcoord_array"}
        if array_enum in mapping:
            self.client_states[mapping[array_enum]] = True

    def gl_disable_client_state(self, array_enum):
        mapping = {0x8074: "vertex_array", 0x8075: "normal_array", 0x8076: "color_array", 0x8078: "texcoord_array"}
        if array_enum in mapping:
            self.client_states[mapping[array_enum]] = False

    def gl_vertex_pointer(self, size, dtype, stride, pointer):
        self.vertex_pointer = {"size": size, "type": dtype, "stride": stride, "pointer": pointer}

    def gl_color_pointer(self, size, dtype, stride, pointer):
        self.color_pointer = {"size": size, "type": dtype, "stride": stride, "pointer": pointer}

    def gl_tex_coord_pointer(self, size, dtype, stride, pointer):
        self.texcoord_pointer = {"size": size, "type": dtype, "stride": stride, "pointer": pointer}

    def gl_gen_textures(self, n):
        return glGenTextures(n)

    def gl_bind_texture(self, target, texture):
        glBindTexture(target, texture)
        self.bound_textures[self.active_texture_unit] = texture

    def gl_tex_image_2d(self, target, level, internalformat, width, height, border, format, type, pixels):
        if not self.uc or not pixels:
            glTexImage2D(target, level, internalformat, width, height, border, format, type, None)
            return
        bpp = 4 if format in [0x1908, 0x80E1] else 3
        raw_pixels = self._read_guest_pointer(pixels, width * height * bpp)
        if raw_pixels:
            glTexImage2D(target, level, internalformat, width, height, border, format, type, raw_pixels)

    def _read_guest_pointer(self, address, size_in_bytes):
        if not self.uc or not address or size_in_bytes <= 0:
            return None
        try:
            data = bytes(self.uc.mem_read(address, size_in_bytes))
            self._persistent_client_buffers.append(data)
            if len(self._persistent_client_buffers) > 64:
                self._persistent_client_buffers.pop(0)
            return data
        except Exception as e:
            logging.error(f"[PowerVR SGX Error] Unsafe guest pointer read at 0x{address:08X}: {e}")
            return None

    def gl_draw_arrays(self, mode, first, count):
        if not self.initialized:
            return
        try:
            self.active_draw_flag = True
            if self.client_states["vertex_array"] and self.vertex_pointer:
                vp = self.vertex_pointer
                elem_size = 4 if vp["type"] == 0x1406 else 4
                stride = vp["stride"] if vp["stride"] > 0 else vp["size"] * elem_size
                total_bytes = stride * count + (first * stride)
                
                raw_data = self._read_guest_pointer(vp["pointer"], total_bytes)
                if raw_data:
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glVertexPointer(vp["size"], GL_FLOAT, vp["stride"], raw_data)
            glDrawArrays(mode, first, count)
        except Exception as e:
            logging.error(f"[PowerVR SGX Render Error] glDrawArrays exception: {e}")

    def gl_draw_elements(self, mode, count, dtype, indices_pointer):
        if not self.initialized:
            return
        try:
            self.active_draw_flag = True
            type_size = 2 if dtype == 0x1403 else 1
            raw_indices = self._read_guest_pointer(indices_pointer, count * type_size)
            if raw_indices:
                glDrawElements(mode, count, dtype, raw_indices)
        except Exception as e:
            logging.error(f"[PowerVR SGX Render Error] glDrawElements exception: {e}")

    def eagl_present_renderbuffer(self):
        # Does NOT call pygame.display.flip() itself - main.py's run loop
        # already does exactly one unconditional flip per tick, which is
        # the sole place presentation should happen. This used to ALSO
        # flip here, so a frame could get flipped twice in one tick while
        # draw_cg_surface() (called every tick, with no glClear of its
        # own) kept compositing onto whatever was left in the back buffer -
        # with double buffering, that meant the two physical buffers
        # accumulated different content over time, and alternating between
        # them each swap is exactly what a flicker looks like. One flip
        # per tick, one clear per tick (see render_frame) fixes both ends
        # of that.
        if self.initialized:
            self.frame_count += 1
            if self.frame_count == 1:
                logging.info("[PowerVR SGX] Framebuffer successfully presented via EAGLContext!")

    def draw_cg_surface(self, cg_surface):
        # Composites the software CoreGraphics layer (built by CGContext*/
        # UIGraphics* stub calls in cpu.py) into the GL framebuffer.
        #
        # Previously used glDrawPixels + glWindowPos2i. That's spec-legal
        # but glWindowPos support and glDrawPixels behavior varies across
        # real-world GL drivers (some silently no-op instead of raising),
        # so it's a poor choice when we can't see the actual GL error
        # state. A texture + quad blit is the standard, universally
        # supported way to composite a CPU-side buffer in compatibility
        # profile, and lets us define the coordinate mapping explicitly
        # instead of relying on window-coordinate raster-position quirks.
        if not self.initialized or cg_surface is None:
            return
        try:
            pixel_data = pygame.image.tostring(cg_surface, "RGBA", False)

            if not hasattr(self, '_cg_texture_id'):
                self._cg_texture_id = glGenTextures(1)

            glBindTexture(GL_TEXTURE_2D, self._cg_texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, pixel_data)

            # Top-left-origin ortho projection: (0,0) is the top-left pixel,
            # (width,height) is bottom-right - matches pygame's own surface
            # coordinate convention, so no manual flipping is needed.
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            was_depth = glIsEnabled(GL_DEPTH_TEST)
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(0, 0)
            glTexCoord2f(1, 0); glVertex2f(self.width, 0)
            glTexCoord2f(1, 1); glVertex2f(self.width, self.height)
            glTexCoord2f(0, 1); glVertex2f(0, self.height)
            glEnd()

            glDisable(GL_TEXTURE_2D)
            if was_depth:
                glEnable(GL_DEPTH_TEST)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
        except Exception as e:
            logging.error(f"[PowerVR SGX Error] CG surface blit failed: {e}")

    def render_frame(self, target_window, active_app_name=""):
        if not self.initialized:
            target_window.fill((0, 0, 0))
            font = pygame.font.SysFont(None, 16)
            txt = font.render(f"App: {active_app_name} | Frames: {self.frame_count} (Waiting for Context)", True, (200, 200, 200))
            target_window.blit(txt, (10, target_window.get_height() - 30))
        else:
            # Clear every tick, unconditionally - draw_cg_surface() (called
            # right after this, every tick, by main.py) composites onto
            # whatever's currently in the back buffer with no clear of its
            # own. Without clearing here first, the two physical buffers of
            # the double-buffered swap chain drift out of sync with each
            # other over time, which is what was causing the flicker.
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if self.active_draw_flag:
                self.eagl_present_renderbuffer()