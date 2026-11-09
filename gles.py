# gles.py
"""
OpenGL ES 1.1 on top of the desktop OpenGL context pygame created.

ES 1.1 is (almost) a subset of the compatibility profile, so most calls are forwarded 1:1 after decoding the
ARM softfp argument layout (floats travel in core registers as raw bits, extra arguments spill to the stack).
The client-array calls are the interesting part: glVertexPointer & friends hand over *guest* addresses, so the
bytes are copied out of emulated memory when a draw call needs them and kept alive until the draw is done.

Also here: the OES framebuffer/renderbuffer entry points (the app's "screen" is the pygame window, so they only
need to hand out names and report a complete framebuffer), glGetString/glGetIntegerv answers that look like an
original iPhone, texture uploads sized by pixel format/type/alignment, and a quiet "not supported" for PVRTC.

handle(uc, name) returns True when it emulated `name` (leading underscores ignored).
"""
import logging
import struct

from unicorn import arm_const as A

try:                                   # desktop GL; absent only in headless unit tests
    import OpenGL.GL as gl
except Exception:                      # pragma: no cover
    gl = None

GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_FLOAT, GL_FIXED = 0x1400, 0x1401, 0x1402, 0x1403, 0x1406, 0x140C
_TYPE_SIZE = {GL_BYTE: 1, GL_UNSIGNED_BYTE: 1, GL_SHORT: 2, GL_UNSIGNED_SHORT: 2, GL_FLOAT: 4, GL_FIXED: 4}
GL_RGBA, GL_RGB, GL_ALPHA, GL_LUMINANCE, GL_LUMINANCE_ALPHA = 0x1908, 0x1907, 0x1906, 0x1909, 0x190A
_PACKED16 = (0x8033, 0x8034, 0x8363)   # 4444, 5551, 565
_CHANNELS = {GL_RGBA: 4, GL_RGB: 3, GL_ALPHA: 1, GL_LUMINANCE: 1, GL_LUMINANCE_ALPHA: 2}
GL_UNPACK_ALIGNMENT = 0x0CF5

_STRINGS = {0x1F00: 'Apple Computer, Inc.', 0x1F01: 'PowerVR MBX Lite', 0x1F02: 'OpenGL ES-CM 1.1',
            0x1F03: 'GL_OES_framebuffer_object GL_OES_rgb8_rgba8 GL_OES_read_format GL_OES_depth24 '
                    'GL_OES_mapbuffer GL_EXT_texture_filter_anisotropic'}


class GLES:
    def __init__(self, cpu):
        self.cpu = cpu
        self.arrays = {}                 # 'vertex' | 'color' | 'texcoord' | 'normal' -> (size, type, stride, ptr)
        self._keep = []                  # buffers handed to GL; must outlive the draw call
        self._strings = {}
        self._next_name = 1
        self.unpack_alignment = 4
        self.frame_drawn = False         # an app draw call happened since the last presented frame

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _f(w):
        return struct.unpack('<f', struct.pack('<I', w & 0xFFFFFFFF))[0]

    def _static_string(self, uc, text):
        if text not in self._strings:
            raw = text.encode() + b'\x00'
            ptr = self.cpu._c_alloc(len(raw))
            uc.mem_write(ptr, raw)
            self._strings[text] = ptr
        return self._strings[text]

    def _keepalive(self, data):
        self._keep.append(data)
        if len(self._keep) > 64:
            self._keep.pop(0)
        return data

    def _array_bytes(self, uc, spec, count, first=0):
        """Copy `count` elements (starting at `first`) of one client array out of guest memory."""
        size, typ, stride, ptr = spec
        elem = size * _TYPE_SIZE.get(typ, 4)
        stride = stride or elem
        total = stride * (first + count - 1) + elem if count > 0 else 0
        try:
            return self._keepalive(bytes(uc.mem_read(ptr, total)))
        except Exception as e:
            logging.warning(f"[GLES] cannot read client array at 0x{ptr:08X}: {e}")
            return None

    def _bind_arrays(self, uc, count, first=0):
        if gl is None:
            return
        spec = self.arrays.get('vertex')
        if spec and gl.glIsEnabled(0x8074):
            data = self._array_bytes(uc, spec, count, first)
            if data is not None:
                gl.glVertexPointer(spec[0], spec[1], spec[2], data)
        spec = self.arrays.get('color')
        if spec and gl.glIsEnabled(0x8076):
            data = self._array_bytes(uc, spec, count, first)
            if data is not None:
                gl.glColorPointer(spec[0], spec[1], spec[2], data)
        spec = self.arrays.get('texcoord')
        if spec and gl.glIsEnabled(0x8078):
            data = self._array_bytes(uc, spec, count, first)
            if data is not None:
                gl.glTexCoordPointer(spec[0], spec[1], spec[2], data)
        spec = self.arrays.get('normal')
        if spec and gl.glIsEnabled(0x8075):
            data = self._array_bytes(uc, spec, count, first)
            if data is not None:
                gl.glNormalPointer(spec[1], spec[2], data)

    # ---- the calls -----------------------------------------------------------
    def handle(self, uc, name):
        base = name.lstrip('_')
        if not (base.startswith('gl') or base.startswith('egl')):
            return False
        cpu = self.cpu
        r0, r1, r2, r3 = (uc.reg_read(x) for x in (A.UC_ARM_REG_R0, A.UC_ARM_REG_R1, A.UC_ARM_REG_R2, A.UC_ARM_REG_R3))
        sp = uc.reg_read(A.UC_ARM_REG_SP)
        f = self._f

        def ret(v):
            uc.reg_write(A.UC_ARM_REG_R0, v & 0xFFFFFFFF)

        def write_ints(addr, values):
            uc.mem_write(addr, b''.join(struct.pack('<I', v & 0xFFFFFFFF) for v in values))

        # ---------- OES framebuffer / renderbuffer: the window is the framebuffer ----------
        if base in ('glGenFramebuffersOES', 'glGenRenderbuffersOES'):
            ids = []
            for _ in range(r0):
                ids.append(self._next_name)
                self._next_name += 1
            write_ints(r1, ids)
            return True
        if base in ('glBindFramebufferOES', 'glBindRenderbufferOES', 'glRenderbufferStorageOES',
                    'glFramebufferRenderbufferOES', 'glFramebufferTexture2DOES', 'glDeleteFramebuffersOES',
                    'glDeleteRenderbuffersOES', 'glFinish', 'glFlush', 'glGenerateMipmapOES'):
            if base == 'glGenerateMipmapOES' and gl is not None:
                try:
                    gl.glGenerateMipmap(r0)
                except Exception:
                    pass
            return True
        if base == 'glCheckFramebufferStatusOES':
            ret(0x8CD5)                                       # GL_FRAMEBUFFER_COMPLETE_OES
            return True
        if base == 'glGetRenderbufferParameterivOES':
            val = {0x8D42: cpu.gpu.width if getattr(cpu, 'gpu', None) else 320,
                   0x8D43: cpu.gpu.height if getattr(cpu, 'gpu', None) else 480}.get(r1, 0)
            write_ints(r2, [val])
            return True

        # ---------- queries ----------
        if base == 'glGetError':
            ret(0)
            return True
        if base == 'glGetString':
            ret(self._static_string(uc, _STRINGS.get(r0, '')))
            return True
        if base in ('glGetIntegerv', 'glGetFloatv'):
            val = {0x0D33: 1024, 0x84E2: 2, 0x0D3A: 1024, 0x0BA2: 0, 0x8B9A: 0x1908, 0x8B9B: 0x1401,
                   0x0D57: 8, 0x0D55: 8, 0x0D56: 8, 0x0D54: 8}.get(r0, 0)
            if base == 'glGetIntegerv':
                write_ints(r1, [val])
            else:
                uc.mem_write(r1, struct.pack('<f', float(val)))
            return True

        # ---------- state with no pointers ----------
        if gl is None:
            return base in ('glTexEnvi', 'glEnable', 'glDisable', 'glTexParameteri', 'glViewport', 'glActiveTexture', 'glClientActiveTexture', 'glAlphaFunc', 'glBlendFunc', 'glCullFace',
                            'glDepthFunc', 'glDepthMask', 'glColorMask', 'glClearDepthf', 'glColor4f', 'glColor4ub',
                            'glHint', 'glShadeModel', 'glScissor', 'glPixelStorei', 'glFogf', 'glFogx', 'glLoadIdentity',
                            'glMatrixMode', 'glOrthof', 'glPushMatrix', 'glPopMatrix', 'glRotatef', 'glScalef',
                            'glTranslatef', 'glTexEnvf', 'glTexParameterf', 'glEnableClientState', 'glDisableClientState')
        try:
            if base == 'glActiveTexture':
                gl.glActiveTexture(r0)
            elif base == 'glClientActiveTexture':
                gl.glClientActiveTexture(r0)
            elif base == 'glEnable':
                gl.glEnable(r0)
            elif base == 'glDisable':
                gl.glDisable(r0)
            elif base == 'glTexParameteri':
                gl.glTexParameteri(r0, r1, r2)
            elif base == 'glViewport':
                pass                                          # the window is the framebuffer; keep our own viewport
            elif base == 'glAlphaFunc':
                gl.glAlphaFunc(r0, f(r1))
            elif base == 'glBlendFunc':
                gl.glBlendFunc(r0, r1)
            elif base == 'glCullFace':
                gl.glCullFace(r0)
            elif base == 'glDepthFunc':
                gl.glDepthFunc(r0)
            elif base == 'glDepthMask':
                gl.glDepthMask(bool(r0 & 0xFF))
            elif base == 'glColorMask':
                gl.glColorMask(bool(r0 & 0xFF), bool(r1 & 0xFF), bool(r2 & 0xFF), bool(r3 & 0xFF))
            elif base == 'glClearDepthf':
                gl.glClearDepth(f(r0))
            elif base == 'glColor4f':
                gl.glColor4f(f(r0), f(r1), f(r2), f(r3))
            elif base == 'glColor4ub':
                gl.glColor4ub(r0 & 0xFF, r1 & 0xFF, r2 & 0xFF, r3 & 0xFF)
            elif base == 'glHint':
                gl.glHint(r0, r1)
            elif base == 'glShadeModel':
                gl.glShadeModel(r0)
            elif base == 'glScissor':
                gl.glScissor(r0, r1, r2, r3)
            elif base == 'glPixelStorei':
                if r0 == GL_UNPACK_ALIGNMENT:
                    self.unpack_alignment = r1 or 4
                gl.glPixelStorei(r0, r1)
            elif base in ('glFogf', 'glFogx'):
                gl.glFogf(r0, f(r1) if base == 'glFogf' else (r1 - (1 << 32) if r1 & 0x80000000 else r1) / 65536.0)
            elif base == 'glFogfv':
                gl.glFogfv(r0, struct.unpack('<4f', bytes(uc.mem_read(r1, 16))))
            elif base == 'glLoadIdentity':
                gl.glLoadIdentity()
            elif base == 'glMatrixMode':
                gl.glMatrixMode(r0)
            elif base in ('glMultMatrixf', 'glLoadMatrixf'):
                m = struct.unpack('<16f', bytes(uc.mem_read(r0, 64)))
                (gl.glMultMatrixf if base == 'glMultMatrixf' else gl.glLoadMatrixf)(m)
            elif base == 'glOrthof':
                s0, s1 = cpu._stack_words(uc, sp, 2)
                gl.glOrtho(f(r0), f(r1), f(r2), f(r3), f(s0), f(s1))
            elif base == 'glPushMatrix':
                gl.glPushMatrix()
            elif base == 'glPopMatrix':
                gl.glPopMatrix()
            elif base == 'glRotatef':
                gl.glRotatef(f(r0), f(r1), f(r2), f(r3))
            elif base == 'glScalef':
                gl.glScalef(f(r0), f(r1), f(r2))
            elif base == 'glTranslatef':
                gl.glTranslatef(f(r0), f(r1), f(r2))
            elif base == 'glTexEnvf':
                gl.glTexEnvf(r0, r1, f(r2))
            elif base == 'glTexEnvi':
                gl.glTexEnvi(r0, r1, r2)
            elif base == 'glTexParameterf':
                gl.glTexParameterf(r0, r1, f(r2))
            elif base == 'glEnableClientState':
                gl.glEnableClientState(r0)
            elif base == 'glDisableClientState':
                gl.glDisableClientState(r0)
            elif base == 'glDeleteTextures':
                names = struct.unpack(f'<{r0}I', bytes(uc.mem_read(r1, 4 * r0))) if r0 else ()
                if names:
                    gl.glDeleteTextures(list(names))

            # ---------- client arrays ----------
            elif base in ('glVertexPointer', 'glColorPointer', 'glTexCoordPointer'):
                key = {'glVertexPointer': 'vertex', 'glColorPointer': 'color', 'glTexCoordPointer': 'texcoord'}[base]
                self.arrays[key] = (r0, r1, r2, r3)
            elif base == 'glNormalPointer':
                self.arrays['normal'] = (3, r0, r1, r2)
            elif base == 'glDrawArrays':
                mode, first, count = r0, r1, r2
                self._bind_arrays(uc, count, first)
                gl.glDrawArrays(mode, first, count)
                self.frame_drawn = True
            elif base == 'glDrawElements':
                mode, count, typ, ptr = r0, r1, r2, r3
                isz = _TYPE_SIZE.get(typ, 2)
                idx = bytes(uc.mem_read(ptr, count * isz)) if count else b''
                values = struct.unpack(f'<{count}{"H" if isz == 2 else "B"}', idx) if count else ()
                self._bind_arrays(uc, (max(values) + 1) if values else 0, 0)
                gl.glDrawElements(mode, count, typ, self._keepalive(idx))
                self.frame_drawn = True

            # ---------- textures ----------
            elif base == 'glTexImage2D':                       # (target, level, internalformat, w, h, border, format, type, pixels)
                height, border, fmt, typ, pixels = cpu._stack_words(uc, sp, 5)
                self._tex_image(uc, r0, r1, r2, r3, height, border, fmt, typ, pixels)
            elif base == 'glCompressedTexImage2D':
                logging.warning("[GLES] glCompressedTexImage2D (PVRTC) is not supported - texture left empty")
            else:
                return False
        except Exception as e:
            logging.warning(f"[GLES] {base} failed: {e}")
        return True

    def _tex_image(self, uc, target, level, ifmt, width, height, border, fmt, typ, pixels):
        if gl is None:
            return
        if not pixels:
            gl.glTexImage2D(target, level, ifmt, width, height, border, fmt, typ, None)
            return
        if typ in _PACKED16:
            bpp = 2
        else:
            bpp = _CHANNELS.get(fmt, 4) * _TYPE_SIZE.get(typ, 1)
        row = width * bpp
        al = self.unpack_alignment or 4
        stride = (row + al - 1) // al * al
        data = bytes(uc.mem_read(pixels, stride * height)) if height else b''
        gl.glTexImage2D(target, level, ifmt, width, height, border, fmt, typ, self._keepalive(data))
