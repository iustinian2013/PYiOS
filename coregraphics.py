# coregraphics.py
import pygame
import logging


class QuartzContext:
    """
    A deliberately simplified software CoreGraphics context.

    Real CGContext supports arbitrary bezier paths, strokes, fonts,
    gradients, and real image decoding. This implements just enough of the
    common "build a path -> clip to it -> draw an image into the clipped
    region" pattern (the standard iOS technique for a masked fill animation,
    e.g. a beer glass filling up as you tap) to produce a visible
    approximation - not a faithful renderer, and not a general Quartz
    implementation.

    Only one CGContextRef is ever active in this emulator, represented by
    the fixed fake pointer CONTEXT_HANDLE - there's no real allocator behind
    it, just enough of a token for handle_api_stub's r0 return value to be
    self-consistent across calls.
    """

    CONTEXT_HANDLE = 0xCAFED00D

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.reset()

    def reset_ctm(self):
        self.ctm_tx, self.ctm_ty = 0.0, 0.0
        self.ctm_sx, self.ctm_sy = 1.0, 1.0

    def reset(self):
        """
        Reset to a fresh per-draw state: CTM, current path, and clip all
        back to defaults, clip_rect back to the full surface.

        Real UIKit hands -drawRect: a brand-new CGContext (or at least a
        freshly re-clipped/re-transformed one) on every redraw pass. This
        emulator instead keeps ONE persistent QuartzContext for the whole
        app lifetime with no CGContextSaveGState/RestoreGState, so without
        an explicit per-frame reset, clip() intersections only ever shrink
        and never recover: a genuinely empty clip on frame N (a legitimate
        "nothing visible yet" state - see clip()'s docstring) would
        permanently blank every draw on every frame after it.

        Call this once per real -drawRect: invocation - the natural hook
        is _UIGraphicsGetCurrentContext, since the app calls that first
        thing every time it wants to draw.
        """
        self.surface.fill((0, 0, 0, 0))  # every drawRect: starts on a cleared, transparent context
        self.reset_ctm()
        self.current_path = []  # (x, y) points of the CURRENT subpath, post-CTM
        self.clip_rect = pygame.Rect(0, 0, self.width, self.height)
        self.clip_mask = None   # per-pixel alpha mask when a non-rectangular (tilted) clip is active
        self._path_started = False

    def translate_ctm(self, tx, ty):
        self.ctm_tx += tx
        self.ctm_ty += ty

    def scale_ctm(self, sx, sy):
        self.ctm_sx *= sx
        self.ctm_sy *= sy

    def _apply_ctm(self, x, y):
        return (x * self.ctm_sx + self.ctm_tx, y * self.ctm_sy + self.ctm_ty)

    def move_to(self, x, y):
        self.current_path = [self._apply_ctm(x, y)]
        self._path_started = True

    def add_line_to(self, x, y):
        if not self._path_started:
            self.move_to(x, y)
            return
        self.current_path.append(self._apply_ctm(x, y))

    def close_path(self):
        if len(self.current_path) > 1:
            self.current_path.append(self.current_path[0])

    @staticmethod
    def _is_axis_aligned_rect(points):
        pts = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
        if len(pts) != 4:
            return False
        xs = sorted({round(p[0], 3) for p in pts})
        ys = sorted({round(p[1], 3) for p in pts})
        return len(xs) <= 2 and len(ys) <= 2

    def clip(self):
        # Real CGContextClip intersects the current clip with the current path.
        # Axis-aligned rectangles (the common fill-level case) stay a cheap
        # bounding box. Anything rotated - e.g. the beer surface when the phone
        # is tilted - is rasterised into a per-pixel mask so the tilt is
        # actually visible instead of collapsing to a huge bounding box.
        if len(self.current_path) < 2:
            return
        xs = [p[0] for p in self.current_path]
        ys = [p[1] for p in self.current_path]
        path_rect = pygame.Rect(
            int(min(xs)), int(min(ys)),
            max(1, int(max(xs) - min(xs))), max(1, int(max(ys) - min(ys)))
        )
        new_clip = self.clip_rect.clip(path_rect)
        # An empty intersection is a REAL result - apps clip to an off-screen rect to mean
        # "draw nothing" (e.g. the foam/slime overlay while there's still beer). reset()
        # runs at the top of every drawRect:, so an empty clip only affects that frame.
        self.clip_rect = new_clip

        if (len(self.current_path) >= 4 and new_clip.width > 0 and new_clip.height > 0
                and not self._is_axis_aligned_rect(self.current_path)):
            mask = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            mask.fill((0, 0, 0, 0))
            pygame.draw.polygon(mask, (255, 255, 255, 255),
                                [(int(round(x)), int(round(y))) for x, y in self.current_path])
            if self.clip_mask is not None:
                mask.blit(self.clip_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.clip_mask = mask

        self.current_path = []
        self._path_started = False

    def draw_image(self, x, y, w, h, surface=None, color=(214, 158, 46, 255)):
        rect = pygame.Rect(int(x), int(y), max(1, int(w)), max(1, int(h)))
        draw_rect = rect.clip(self.clip_rect)
        if draw_rect.width <= 0 or draw_rect.height <= 0:
            return
        if self.clip_mask is not None:
            layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            layer.fill((0, 0, 0, 0))
            if surface is not None:
                layer.blit(pygame.transform.smoothscale(surface, (rect.width, rect.height)), rect.topleft)
            else:
                pygame.draw.rect(layer, color, rect)
            layer.blit(self.clip_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.surface.blit(layer, (0, 0))
            return
        if surface is not None:
            scaled = pygame.transform.smoothscale(surface, (rect.width, rect.height))
            self.surface.blit(scaled, draw_rect.topleft, area=pygame.Rect(
                draw_rect.x - rect.x, draw_rect.y - rect.y, draw_rect.width, draw_rect.height))
        else:
            pygame.draw.rect(self.surface, color, draw_rect)

    def get_surface(self):
        return self.surface