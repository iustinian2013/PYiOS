# nib_loader.py
"""
Loads compiled Interface Builder NIBs (NSKeyedArchiver binary plists) and
gives the emulator a real UIKit-ish view layer: a view tree with frames,
images, labels and buttons; outlet and action connections; hit testing;
and a compositor that draws it all, including the output of the app's
own -drawRect: (custom-class views), into one frame.

This is the piece that was missing: without it the app's window, glass,
beer, button and every IBOutlet/IBAction simply did not exist, so the
only thing that could ever run was one custom view's drawRect:.
"""
import logging
import math
import os
import plistlib
import re
import time

import pygame

_NUM = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

# Classes that are views (or view-like) for tree building.
_VIEW_CLASSES = {
    'UIView', 'UIImageView', 'UIButton', 'UILabel', 'UIWindow', 'UIScrollView',
    'UIControl', 'UITextField', 'UITextView', 'UIToolbar', 'UINavigationBar',
    'UIRoundedRectButton', 'UISlider', 'UISwitch', 'UIActivityIndicatorView',
}


def _numbers(value):
    """Extract floats from a NIB geometry value (string, bytes or NSData)."""
    if isinstance(value, (bytes, bytearray)):
        text = bytes(value).decode('latin-1', errors='ignore')
        nums = [float(n) for n in _NUM.findall(text)]
        if len(nums) >= 2:
            return nums
        import struct
        for width in (16, 8):
            if len(value) >= width:
                try:
                    return list(struct.unpack('<%df' % (width // 4), bytes(value)[-width:]))
                except Exception:
                    pass
        return []
    if isinstance(value, str):
        return [float(n) for n in _NUM.findall(value)]
    return []


class KeyedArchive:
    """Decodes an NSKeyedArchiver plist into plain Python dicts/lists, preserving object identity."""

    def __init__(self, plist):
        self.objects = plist['$objects']
        self.top = plist['$top']
        self._cache = {}

    def decode(self, value):
        if isinstance(value, plistlib.UID):
            idx = value.data
            if idx in self._cache:
                return self._cache[idx]
            raw = self.objects[idx]
            return self._decode_object(idx, raw)
        return self._decode_plain(value)

    def _decode_plain(self, value):
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if isinstance(value, dict):
            return {k: self.decode(v) for k, v in value.items()}
        return value

    def _decode_object(self, idx, raw):
        if raw == '$null':
            self._cache[idx] = None
            return None
        if not isinstance(raw, dict):
            self._cache[idx] = raw
            return raw

        class_name = None
        if '$class' in raw:
            cls_info = self.objects[raw['$class'].data]
            class_name = cls_info.get('$classname')

        if class_name in ('NSArray', 'NSMutableArray', 'NSSet', 'NSMutableSet'):
            out = []
            self._cache[idx] = out
            out.extend(self.decode(v) for v in raw.get('NS.objects', []))
            return out

        if class_name in ('NSDictionary', 'NSMutableDictionary'):
            out = {'__class__': class_name}
            self._cache[idx] = out
            keys = [self.decode(k) for k in raw.get('NS.keys', [])]
            vals = [self.decode(v) for v in raw.get('NS.objects', [])]
            for k, v in zip(keys, vals):
                out[k] = v
            return out

        if class_name in ('NSString', 'NSMutableString'):
            out = raw.get('NS.string', '')
            self._cache[idx] = out
            return out

        node = {'__class__': class_name}
        self._cache[idx] = node
        for k, v in raw.items():
            if k == '$class':
                continue
            node[k] = self.decode(v)
        return node


def find_nib(bundle_dir, name):
    """Find <name>.nib (a file, or a directory holding objects.nib/runtime.nib) anywhere in the bundle."""
    target = name if name.lower().endswith('.nib') else name + '.nib'
    for root, dirs, files in os.walk(bundle_dir):
        for f in files:
            if f.lower() == target.lower():
                return os.path.join(root, f)
        for d in dirs:
            if d.lower() == target.lower():
                return os.path.join(root, d)
    return None


def _load_plist(path):
    if os.path.isdir(path):
        candidates = sorted(os.listdir(path))
        ordered = [c for c in candidates if c.startswith('objects')] + \
                  [c for c in candidates if c == 'runtime.nib']
        for c in ordered:
            try:
                with open(os.path.join(path, c), 'rb') as f:
                    return plistlib.load(f)
            except Exception:
                continue
        raise ValueError(f"no readable keyed archive inside nib directory {path}: {candidates}")
    with open(path, 'rb') as f:
        return plistlib.load(f)


class NibView:
    """One node of the emulator's view tree."""

    def __init__(self, node, class_name, custom_class):
        self.node = node
        self.class_name = class_name          # UIKit class (UIImageView, UIButton...)
        self.custom_class = custom_class      # app class name if UIClassSwapper, else None
        self.children = []
        self.parent = None
        self.guest = 0                        # guest object pointer, assigned by cpu.py
        self.frame = (0.0, 0.0, 0.0, 0.0)     # relative to parent: x, y, w, h
        self.hidden = bool(node.get('UIHidden', False))
        _a = node.get('UIAlpha')
        self.alpha = 1.0 if _a is None else float(_a)
        self.transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)   # CGAffineTransform a,b,c,d,tx,ty (from setTransform:)
        self.anchor = (0.5, 0.5)              # CALayer.anchorPoint; transforms pivot around it
        self.tag = int(node.get('UITag') or 0)
        self.on = bool(node.get('UIOn', False))   # UISwitch state
        self.on_set = False                   # True once the app (or a tap) has set the switch
        self.pres = None                      # in-flight animated values {'frame'|'alpha'|'transform': v} (animations.py)
        self.state_images = None              # {UIControlState: Surface} from setImage:/setBackgroundImage:forState:
        self.selected = False
        self.highlighted = False
        self.enabled = True
        self.font_size = 17.0
        self.text_align = 0                   # NSTextAlignment: 0 left, 1 center, 2 right
        self.anim_images = None               # UIImageView.animationImages (list of pygame.Surface)
        self.animating = False
        self.anim_finished = False
        self.anim_start = 0.0
        self.anim_duration = 0.0
        self.anim_repeat = 0                  # 0 = loop forever
        self.image_name = None
        self.image_override = None            # pygame.Surface set at runtime by the app
        self.text = None
        self.bg_color = None
        self.text_color = (255, 255, 255)
        self.custom_draw = False
        self.draw_surface = None              # what this view's own -drawRect: painted
        self.actions = []                     # [(event_mask, target_node, selector_name)]

    @property
    def v_frame(self):
        return self.pres['frame'] if self.pres and 'frame' in self.pres else self.frame

    @property
    def v_alpha(self):
        return self.pres['alpha'] if self.pres and 'alpha' in self.pres else self.alpha

    @property
    def v_transform(self):
        return self.pres['transform'] if self.pres and 'transform' in self.pres else self.transform

    def absolute_frame(self):
        x, y, w, h = self.frame
        p = self.parent
        while p is not None:
            x += p.frame[0]
            y += p.frame[1]
            p = p.parent
        return x, y, w, h

    @property
    def is_button(self):
        return self.class_name in ('UIButton', 'UIRoundedRectButton') or bool(self.actions)


def _color(value):
    if not isinstance(value, dict):
        return None
    raw = value.get('UIRGB')
    if raw is not None:
        nums = _numbers(raw)
        if len(nums) >= 3:
            scale = 255.0 if max(nums[:3]) <= 1.0 else 1.0
            return tuple(int(max(0, min(255, n * scale))) for n in nums[:3])
    white = value.get('UIWhite')
    if white is not None:
        nums = _numbers(white)
        if nums:
            g = int(max(0, min(255, nums[0] * 255.0)))
            return (g, g, g)
    return None


def _find_image_name(node, depth=0, seen=None):
    """Search a node (e.g. a UIButton's stateful content) for an image resource name."""
    if seen is None:
        seen = set()
    if depth > 6 or id(node) in seen:
        return None
    seen.add(id(node))
    if isinstance(node, dict):
        cls = node.get('__class__')
        if cls in ('UIImage', 'NSImage'):
            for key in ('UIResourceName', 'UIImageName', 'NSResourceName', 'UIFilename'):
                if isinstance(node.get(key), str):
                    return node[key]
        for k, v in node.items():
            if k in ('UISuperview', '__class__'):
                continue
            found = _find_image_name(v, depth + 1, seen)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_image_name(v, depth + 1, seen)
            if found:
                return found
    return None


class NibDocument:
    """A parsed nib: view tree, outlet connections, action connections, custom-class object nodes."""

    def __init__(self, path):
        self.path = path
        plist = _load_plist(path)
        self.archive = KeyedArchive(plist)
        top = self.archive.top
        self.objects_root = self.archive.decode(top.get('UINibObjectsKey'))
        self.connections = self.archive.decode(top.get('UINibConnectionsKey')) or []
        self.views = {}          # id(node) -> NibView
        self.roots = []          # top-level NibViews
        self.instances = []      # every node that becomes a guest object (dicts, incl. views)
        self._walk(self.objects_root, None, set())

    # --- tree building -------------------------------------------------
    @staticmethod
    def _class_info(node):
        cls = node.get('__class__')
        if cls == 'UIClassSwapper':
            return node.get('UIOriginalClassName') or 'UIView', node.get('UIClassName')
        return cls, None

    def _walk(self, node, parent_view, seen):
        if isinstance(node, list):
            for n in node:
                self._walk(n, parent_view, seen)
            return
        if not isinstance(node, dict) or id(node) in seen:
            return
        seen.add(id(node))

        cls, custom = self._class_info(node)
        if cls is None:
            return

        if cls == 'UIProxyObject':
            # File's Owner / First Responder proxies aren't instantiated, but the
            # root proxy's UIChildren holds the nib's real top-level objects.
            for key in ('UIChildren', 'UIObjectsKey'):
                if key in node:
                    self._walk(node[key], parent_view, seen)
            return

        is_view = cls in _VIEW_CLASSES or (custom and ('UIBounds' in node or 'UICenter' in node or 'UIFrame' in node))
        if is_view:
            view = self._make_view(node, cls, custom)
            view.parent = parent_view
            if parent_view is not None:
                parent_view.children.append(view)
            else:
                self.roots.append(view)
            self.views[id(node)] = view
            self.instances.append(node)
            for sub in node.get('UISubviews', []) or []:
                self._walk(sub, view, seen)
            return

        # Non-view object (app delegate, view controller, etc.): keep it as an instance
        # if it carries an app class, then keep searching its children.
        if custom or cls not in ('NSObject',):
            self.instances.append(node)
        for key in ('UIChildren', 'UIObjectsKey', 'NS.objects'):
            if key in node:
                self._walk(node[key], parent_view, seen)
        for key, value in node.items():
            if key in ('UISuperview', '__class__', 'UIChildren', 'UIObjectsKey', 'NS.objects'):
                continue
            if isinstance(value, (dict, list)):
                self._walk(value, parent_view, seen)

    def _make_view(self, node, cls, custom):
        view = NibView(node, cls, custom)
        bounds = _numbers(node.get('UIBounds'))
        center = _numbers(node.get('UICenter'))
        frame = _numbers(node.get('UIFrame'))
        if len(frame) >= 4:
            view.frame = tuple(frame[:4])
        elif len(bounds) >= 4 and len(center) >= 2:
            w, h = bounds[2], bounds[3]
            view.frame = (center[0] - w / 2.0, center[1] - h / 2.0, w, h)
        elif len(bounds) >= 4:
            view.frame = (bounds[0], bounds[1], bounds[2], bounds[3])
        view.bg_color = _color(node.get('UIBackgroundColor'))
        tc = _color(node.get('UITextColor'))
        if tc:
            view.text_color = tc
        text = node.get('UIText') or node.get('UITitle')
        if isinstance(text, str):
            view.text = text
        img = node.get('UIImage')
        name = _find_image_name(img) if img else None
        if not name and cls in ('UIButton', 'UIRoundedRectButton'):
            name = _find_image_name(node.get('UIButtonStatefulContent')) or _find_image_name(node)
        view.image_name = name
        return view

    # --- introspection for logging ------------------------------------
    def dump(self, logger=logging.info):
        def rec(v, indent):
            x, y, w, h = v.frame
            logger(f"[NIB] {'  ' * indent}{v.class_name}"
                   f"{'(' + v.custom_class + ')' if v.custom_class else ''} "
                   f"frame=({x:.0f},{y:.0f},{w:.0f},{h:.0f})"
                   f"{' image=' + v.image_name if v.image_name else ''}"
                   f"{' text=' + repr(v.text) if v.text else ''}"
                   f"{' hidden' if v.hidden else ''}")
            for c in v.children:
                rec(c, indent + 1)
        for r in self.roots:
            rec(r, 0)
        for c in self.connections:
            if isinstance(c, dict):
                logger(f"[NIB] connection {c.get('__class__')}: label={c.get('UILabel')!r} "
                       f"mask={c.get('UIEventMask')}")


class UIKitRenderer:
    """
    Composites the NIB view trees (and custom-drawn views) into one RGBA frame.

    Supports what the beer app needs: per-view alpha (inherited down the tree),
    UIImageView.animationImages playback (the foam/bubble pours), CGAffineTransform /
    CATransform3D-derived rotation + scale about the layer's anchorPoint (the tilting
    beer), and a drawn UISwitch. A transformed view is rendered together with its
    subviews into an offscreen group, then rotated/scaled and blitted.
    """
    MAX_GROUP = 2048

    def __init__(self, width, height, image_loader):
        self.width = width
        self.height = height
        self.load_image = image_loader
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._font = None
        self._small_font = None
        self._now = 0.0
        self._used_custom = False

    def _get_font(self):
        if self._font is None:
            self._font = pygame.font.SysFont(None, 22)
        return self._font

    @staticmethod
    def _readable(target, color, rect):
        """
        All text is black, like the stock iPhone OS UI, unless the pixels already painted behind it are genuinely
        dark - then it flips to white so it never disappears. The app's own text colour is ignored: apps written
        for a dark screen leave labels and button titles white, which is invisible on the white Settings screen.
        """
        try:
            area = rect.clip(target.get_rect())
            if area.width <= 0 or area.height <= 0:
                return color
            avg = pygame.transform.average_color(target, area)
        except Exception:
            return color
        if avg[3] < 200:                       # translucent backdrop (offscreen group): leave the colour alone
            return color
        lum = (0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2]) / 255.0
        return (0, 0, 0) if lum > 0.25 else (255, 255, 255)

    def _font_for(self, points):
        px = max(8, int(round(points * 1.3)))     # pygame's default font runs small next to a point size
        cache = self.__dict__.setdefault('_font_cache', {})
        if px not in cache:
            cache[px] = pygame.font.SysFont(None, px)
        return cache[px]

    def _get_small_font(self):
        if self._small_font is None:
            self._small_font = pygame.font.SysFont(None, 13)
        return self._small_font

    def compose(self, roots, custom_draw_surface, allow_fallback=True):
        self.surface.fill((0, 0, 0, 255))
        self._used_custom = False
        self._now = time.monotonic()
        for root in roots:
            self._draw_view(self.surface, root, custom_draw_surface, 0.0, 0.0, 1.0)
        if allow_fallback and not self._used_custom and custom_draw_surface is not None:
            # App drew a view that isn't in any nib (created in code): overlay it full-frame.
            self.surface.blit(custom_draw_surface, (0, 0))
        return self.surface

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _is_identity(t):
        return (abs(t[0] - 1) < 1e-4 and abs(t[1]) < 1e-4 and abs(t[2]) < 1e-4 and
                abs(t[3] - 1) < 1e-4 and abs(t[4]) < 1e-3 and abs(t[5]) < 1e-3)

    @staticmethod
    def _blit_alpha(target, surf, pos, alpha):
        if alpha >= 0.999:
            target.blit(surf, pos)
            return
        surf.set_alpha(max(0, min(255, int(alpha * 255))))
        try:
            target.blit(surf, pos)
        finally:
            surf.set_alpha(None)

    @staticmethod
    def _fill(target, rect, color, alpha):
        if alpha >= 0.999:
            pygame.draw.rect(target, color, rect)
        else:
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], max(0, min(255, int(alpha * 255)))))
            target.blit(s, rect.topleft)

    def _anim_frame(self, view, frames):
        n = len(frames)
        if view.anim_start is None:
            view.anim_start = self._now          # first frame this view is actually on screen
        duration = view.anim_duration if view.anim_duration > 0 else n / 30.0
        cycles = (self._now - view.anim_start) / duration
        if view.anim_repeat > 0 and cycles >= view.anim_repeat:
            view.animating = False
            view.anim_finished = True
            return view.image_override if view.image_override is not None else frames[-1]
        return frames[min(n - 1, int((cycles % 1.0) * n))]

    def _current_image(self, view):
        frames = view.anim_images
        img = view.image_override
        if frames:
            if view.animating:
                return self._anim_frame(view, frames)
            if img is None and view.anim_finished:
                return frames[-1]
        states = view.state_images
        if states:
            for flag, state in ((not view.enabled, 2), (view.selected, 4), (view.highlighted, 1)):
                if flag and state in states:
                    return states[state]
            if 0 in states:
                return states[0]
        if img is None and view.image_name:
            img = self.load_image(view.image_name)
        return img

    def _subtree_bounds(self, view):
        x0, y0, x1, y1 = 0.0, 0.0, max(1.0, view.v_frame[2]), max(1.0, view.v_frame[3])

        def rec(v, ox, oy):
            nonlocal x0, y0, x1, y1
            for c in v.children:
                if c.hidden or c.v_alpha <= 0.0:
                    continue
                cx, cy, cw, ch = c.v_frame
                x0, y0 = min(x0, ox + cx), min(y0, oy + cy)
                x1, y1 = max(x1, ox + cx + cw), max(y1, oy + cy + ch)
                rec(c, ox + cx, oy + cy)
        rec(view, 0.0, 0.0)
        return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)

    # ---- drawing -------------------------------------------------------
    def _draw_view(self, target, view, custom, ox, oy, parent_alpha):
        if view.hidden:
            return
        eff = parent_alpha * view.v_alpha
        if eff <= 0.0:
            return
        if self._is_identity(view.v_transform):
            self._draw_plain(target, view, custom, ox, oy, eff)
        else:
            self._draw_group(target, view, custom, ox, oy, eff)

    def _draw_group(self, target, view, custom, ox, oy, eff):
        x, y, w, h = view.v_frame
        a, b, c, d, tx, ty = view.v_transform
        bx0, by0, bx1, by1 = self._subtree_bounds(view)
        bw, bh = bx1 - bx0, by1 - by0
        sx, sy = math.hypot(a, b), math.hypot(c, d)
        if bw <= 0 or bh <= 0 or bw > self.MAX_GROUP or bh > self.MAX_GROUP or sx < 1e-4 or sy < 1e-4:
            if sx >= 1e-4 and sy >= 1e-4:
                self._draw_plain(target, view, custom, ox, oy, eff)
            return
        tmp = pygame.Surface((bw, bh), pygame.SRCALPHA)
        tmp.fill((0, 0, 0, 0))
        self._draw_plain(tmp, view, custom, -bx0 - x, -by0 - y, 1.0)

        surf = tmp
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            nw, nh = max(1, int(round(bw * sx))), max(1, int(round(bh * sy)))
            if nw > self.MAX_GROUP * 2 or nh > self.MAX_GROUP * 2:
                return
            surf = pygame.transform.smoothscale(surf, (nw, nh))
        angle = math.degrees(math.atan2(b, a))
        if abs(angle) > 0.01:
            surf = pygame.transform.rotate(surf, -angle)   # +angle is clockwise on screen (y-down)

        px, py = view.anchor[0] * w, view.anchor[1] * h    # pivot, in view-local points
        vx0, vy0 = bx0 + bw / 2.0 - px, by0 + bh / 2.0 - py   # pivot -> group centre, untransformed
        vx, vy = vx0 * a + vy0 * c, vx0 * b + vy0 * d         # row-vector convention, like CGAffineTransform
        cx, cy = ox + x + px + vx + tx, oy + y + py + vy + ty
        self._blit_alpha(target, surf, surf.get_rect(center=(int(round(cx)), int(round(cy)))).topleft, eff)

    def _draw_plain(self, target, view, custom, ox, oy, eff):
        x, y, w, h = view.v_frame
        rect = pygame.Rect(int(round(ox + x)), int(round(oy + y)), max(1, int(round(w))), max(1, int(round(h))))

        if view.bg_color is not None:
            self._fill(target, rect, view.bg_color, eff)

        img = self._current_image(view)
        if img is not None:
            if w <= 0 and h <= 0:                      # UIImageView built without a frame: take the image's size
                rect.size = img.get_size()
            if view.class_name in ('UIButton', 'UIRoundedRectButton') and img.get_width() < rect.width \
                    and img.get_height() < rect.height:
                self._blit_alpha(target, img, img.get_rect(center=rect.center).topleft, eff)
            elif img.get_size() == rect.size:
                self._blit_alpha(target, img, rect.topleft, eff)
            else:
                self._blit_alpha(target, pygame.transform.smoothscale(img, rect.size), rect.topleft, eff)

        if view.class_name == 'UISwitch':
            self._draw_switch(target, rect, view, eff)

        if view.text:
            font = self._font_for(view.font_size)
            color = self._readable(target, view.text_color, rect)
            lines = [font.render(t, True, color) for t in view.text.split('\n')]
            total_h = sum(l.get_height() for l in lines)
            ty = rect.centery - total_h // 2
            for label in lines:
                if view.is_button or view.text_align == 1:
                    lx = rect.centerx - label.get_width() // 2
                elif view.text_align == 2:
                    lx = rect.right - label.get_width()
                else:
                    lx = rect.x
                self._blit_alpha(target, label, (lx, ty), eff)
                ty += label.get_height()

        if view.custom_draw:
            painted = view.draw_surface if view.draw_surface is not None else custom
            if painted is not None:
                self._blit_alpha(target, painted, rect.topleft, eff)
                self._used_custom = True

        for child in view.children:
            self._draw_view(target, child, custom, ox + x, oy + y, eff)

    def _draw_switch(self, target, rect, view, eff):
        sw, sh = 79, 27
        s = pygame.Surface((sw, sh), pygame.SRCALPHA)
        track = (0, 122, 255) if view.on else (190, 190, 195)
        pygame.draw.rect(s, track + (255,), (0, 0, sw, sh), border_radius=sh // 2)
        knob_x = sw - sh // 2 - 1 if view.on else sh // 2 + 1
        pygame.draw.circle(s, (255, 255, 255, 255), (knob_x, sh // 2), sh // 2 - 2)
        txt = self._get_small_font().render('ON' if view.on else 'OFF', True, (255, 255, 255) if view.on else (110, 110, 115))
        s.blit(txt, ((10 if view.on else sw - txt.get_width() - 10), (sh - txt.get_height()) // 2))
        self._blit_alpha(target, s, s.get_rect(center=rect.center).topleft, eff)


def hit_test(roots, x, y):
    """
    Returns (interactive, any_view): the topmost visible button/action view
    containing (x, y) and the topmost visible view of any kind. Buttons win
    over big non-interactive images stacked above them, since UIImageViews
    have userInteractionEnabled=NO by default.

    The point is carried down the tree in each view's own coordinate space and run through the
    inverse of its transform (about its anchor point), exactly like UIKit's -hitTest:withEvent:.
    A view scaled to 0.01 - the way this app "hides" panels - therefore no longer swallows taps
    meant for what's underneath it.
    """
    best_any = None
    best_interactive = None

    def rec(view, px, py):
        nonlocal best_any, best_interactive
        if view.hidden or view.v_alpha < 0.01:
            return
        vx, vy, vw, vh = view.frame
        a, b, c, d, tx, ty = view.transform
        if not (abs(a - 1) < 1e-4 and abs(b) < 1e-4 and abs(c) < 1e-4 and abs(d - 1) < 1e-4
                and abs(tx) < 1e-3 and abs(ty) < 1e-3):
            det = a * d - b * c
            if abs(det) < 1e-9:
                return
            ax, ay = view.anchor
            pvx, pvy = vx + ax * vw, vy + ay * vh
            dx, dy = px - pvx - tx, py - pvy - ty
            px, py = pvx + (dx * d - dy * c) / det, pvy + (-dx * b + dy * a) / det
        if vx <= px < vx + vw and vy <= py < vy + vh:
            best_any = view
            if view.is_button or view.custom_class:
                best_interactive = view
        for child in view.children:
            rec(child, px - vx, py - vy)

    for r in roots:
        rec(r, x, y)
    return best_interactive, best_any


def dump_tree(roots):
    """Indented text snapshot of the view tree: what's on screen, what's animating, what can take a tap."""
    lines = []

    def rec(v, depth):
        x, y, w, h = v.frame
        px, py, pw, ph = v.v_frame
        t = v.transform
        flags = []
        if v.hidden:
            flags.append('HIDDEN')
        if v.alpha != 1.0 or v.v_alpha != v.alpha:
            flags.append(f'alpha={v.alpha:.2f}' + (f'(now {v.v_alpha:.2f})' if v.v_alpha != v.alpha else ''))
        if abs(t[0] - 1) > 1e-4 or abs(t[1]) > 1e-4 or abs(t[4]) > 1e-3 or abs(t[5]) > 1e-3:
            flags.append(f'xform=rot{math.degrees(math.atan2(t[1], t[0])):.0f} scale{math.hypot(t[0], t[1]):.2f}')
        if v.pres:
            flags.append('ANIMATING')
        if v.is_button or v.custom_class:
            flags.append('TAKES-TOUCH')
        if v.actions:
            flags.append('actions=' + ','.join(f'{sel}' for _m, _t, sel in v.actions))
        if v.image_override is not None or v.image_name:
            flags.append(f'img={v.image_name or "surface"}')
        if v.text:
            flags.append(f'text={v.text[:20]!r}')
        if getattr(v, 'guest', 0):
            flags.append(f'0x{v.guest:08X}')
        name = v.class_name + (f'({v.custom_class})' if v.custom_class else '')
        lines.append(f"{'  ' * depth}{name} ({x:.0f},{y:.0f},{w:.0f},{h:.0f}) {' '.join(flags)}")
        for c in v.children:
            rec(c, depth + 1)
    for r in roots:
        rec(r, 0)
    return '\n'.join(lines)