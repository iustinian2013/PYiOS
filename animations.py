# animations.py
"""
UIKit / Core Animation animation emulation.

Before this module every `[UIView beginAnimations:...] ... [UIView commitAnimations]` block
applied its changes instantly and only honoured the did-stop callback, via a plain timer. So
nothing ever moved: the splash never zoomed, panels popped, and the app's sequencing callbacks
fired against a UI that had already jumped to its end state.

What lives here
  AnimationEngine  - pure Python, no CPU/guest knowledge. Tweens a view's `frame`, `alpha` and
                     `transform` between two values over time and writes the in-flight value into
                     `view.pres` (the *presentation* layer; the renderer reads it, while `view.frame`
                     etc. stay the *model* value the app reads back). Implements UIView animation
                     blocks (duration, delay, curve, repeat, autoreverse, beginsFromCurrentState,
                     will-start / did-stop delegate callbacks with the correct `finished` flag and
                     context, cancellation when a property is overwritten mid-flight, nesting,
                     setAnimationsEnabled:) and CABasicAnimation on a layer (addAnimation:forKey:).
  AnimationBridge  - the guest-facing half: decodes the Objective-C calls (+beginAnimations:context:,
                     +setAnimationDuration:, CABasicAnimation setters, NSValue wrappers,
                     CAMediaTimingFunction, -[CALayer addAnimation:forKey:] ...) into engine calls and
                     turns engine callbacks into queued guest messages.

Easing uses the real Core Animation cubic Béziers (ease-in-out = 0.42,0,0.58,1 etc.).
Transforms are interpolated as rotation / scale / translation, not matrix entries, so a rotation
sweeps through an arc instead of shrinking through zero.
"""
import logging
import math
import struct
import time

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

# UIViewAnimationCurve -> Bézier control points. None = linear.
CURVES = {0: (0.42, 0.0, 0.58, 1.0),   # EaseInOut
          1: (0.42, 0.0, 1.0, 1.0),    # EaseIn
          2: (0.0, 0.0, 0.58, 1.0),    # EaseOut
          3: None}                     # Linear


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------
def decompose(t):
    """CGAffineTransform (a,b,c,d,tx,ty) -> (angle, sx, sy, tx, ty)."""
    a, b, c, d, tx, ty = t
    sx = math.hypot(a, b)
    if sx < 1e-9:
        return 0.0, 0.0, math.hypot(c, d), tx, ty
    sy = (a * d - b * c) / sx
    return math.atan2(b, a), sx, sy, tx, ty


def compose(angle, sx, sy, tx, ty):
    c, s = math.cos(angle), math.sin(angle)
    return (sx * c, sx * s, -sy * s, sy * c, tx, ty)


def _bezier_ease(points, x):
    """y for a given x on the unit cubic Bézier (0,0)(x1,y1)(x2,y2)(1,1)."""
    if points is None:
        return x
    x1, y1, x2, y2 = points
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def bx(t):
        return 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3

    def by(t):
        return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3
    lo, hi = 0.0, 1.0
    t = x
    for _ in range(40):                      # bisection is plenty at 40 steps and can't diverge
        v = bx(t)
        if abs(v - x) < 1e-6:
            break
        if v < x:
            lo = t
        else:
            hi = t
        t = (lo + hi) / 2.0
    return by(t)


def ease(curve, x):
    pts = CURVES.get(curve) if not isinstance(curve, tuple) else curve
    return _bezier_ease(pts, x)


def interpolate(prop, a, b, p):
    if prop == 'alpha':
        return a + (b - a) * p
    if prop == 'frame':
        return tuple(a[i] + (b[i] - a[i]) * p for i in range(4))
    if prop == 'transform':
        da, db = decompose(a), decompose(b)
        return compose(*[da[i] + (db[i] - da[i]) * p for i in range(5)])
    return b if p >= 1.0 else a


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, id_ptr, context):
        self.id_ptr = id_ptr
        self.context = context
        self.duration = 0.2          # UIKit's default
        self.delay = 0.0
        self.curve = 0
        self.repeat = 1.0
        self.autoreverse = False
        self.from_current = False
        self.delegate = 0
        self.will_start = None
        self.did_stop = None
        self.anims = []
        self.cancelled = False
        self.committed = False
        self.started = False
        self.fired = False
        self.t0 = None
        self.end = None


class _Anim:
    def __init__(self, view, prop, frm, to, kind):
        self.view, self.prop, self.frm, self.to, self.kind = view, prop, frm, to, kind
        self.t0 = None
        self.total = 0.0
        self.duration = 0.0
        self.curve = 0
        self.autoreverse = False
        self.block = None            # UIView block that owns it (kind 'ui')
        self.key = None              # layer animation key (kind 'ca')
        self.hold = False            # CA: keep showing the end value after finishing (removedOnCompletion = NO)
        self.delegate = 0
        self.obj = 0                 # guest pointer of the CAAnimation (for animationDidStop:finished:)


class AnimationEngine:
    """fire(kind, **payload) is called for: 'ui_start' / 'ui_stop' (block=, finished=) and 'ca_stop' (anim=, finished=)."""

    def __init__(self, fire, clock=time.monotonic):
        self.fire = fire
        self.clock = clock
        self.block = None
        self.enabled = True
        self._stack = []
        self.anims = []
        self.blocks = []
        self.held = {}               # (id(view), key) -> (view, prop): CA end values kept on screen

    # ---- presentation helpers --------------------------------------------
    @staticmethod
    def presentation(view, prop):
        pres = getattr(view, 'pres', None)
        if pres and prop in pres:
            return pres[prop]
        return getattr(view, prop)

    @staticmethod
    def _set_pres(view, prop, value):
        if getattr(view, 'pres', None) is None:
            view.pres = {}
        view.pres[prop] = value

    @staticmethod
    def _clear_pres(view, prop):
        pres = getattr(view, 'pres', None)
        if pres:
            pres.pop(prop, None)
            if not pres:
                view.pres = None

    def is_animating(self, view):
        return any(a.view is view and a.t0 is not None for a in self.anims)

    # ---- UIView animation blocks -----------------------------------------
    def begin_block(self, id_ptr=0, context=0):
        if self.block is not None:
            self._stack.append(self.block)
        self.block = _Block(id_ptr, context)
        return self.block

    def commit_block(self):
        blk = self.block
        self.block = self._stack.pop() if self._stack else None
        if blk is None:
            return None
        now = self.clock()
        cycles = blk.repeat if blk.repeat > 0 else 1.0
        total = blk.duration * cycles * (2.0 if blk.autoreverse else 1.0)
        blk.committed = True
        blk.t0 = now + blk.delay
        blk.end = blk.t0 + total
        for a in blk.anims:
            a.t0, a.total, a.duration = blk.t0, total, blk.duration
        self.blocks.append(blk)
        return blk

    def set_view_prop(self, view, prop, value):
        """Every animatable setter funnels through here (frame / alpha / transform)."""
        blk = self.block
        old_model = getattr(view, prop)
        current = self.presentation(view, prop)
        animate = (self.enabled and blk is not None and (blk.duration > 0 or blk.delay > 0)
                   and value != old_model)
        self._cancel(view, prop, 'ui')                      # a new value interrupts anything in flight
        if not animate:
            setattr(view, prop, value)
            self._clear_pres(view, prop)
            return
        frm = current if blk.from_current else old_model
        setattr(view, prop, value)                          # model value is final immediately, like UIKit
        anim = _Anim(view, prop, frm, value, 'ui')
        anim.block, anim.curve, anim.autoreverse = blk, blk.curve, blk.autoreverse
        blk.anims.append(anim)
        self.anims.append(anim)
        self._set_pres(view, prop, frm)                     # no flash of the end state before commit

    def _cancel(self, view, prop, kind, key=None, notify_ca=True):
        for a in [a for a in self.anims if a.view is view and a.prop == prop and a.kind == kind
                  and (key is None or a.key == key)]:
            self.anims.remove(a)
            if kind == 'ui':
                blk = a.block
                if a in blk.anims:
                    blk.anims.remove(a)
                blk.cancelled = True
                if blk.committed and not blk.anims and not blk.fired:
                    self._fire_block(blk, finished=False)
            elif notify_ca and a.delegate:
                self.fire('ca_stop', anim=a, finished=False)

    def _fire_block(self, blk, finished):
        blk.fired = True
        if blk in self.blocks:
            self.blocks.remove(blk)
        if blk.delegate and blk.did_stop:
            self.fire('ui_stop', block=blk, finished=finished)

    # ---- CABasicAnimation -------------------------------------------------
    def add_layer_animation(self, view, key, spec):
        """spec: keypath, from/to/by (number, tuple or None), duration, repeat, autoreverse, delay, curve, hold, delegate, obj."""
        ends = self._ca_endpoints(view, spec)
        if ends is None:
            logging.warning(f"[CAAnimation] unsupported keyPath {spec.get('keypath')!r} - ignored")
            return False
        prop, frm, to = ends
        if key is not None:
            for a in [a for a in self.anims if a.view is view and a.kind == 'ca' and a.key == key]:
                self._cancel(view, a.prop, 'ca', key)
        anim = _Anim(view, prop, frm, to, 'ca')
        anim.key = key
        anim.duration = max(0.0, spec.get('duration', 0.25))
        cycles = spec.get('repeat') or 1.0
        anim.total = anim.duration * cycles * (2.0 if spec.get('autoreverse') else 1.0)
        anim.curve = spec.get('curve', 3)
        anim.autoreverse = bool(spec.get('autoreverse'))
        anim.hold = bool(spec.get('hold'))
        anim.delegate = spec.get('delegate', 0)
        anim.obj = spec.get('obj', 0)
        anim.t0 = self.clock() + max(0.0, spec.get('delay', 0.0))
        self.anims.append(anim)
        self._set_pres(view, prop, frm)
        return True

    def remove_layer_animation(self, view, key):
        for hk, (hv, hp) in list(self.held.items()):
            if hv is view and (key is None or hk[1] == key):
                del self.held[hk]
                self._clear_pres(hv, hp)
        for a in [a for a in self.anims if a.view is view and a.kind == 'ca' and (key is None or a.key == key)]:
            self._cancel(view, a.prop, 'ca', a.key)
            self._clear_pres(view, a.prop)

    def has_layer_animation(self, view, key):
        return any(a.view is view and a.kind == 'ca' and a.key == key for a in self.anims) \
            or (id(view), key) in self.held

    @staticmethod
    def _ca_endpoints(view, spec):
        kp = spec.get('keypath') or ''
        frm, to, by = spec.get('from'), spec.get('to'), spec.get('by')
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)

        def pick(base):
            f = frm if num(frm) else base
            t = to if num(to) else (f + by if num(by) else base)
            return f, t

        if kp == 'opacity':
            f, t = pick(view.alpha)
            return 'alpha', f, t
        if kp in ('transform.rotation', 'transform.rotation.z', 'transform.scale', 'transform.scale.x',
                  'transform.scale.y', 'transform.translation.x', 'transform.translation.y'):
            ang, sx, sy, tx, ty = decompose(view.transform)
            comp = {'transform.rotation': 0, 'transform.rotation.z': 0, 'transform.scale': 1,
                    'transform.scale.x': 1, 'transform.scale.y': 2, 'transform.translation.x': 3,
                    'transform.translation.y': 4}[kp]
            base = (ang, sx, sy, tx, ty)
            f, t = pick(base[comp])

            def build(v):
                vals = list(base)
                vals[comp] = v
                if kp == 'transform.scale':
                    vals[2] = v
                return compose(*vals)
            return 'transform', build(f), build(t)
        if kp == 'transform':
            f = frm if isinstance(frm, tuple) and len(frm) == 6 else view.transform
            t = to if isinstance(to, tuple) and len(to) == 6 else view.transform
            return 'transform', f, t
        if kp in ('position', 'position.x', 'position.y'):
            x, y, w, h = view.frame
            ax, ay = getattr(view, 'anchor', (0.5, 0.5))
            px, py = x + ax * w, y + ay * h

            def to_frame(px_, py_):
                return (px_ - ax * w, py_ - ay * h, w, h)
            if kp == 'position':
                f = frm if isinstance(frm, tuple) and len(frm) == 2 else (px, py)
                t = to if isinstance(to, tuple) and len(to) == 2 else (px, py)
                return 'frame', to_frame(*f), to_frame(*t)
            if kp == 'position.x':
                f, t = pick(px)
                return 'frame', to_frame(f, py), to_frame(t, py)
            f, t = pick(py)
            return 'frame', to_frame(px, f), to_frame(px, t)
        return None

    # ---- per-frame update -------------------------------------------------
    def tick(self, now=None):
        now = self.clock() if now is None else now
        for a in list(self.anims):
            if a.t0 is None:                                 # UI anim whose block hasn't been committed yet
                continue
            el = now - a.t0
            if el < 0:
                continue                                      # still in the delay
            if a.kind == 'ui' and not a.block.started:
                a.block.started = True
                if a.block.delegate and a.block.will_start:
                    self.fire('ui_start', block=a.block)
            if el >= a.total:
                self.anims.remove(a)
                if a.kind == 'ui':
                    if a in a.block.anims:
                        a.block.anims.remove(a)
                    self._clear_pres(a.view, a.prop)           # model already holds the final value
                else:
                    if a.hold:
                        self._set_pres(a.view, a.prop, a.to)
                        self.held[(id(a.view), a.key)] = (a.view, a.prop)
                    else:
                        self._clear_pres(a.view, a.prop)
                    if a.delegate:
                        self.fire('ca_stop', anim=a, finished=True)
                continue
            u = el / a.duration if a.duration > 0 else 1.0
            if a.autoreverse:
                ph = u % 2.0
                lin = ph if ph <= 1.0 else 2.0 - ph
            else:
                lin = u % 1.0
            self._set_pres(a.view, a.prop, interpolate(a.prop, a.frm, a.to, ease(a.curve, lin)))
        for blk in list(self.blocks):
            if not blk.started and blk.t0 is not None and now >= blk.t0:
                blk.started = True
                if blk.delegate and blk.will_start:
                    self.fire('ui_start', block=blk)
            if blk.end is not None and now >= blk.end and not blk.fired:
                self._fire_block(blk, finished=not blk.cancelled)


# ---------------------------------------------------------------------------
# guest-facing half
# ---------------------------------------------------------------------------
class AnimationBridge:
    """Decodes Objective-C messages about animation and drives an AnimationEngine."""

    TIMING_NAMES = (('linear', 3), ('easein', 1), ('easeout', 2), ('easeineaseout', 0), ('default', 0))

    def __init__(self, cpu):
        self.cpu = cpu
        self.engine = AnimationEngine(self._fire)
        self.specs = {}      # CAAnimation obj -> spec dict
        self.values = {}     # NSValue obj -> ('point', (x, y)) | ('affine', t6) | ('raw', None)
        self.timing = {}     # CAMediaTimingFunction obj -> curve
        self.layer_keys = {}  # (view id, key) -> animation obj, for -animationForKey:
        self._auto_key = 0

    # ---- engine -> guest -------------------------------------------------
    def _fire(self, kind, **kw):
        cpu = self.cpu
        if kind == 'ui_stop':
            blk = kw['block']
            flag = cpu.objc_alloc(0)
            cpu.numbers[flag] = 1 if kw['finished'] else 0       # the NSNumber `finished` argument
            cpu._action_queue.append({'target': blk.delegate, 'sel': blk.did_stop, 'arg2': blk.id_ptr,
                                      'arg3': flag, 'stack': [blk.context]})
            logging.info(f"[Animation] block done ({'finished' if kw['finished'] else 'cancelled'}) -> {blk.did_stop}")
        elif kind == 'ui_start':
            blk = kw['block']
            cpu._action_queue.append({'target': blk.delegate, 'sel': blk.will_start, 'arg2': blk.id_ptr,
                                      'arg3': blk.context})
        elif kind == 'ca_stop':
            a = kw['anim']
            cpu._action_queue.append({'target': a.delegate, 'sel': 'animationDidStop:finished:', 'arg2': a.obj,
                                      'arg3': 1 if kw['finished'] else 0})

    def tick(self, now):
        self.engine.tick(now)

    # ---- helpers ----------------------------------------------------------
    def _value_of(self, ptr):
        cpu = self.cpu
        if ptr in cpu.numbers:
            return float(cpu.numbers[ptr])
        v = self.values.get(ptr)
        return v[1] if v else None

    def _curve_for(self, ptr):
        return self.timing.get(ptr, 3)

    # ---- message handling -------------------------------------------------
    def handle(self, uc, receiver, sel_name):
        """Returns True if handled (R0 already set)."""
        from unicorn import arm_const as A
        cpu = self.cpu
        eng = self.engine
        R0 = A.UC_ARM_REG_R0
        r2, r3 = uc.reg_read(A.UC_ARM_REG_R2), uc.reg_read(A.UC_ARM_REG_R3)
        sp = uc.reg_read(A.UC_ARM_REG_SP)
        is_class = receiver in cpu.classes_by_addr
        cls_addr = receiver if is_class else cpu.instance_isa.get(receiver)
        cls_name = cpu.classes_by_addr.get(cls_addr, {}).get('name') if cls_addr else None
        dbl = lambda lo, hi: struct.unpack('<d', struct.pack('<II', lo, hi))[0]

        # -------- UIView class-level animation block API --------
        if is_class:
            blk = eng.block
            if sel_name == 'beginAnimations:context:':
                eng.begin_block(r2, r3)
                return True
            if sel_name == 'commitAnimations':
                blk = eng.commit_block()
                if blk is not None:
                    logging.info(f"[Animation] block committed: {blk.duration:.2f}s delay {blk.delay:.2f}s "
                                 f"curve {blk.curve} repeat {blk.repeat:g}{' autoreverse' if blk.autoreverse else ''}, "
                                 f"{len(blk.anims)} animated propert{'y' if len(blk.anims) == 1 else 'ies'}"
                                 f"{' -> ' + blk.did_stop if blk.did_stop else ''}")
                return True
            if sel_name == 'setAnimationsEnabled:':
                eng.enabled = bool(r2 & 0xFF)
                return True
            if sel_name == 'areAnimationsEnabled':
                uc.reg_write(R0, 1 if eng.enabled else 0)
                return True
            if sel_name.startswith('setAnimation') and sel_name != 'setAnimationImages:' and cls_name not in (
                    'CABasicAnimation', 'CAAnimation'):
                if blk is not None:
                    if sel_name == 'setAnimationDuration:':
                        blk.duration = max(0.0, dbl(r2, r3))
                    elif sel_name == 'setAnimationDelay:':
                        blk.delay = max(0.0, dbl(r2, r3))
                    elif sel_name == 'setAnimationCurve:':
                        blk.curve = r2
                    elif sel_name == 'setAnimationRepeatCount:':
                        blk.repeat = struct.unpack('<f', struct.pack('<I', r2))[0]
                    elif sel_name == 'setAnimationRepeatAutoreverses:':
                        blk.autoreverse = bool(r2 & 0xFF)
                    elif sel_name == 'setAnimationBeginsFromCurrentState:':
                        blk.from_current = bool(r2 & 0xFF)
                    elif sel_name == 'setAnimationDelegate:':
                        blk.delegate = r2
                    elif sel_name == 'setAnimationDidStopSelector:':
                        blk.did_stop = cpu._cstring_at(r2)
                    elif sel_name == 'setAnimationWillStartSelector:':
                        blk.will_start = cpu._cstring_at(r2)
                return True           # setAnimationTransition:forView:cache: etc. - applied instantly

        # -------- CAMediaTimingFunction --------
        if is_class and cls_name == 'CAMediaTimingFunction':
            if sel_name == 'functionWithName:':
                name = (cpu._str_of(r2) or '').lower().replace('-', '').replace('_', '')
                curve = 0
                for key, c in self.TIMING_NAMES:
                    if key in name:
                        curve = c
                        if key == 'easeineaseout':
                            break
                obj = cpu.objc_alloc(0)
                self.timing[obj] = curve
                uc.reg_write(R0, obj)
                return True
            if sel_name == 'functionWithControlPoints::::':
                s0, s1 = cpu._stack_words(uc, sp, 2)
                obj = cpu.objc_alloc(0)
                self.timing[obj] = (cpu._f32(r2), cpu._f32(r3), cpu._f32(s0), cpu._f32(s1))
                uc.reg_write(R0, obj)
                return True

        # -------- NSValue wrappers (CAAnimation from/to/by values) --------
        if is_class and cls_name == 'NSValue':
            payload = None
            if sel_name == 'valueWithCGPoint:':
                payload = ('point', (cpu._f32(r2), cpu._f32(r3)))
            elif sel_name == 'valueWithCATransform3D:':
                m = [cpu._f32(w) for w in (r2, r3) + cpu._stack_words(uc, sp, 14)]
                payload = ('affine', (m[0], m[1], m[4], m[5], m[12], m[13]))
            elif sel_name == 'valueWithCGAffineTransform:':
                payload = ('affine', tuple(cpu._f32(w) for w in (r2, r3) + cpu._stack_words(uc, sp, 4)))
            elif sel_name in ('valueWithCGSize:', 'valueWithCGRect:'):
                payload = ('raw', None)
            if payload is not None:
                obj = cpu.objc_alloc(0)
                self.values[obj] = payload
                uc.reg_write(R0, obj)
                return True

        # -------- CABasicAnimation objects --------
        if is_class and cls_name in ('CABasicAnimation', 'CAAnimation', 'CAPropertyAnimation') \
                and sel_name in ('animationWithKeyPath:', 'animation'):
            obj = cpu.objc_alloc(0)
            self.specs[obj] = {'keypath': cpu._str_of(r2) if sel_name == 'animationWithKeyPath:' else None,
                               'duration': 0.25, 'curve': 3, 'obj': obj}
            uc.reg_write(R0, obj)
            return True
        if receiver in self.specs or (not is_class and cls_name in ('CABasicAnimation', 'CAAnimation')):
            spec = self.specs.setdefault(receiver, {'keypath': None, 'duration': 0.25, 'curve': 3, 'obj': receiver})
            ok = True
            if sel_name == 'init':
                pass
            elif sel_name == 'setKeyPath:':
                spec['keypath'] = cpu._str_of(r2)
            elif sel_name == 'setFromValue:':
                spec['from'] = self._value_of(r2)
            elif sel_name == 'setToValue:':
                spec['to'] = self._value_of(r2)
            elif sel_name == 'setByValue:':
                spec['by'] = self._value_of(r2)
            elif sel_name == 'setDuration:':
                spec['duration'] = dbl(r2, r3)
            elif sel_name == 'setRepeatCount:':
                spec['repeat'] = struct.unpack('<f', struct.pack('<I', r2))[0]
            elif sel_name == 'setAutoreverses:':
                spec['autoreverse'] = bool(r2 & 0xFF)
            elif sel_name == 'setBeginTime:':
                bt = dbl(r2, r3)
                spec['delay'] = max(0.0, bt - time.monotonic()) if bt > 1e3 else max(0.0, bt)
            elif sel_name == 'setRemovedOnCompletion:':
                spec['hold'] = not bool(r2 & 0xFF)
            elif sel_name == 'setDelegate:':
                spec['delegate'] = r2
            elif sel_name == 'setTimingFunction:':
                spec['curve'] = self._curve_for(r2)
            elif sel_name in ('setFillMode:', 'setCumulative:', 'setAdditive:'):
                pass
            else:
                ok = False
            if ok:
                uc.reg_write(R0, receiver)
                return True

        # -------- CALayer animation management --------
        owner = cpu._layer_owner.get(receiver)
        view = cpu.guest_views.get(owner) if owner is not None else None
        if view is not None:
            if sel_name == 'addAnimation:forKey:':
                spec = self.specs.get(r2)
                if spec is not None:
                    key = cpu._str_of(r3) if r3 else None
                    if key is None:
                        self._auto_key += 1
                        key = f'__anim{self._auto_key}'
                    if eng.add_layer_animation(view, key, dict(spec)):
                        self.layer_keys[(id(view), key)] = r2
                        logging.info(f"[CAAnimation] {spec.get('keypath')} on 0x{owner:08X} key={key!r} "
                                     f"{spec.get('duration', 0.25):.2f}s")
                return True
            if sel_name == 'removeAnimationForKey:':
                eng.remove_layer_animation(view, cpu._str_of(r2))
                return True
            if sel_name == 'removeAllAnimations':
                eng.remove_layer_animation(view, None)
                return True
            if sel_name == 'animationForKey:':
                key = cpu._str_of(r2)
                uc.reg_write(R0, self.layer_keys.get((id(view), key), 0) if eng.has_layer_animation(view, key) else 0)
                return True
        return False