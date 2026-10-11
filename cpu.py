# cpu.py

from accelerometer import Accelerometer
import struct
import os
import unicorn as uc
from unicorn import arm_const
import logging
import pygame
import math
import plistlib
import re
import time

import numpy as np

from animations import AnimationBridge
from openal import OpenALEmulator
from cgbi_png import is_cgbi_png, decode_cgbi_png
from nib_loader import NibDocument, NibView, find_nib, UIKitRenderer, hit_test

# ---- CATransform3D / CGAffineTransform math (row-vector convention, exactly like Core Animation) ----
_M3_IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _m3_mul(a, b):
    """a then b (CATransform3DConcat)."""
    return [sum(a[i * 4 + k] * b[k * 4 + j] for k in range(4)) for i in range(4) for j in range(4)]


def _m3_rotation(angle, x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return list(_M3_IDENTITY)
    x, y, z = x / n, y / n, z / n
    c, s_, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [t * x * x + c, t * x * y + z * s_, t * x * z - y * s_, 0.0,
            t * x * y - z * s_, t * y * y + c, t * y * z + x * s_, 0.0,
            t * x * z + y * s_, t * y * z - x * s_, t * z * z + c, 0.0,
            0.0, 0.0, 0.0, 1.0]


def _m3_scale(sx, sy, sz):
    m = list(_M3_IDENTITY)
    m[0], m[5], m[10] = sx, sy, sz
    return m


def _m3_translation(tx, ty, tz):
    m = list(_M3_IDENTITY)
    m[12], m[13], m[14] = tx, ty, tz
    return m


def _aff_mul(t1, t2):
    """t1 then t2 (CGAffineTransformConcat); t = (a, b, c, d, tx, ty)."""
    a1, b1, c1, d1, x1, y1 = t1
    a2, b2, c2, d2, x2, y2 = t2
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2, c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            x1 * a2 + y1 * c2 + x2, x1 * b2 + y1 * d2 + y2)


def _aff_from_m3(m):
    return (m[0], m[1], m[4], m[5], m[12], m[13])


def _m3_from_aff(t):
    m = list(_M3_IDENTITY)
    m[0], m[1], m[4], m[5], m[12], m[13] = t
    return m


class ARMCpuCore:
    def __init__(self, gpu_instance=None, bridge_instance=None, cg_instance=None):
        self.gpu = gpu_instance
        self.bridge = bridge_instance
        self.cg = cg_instance
        self.uc = uc.Uc(uc.UC_ARCH_ARM, uc.UC_MODE_ARM | uc.UC_MODE_THUMB)
        logging.info("[ARM Core] Unicorn Engine ARMv7 context successfully instantiated for iOS runtime.")

        self.stub_base = 0xFF000000
        self.uc.mem_map(self.stub_base, 0x20000, uc.UC_PROT_ALL)

        # Objective-C runtime state, populated by the loader once class
        # metadata has been parsed out of __objc_classlist.
        self.classes_by_addr = {}
        self.classes_by_name = {}
        self.objc_ready = False

        # Bump allocator backing +alloc. Real objc uses malloc and a real
        # ivar layout; we just hand out fixed-size zeroed blocks whose first
        # word is the isa pointer, which is all msgSend dispatch needs.
        self.heap_base = 0x70000000
        self.heap_size = 0x100000
        self.heap_next = self.heap_base
        self.uc.mem_map(self.heap_base, self.heap_size, uc.UC_PROT_ALL)
        self.instance_isa = {}  # object_addr -> class_addr
        self.view_hierarchy = []      # views the app passed to addSubview:
        self._uikit_objects = {}      # (receiver, selector) -> synthetic object
        self._uiapp_stage = 0         # UIApplicationMain lifecycle: 0=start, 1=post-init, 2=done
        self._delegate_class = None
        self._delegate_launch_sel = None
        self._delegate_obj = None
        self._draw_queue_index = 0
        self._views_bootstrapped = False
        self._pending_view_inits = []  # [(obj, imp), ...] real -initWithFrame: calls still owed
        self.audio = None              # set by main.py once IOSAudioEngine exists for the launched app
        self._cfurl_paths = {}         # synthetic CFURLRef handle -> resolved absolute file path
        self._next_cfurl_handle = 0x73000000

        # --- NIB / UIKit view layer ---
        self.nib_docs = []
        self.nib_roots = []            # NibView roots, drawn in order
        self.guest_views = {}          # guest ptr -> NibView
        self.node_guests = {}          # id(nib node) -> guest ptr
        self.vc_views = {}             # view controller guest -> its root view guest
        self._nib_loaded = False
        self._nib_delegate = None      # (guest, class_addr, launch_selector)
        self._action_queue = []        # pending calls into app code (IBActions, timers, touches...)
        self._dirty_views = []         # views the app called setNeedsDisplay on
        self._touch_down_view = None
        self.touch_obj = 0
        self.touch_pos = (0.0, 0.0)
        self.renderer = None
        self.timers = []
        self._timer_objs = {}
        self.accel = Accelerometer()
        self._call_log_counts = {}     # (class, selector) -> times logged, for _log_call()
        self._cg_log_left = 14         # CoreGraphics call lines shown at INFO before going DEBUG-only
        self._next_redraw = 0.0
        self._view_layer = {}          # view guest ptr -> its (stable) CALayer handle
        self._layer_owner = {}         # CALayer handle -> view guest ptr
        self.external_class_addrs = set()   # synthetic framework classes (UIImageView, UIButton...)
        self.color_objs = {}                # UIColor object -> (r,g,b) or None for clear
        self.anim_bridge = AnimationBridge(self)   # UIView animation blocks + CABasicAnimation (animations.py)
        self.anim = self.anim_bridge.engine        # tweens view.frame / alpha / transform into view.pres
        self.fonts = {}                     # UIFont obj -> point size
        self._cb_pending = None             # on_return callback of the guest call currently running
        self.index_paths = {}               # NSIndexPath obj -> (section, row)
        self.table_styles = {}              # UITableViewController obj -> UITableViewStyle
        self._cell_labels = {}              # (cell obj, 'textLabel'|'detailTextLabel') -> label obj
        self.openal = OpenALEmulator(self)  # OpenAL + AudioFile (the app's looping sound engine)
        self._drawing_view = 0
        self._touch_active = False
        self._touch_phase = 0
        self._touch_hit_view = None
        self.status_bar_hidden = False

        # --- minimal C runtime (malloc etc.) + unimplemented-symbol tracking ---
        self.cmalloc_base = 0x75000000
        self.cmalloc_size = 0x1000000
        self.cmalloc_next = self.cmalloc_base
        self.uc.mem_map(self.cmalloc_base, self.cmalloc_size, uc.UC_PROT_ALL)
        self.unimplemented_symbols = {}   # stub index -> symbol name (filled by the Mach-O loader)
        self._warned_unimplemented = set()
        self.app_resource_dir = None   # set by main.py once the app bundle path is known
        self._resource_index = None    # lazy: {lowercase filename: full path}, whole bundle tree
        self.image_objects = {}        # object_addr -> real pygame.Surface (loaded artwork)
        self._image_file_cache = {}    # filename -> pygame.Surface or None (load failure)

        # --- Foundation objects (all just guest pointers backed by Python values) ---
        self.strings = {}              # NSString obj -> str
        self._string_cptrs = {}        # str -> guest char* (for -UTF8String)
        self.arrays = {}               # NSArray/NSMutableArray obj -> [guest pointers]
        self.numbers = {}              # NSNumber obj -> int/float
        self.dates = {}                # NSDate obj -> unix time
        self.urls = {}                 # NSURL obj -> path/string
        self.defaults = {}             # NSUserDefaults key -> int/float/('obj', ptr)
        self._defaults_obj = 0
        self._defaults_path = None
        self._bundle_obj = 0
        self._trace = None             # EMU_TRACE=1 -> every framework message goes to trace.log
        self._unhandled_seen = set()   # (class, selector) pairs already reported by _log_unhandled
        
        self.uc.hook_add(uc.UC_HOOK_CODE, self._hook_code_execution)
        self.uc.hook_add(uc.UC_HOOK_INTR, self._hook_interrupt)
        self.uc.hook_add(
            uc.UC_HOOK_MEM_READ_UNMAPPED | 
            uc.UC_HOOK_MEM_WRITE_UNMAPPED | 
            uc.UC_HOOK_MEM_FETCH_UNMAPPED, 
            self._hook_unmapped_memory
        )
        self.setup_stubs()

    def setup_stubs(self):
        for i in range(512):
            stub_addr = self.stub_base + (i * 4)
            trap_code = struct.pack('<I', 0xE1200070 | i)
            self.uc.mem_write(stub_addr, trap_code)

    def setup_registers(self, entry_point, stack_pointer, app_name="App"):
        string_base = 0x60000000
        self.uc.mem_map(string_base, 0x1000, uc.UC_PROT_READ | uc.UC_PROT_WRITE)
        
        app_name_bytes = app_name.encode('utf-8', errors='ignore')[:255] + b'\x00'
        self.uc.mem_write(string_base, app_name_bytes)
        
        stack_data = struct.pack('<I I I I', 
            1,              # argc = 1
            string_base,    # argv[0] -> "iBeer"
            0,              # argv terminator
            0               # envp terminator
        )
        
        actual_sp = stack_pointer - len(stack_data) - 0x100
        self.uc.mem_write(actual_sp, stack_data)

        cpsr = self.uc.reg_read(arm_const.UC_ARM_REG_CPSR)
        
        if (entry_point & 1) or self._is_thumb(entry_point):
            entry_point |= 1
            cpsr |= (1 << 5)  # Set Thumb mode bit (T-bit)
        else:
            entry_point &= ~1
            cpsr &= ~(1 << 5) # Clear Thumb mode bit
            
        self.uc.reg_write(arm_const.UC_ARM_REG_CPSR, cpsr)
        self.uc.reg_write(arm_const.UC_ARM_REG_SP, actual_sp)
        self.uc.reg_write(arm_const.UC_ARM_REG_PC, entry_point & ~1)
        
        for reg in [arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R1, arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3]:
            self.uc.reg_write(reg, 0)
            
        logging.info(f"[ARM Core] Registers configured -> PC: 0x{entry_point & ~1:08X}, SP: 0x{actual_sp:08X}, Thumb: {bool(cpsr & (1 << 5))}")

    def install_objc_runtime(self, classes_by_addr, classes_by_name):
        self.classes_by_addr = classes_by_addr
        self.classes_by_name = classes_by_name
        self.objc_ready = bool(classes_by_addr)
        logging.info(f"[ObjC Runtime] Installed {len(classes_by_addr)} classes: "
                     f"{', '.join(sorted(c['name'] for c in classes_by_addr.values() if c['name'])[:12])}")

    def objc_alloc(self, class_addr):
        """Hand out a zeroed instance whose isa (first word) points at class_addr."""
        obj = self.heap_next
        real = self.classes_by_addr.get(class_addr, {}).get('inst_size', 0) if class_addr else 0
        inst_size = max(0x100, (real + 0x3F) & ~0x3F)  # real class instance size when known
        if obj + inst_size >= self.heap_base + self.heap_size:
            logging.warning("[ObjC Runtime] Instance heap exhausted; reusing base.")
            obj = self.heap_base
            self.heap_next = self.heap_base
        self.heap_next = obj + inst_size
        self.uc.mem_write(obj, b'\x00' * inst_size)
        self.uc.mem_write(obj, struct.pack('<I', class_addr))
        self.instance_isa[obj] = class_addr
        return obj

    def _class_of(self, receiver):
        """Resolve a receiver pointer to a class_t address (class object or instance)."""
        if receiver in self.classes_by_addr:
            return receiver  # receiver IS a class (a class-method style send)
        if receiver in self.instance_isa:
            return self.instance_isa[receiver]
        try:
            isa = struct.unpack('<I', bytes(self.uc.mem_read(receiver, 4)))[0]
            if isa in self.classes_by_addr:
                return isa
        except Exception:
            pass
        return None

    # UIKit selectors that must return a usable object rather than nil, or
    # the app's view-hierarchy setup silently collapses. These are framework
    # methods with no IMP in the binary, so there's nothing to dispatch to -
    # we just have to produce a plausible object identity.
    _UIKIT_FACTORY_SELECTORS = (
        "sharedApplication", "new", "view", "window",
        "mainScreen", "bounds", "frame", "layer", "currentDevice",
        "defaultCenter", "standardUserDefaults", "mainBundle",
    )

    def _inherits(self, class_addr, name, max_hops=12):
        """True if the class (or any superclass up the chain) is called `name`."""
        hops = 0
        while class_addr and class_addr in self.classes_by_addr and hops < max_hops:
            if self.classes_by_addr[class_addr].get('name') == name:
                return True
            class_addr = self.classes_by_addr[class_addr].get('super')
            hops += 1
        return False

    def _read_cfstring_arg(self, uc_engine):
        """
        Reads the actual characters of an NSString/@"literal" argument
        passed as the first real argument (r2) to a selector like
        imageNamed:. Compile-time string literals are __NSCFConstantString
        structs living in __cfstring: {isa, flags, char* data, uint32 length}
        on the 32-bit ABI - we just need the data pointer at offset 8.
        Returns None if the argument doesn't look like a valid string.
        """
        try:
            ns_string_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            if not ns_string_ptr:
                return None
            if ns_string_ptr in self.strings:
                return self.strings[ns_string_ptr]
            data_ptr = struct.unpack('<I', bytes(uc_engine.mem_read(ns_string_ptr + 8, 4)))[0]
            if not data_ptr:
                return None
            raw = bytes(uc_engine.mem_read(data_ptr, 256))
            return raw.split(b'\x00')[0].decode('utf-8', errors='ignore')
        except Exception as e:
            logging.warning(f"[UIImage] Failed to read imageNamed: argument string: {e}")
            return None

    def _build_resource_index(self):
        """
        Walk the ENTIRE extracted .app bundle tree (not just the folder
        the executable happens to sit in) and index every file by
        lowercase basename. Old pre-asset-catalog bundles routinely stash
        resources in subfolders (e.g. an Images/ or Resources/ dir, or a
        *.lproj localization folder), and the real iOS filesystem is
        case-sensitive while asset lookups on disk here shouldn't have to
        match case exactly - so a single flat os.path.join in one
        directory misses real, present files. Built once and cached.
        """
        index = {}
        if not self.app_resource_dir or not os.path.isdir(self.app_resource_dir):
            return index
        for root, _dirs, files in os.walk(self.app_resource_dir):
            for fname in files:
                key = fname.lower()
                full_path = os.path.join(root, fname)
                if key not in index:
                    index[key] = full_path
                else:
                    logging.debug(f"[UIImage] Duplicate basename '{fname}' in bundle - "
                                   f"keeping {index[key]}, ignoring {full_path}")
        logging.info(f"[UIImage] Indexed {len(index)} files across the whole app bundle "
                     f"under {self.app_resource_dir}")
        return index

    def _load_image_file(self, path):
        """
        Loads an image file, transparently handling Apple's CgBI-mangled
        PNGs (see cgbi_png.py) that standard libpng refuses outright.
        Checks the file's own magic bytes for CgBI FIRST and routes
        straight to our decoder when it is - trying pygame.image.load on
        a CgBI file first (then catching its failure) works, but SDL's
        libpng prints "libpng error: CgBI: unhandled critical chunk"
        directly to stderr before we ever see the exception, and this app
        alone loads 100+ CgBI PNGs, so that was the wall of noise in the
        log. Every other format still goes straight through pygame.
        """
        try:
            with open(path, 'rb') as f:
                raw = f.read()
        except Exception:
            return pygame.image.load(path).convert_alpha()  # let pygame's own error surface

        if is_cgbi_png(raw):
            width, height, pixels, channels = decode_cgbi_png(raw)
            mode = 'RGBA' if channels == 4 else 'RGB'
            surface = pygame.image.frombuffer(pixels, (width, height), mode)
            logging.info(f"[UIImage] Decoded CgBI PNG {os.path.basename(path)} ({width}x{height})")
            return surface.convert_alpha()

        return pygame.image.load(path).convert_alpha()

    def _load_app_image(self, filename):
        """
        Resolves a UIImage imageNamed: filename to a real file anywhere in
        the extracted app bundle and loads it. Real UIKit also considers
        @2x/@3x scale variants and searches asset catalogs; we try the
        most common naming patterns, matched case-insensitively against
        every file in the bundle tree, not just the executable's own
        directory.
        """
        if not filename or not self.app_resource_dir:
            return None
        if filename in self._image_file_cache:
            return self._image_file_cache[filename]

        if self._resource_index is None:
            self._resource_index = self._build_resource_index()

        base, ext = os.path.splitext(filename)
        if not ext:
            candidates = [f"{filename}.png", f"{filename}@2x.png", f"{filename}.jpg", f"{filename}.jpeg"]
        else:
            candidates = [filename, f"{base}@2x{ext}"]

        surface = None
        for candidate in candidates:
            path = self._resource_index.get(candidate.lower())
            if path:
                try:
                    surface = self._load_image_file(path)
                    logging.debug(f"[UIImage] Matched '{filename}' -> {path}")
                    break
                except Exception as e:
                    logging.warning(f"[UIImage] Found {path} but failed to load it: {e}")

        self._image_file_cache[filename] = surface
        return surface

    def _uikit_shim(self, uc_engine, receiver, sel_name):
        """
        Handle framework selectors we can't dispatch to real code.
        Returns True if handled (caller returns via normal stub epilogue).
        """
        R0, R2, R3 = arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3
        r2 = uc_engine.reg_read(R2)
        r3 = uc_engine.reg_read(R3)
        view = self.guest_views.get(receiver)

        if self.anim_bridge.handle(uc_engine, receiver, sel_name):
            return True

        if self._foundation_message(uc_engine, receiver, sel_name):
            return True

        if self._view_message(uc_engine, receiver, sel_name):
            return True

        if sel_name in ('setNeedsDisplay', 'setNeedsDisplayInRect:'):
            if receiver not in self._dirty_views:
                self._dirty_views.append(receiver)
            uc_engine.reg_write(R0, receiver)
            return True

        if sel_name == 'view' and receiver in self.vc_views:
            uc_engine.reg_write(R0, self.vc_views[receiver])
            return True

        # --- accelerometer ---
        if sel_name == 'sharedAccelerometer':
            uc_engine.reg_write(R0, self.accel.shared_instance(self.objc_alloc))
            return True
        if self.accel.handle_message(receiver, sel_name, r2, r3):
            uc_engine.reg_write(R0, receiver)
            return True
        if self.accel.is_sample(receiver) and sel_name in ('x', 'y', 'z'):
            lo, hi = self.accel.component_regs(sel_name)
            uc_engine.reg_write(R0, lo)
            if hi is not None:
                uc_engine.reg_write(arm_const.UC_ARM_REG_R1, hi)
            return True

        # --- touches ---
        if receiver and receiver == self.touch_obj:
            if sel_name in ('anyObject', 'allObjects', 'objectEnumerator', 'allTouches', 'touchesForView:',
                            'touchesForWindow:'):
                uc_engine.reg_write(R0, self.touch_obj)
                return True
            if sel_name in ('count', 'tapCount'):
                uc_engine.reg_write(R0, 1)
                return True
            if sel_name == 'phase':
                uc_engine.reg_write(R0, self._touch_phase)
                return True
            if sel_name in ('containsObject:', 'isMultipleTouchEnabled'):
                uc_engine.reg_write(R0, 1 if sel_name == 'containsObject:' else 0)
                return True

        # --- timers / delayed selectors ---
        if sel_name in ('scheduledTimerWithTimeInterval:target:selector:userInfo:repeats:',
                        'timerWithTimeInterval:target:selector:userInfo:repeats:'):
            sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
            interval = struct.unpack('<d', struct.pack('<II', r2, r3))[0]
            target, selptr, _userinfo, repeats = struct.unpack('<IIII', bytes(uc_engine.mem_read(sp, 16)))
            sel = self._cstring_at(selptr)
            obj = self.objc_alloc(0)
            if sel and target:
                timer = {'interval': max(0.005, interval), 'target': target, 'sel': sel,
                         'repeats': bool(repeats & 0xFF), 'active': True, 'obj': obj,
                         'send_timer': True, 'next': time.monotonic() + max(0.005, interval)}
                self.timers.append(timer)
                self._timer_objs[obj] = timer
                logging.info(f"[NSTimer] every {interval:.3f}s -> [0x{target:08X} {sel}] repeats={timer['repeats']}")
            uc_engine.reg_write(R0, obj)
            return True
        if sel_name == 'performSelector:withObject:afterDelay:':
            sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
            delay = struct.unpack('<d', bytes(uc_engine.mem_read(sp, 8)))[0]
            sel = self._cstring_at(r2)
            if sel:
                self.timers.append({'interval': delay, 'target': receiver, 'sel': sel, 'repeats': False,
                                    'active': True, 'obj': 0, 'send_timer': False, 'arg': r3,
                                    'next': time.monotonic() + max(0.0, delay)})
                logging.info(f"[NSObject] performSelector {sel} after {delay:.3f}s")
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name in ('performSelector:', 'performSelector:withObject:'):
            sel = self._cstring_at(r2)
            if sel:
                self._action_queue.append({'target': receiver, 'sel': sel, 'arg2': r3 if sel_name.endswith('Object:') else 0})
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name == 'invalidate' and receiver in self._timer_objs:
            self._timer_objs[receiver]['active'] = False
            uc_engine.reg_write(R0, receiver)
            return True

        if sel_name == "imageNamed:":
            filename = self._read_cfstring_arg(uc_engine)
            surface = self._load_app_image(filename) if filename else None
            obj = self.objc_alloc(0)
            if surface is not None:
                self.image_objects[obj] = surface
                seen = self.__dict__.setdefault('_image_names_logged', set())
                (logging.debug if filename in seen else logging.info)(
                    f"[UIImage] imageNamed:\"{filename}\" -> loaded real artwork {surface.get_size()}")
                seen.add(filename)
            else:
                logging.warning(f"[UIImage] imageNamed:\"{filename}\" -> file not found, using placeholder")
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, obj)
            return True

        if sel_name in ('view', 'tableView') and receiver not in self.classes_by_addr \
                and self._inherits(self._class_of(receiver), 'UITableViewController'):
            # A UITableViewController's view is a UITableView. Old iPhone OS tables are drawn on white (plain) -
            # and the app's Settings screen is one of these - so give it a full-screen white background.
            key = (receiver, 'view')
            if key not in self._uikit_objects:
                obj = self.objc_alloc(0)
                self._uikit_objects[key] = obj
                tv = self._get_view(obj)
                tv.class_name = 'UITableView'
                tv.bg_color = (255, 255, 255)
                tv.frame = (0.0, 0.0, 320.0, 480.0)
                tv.table_controller = receiver
                tv.table_style = self.table_styles.get(receiver, 0)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self._uikit_objects[key])
            return True

        if sel_name in self._UIKIT_FACTORY_SELECTORS:
            # Return a stable per-(receiver,selector) synthetic object so
            # repeated sends give consistent identity.
            key = (receiver, sel_name)
            if key not in self._uikit_objects:
                self._uikit_objects[key] = self.objc_alloc(receiver if receiver in self.classes_by_addr else 0)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self._uikit_objects[key])
            return True

        return False

    # ------------------------------------------------------------------
    # NIB loading / UIKit view layer / input / timers
    # ------------------------------------------------------------------
    def _f32(self, reg_value):
        return struct.unpack('<f', struct.pack('<I', reg_value & 0xFFFFFFFF))[0]

    def _f32_bits(self, value):
        return struct.unpack('<I', struct.pack('<f', float(value)))[0]

    def _cstring_at(self, ptr, limit=256):
        try:
            if not ptr:
                return None
            return bytes(self.uc.mem_read(ptr, limit)).split(b'\x00')[0].decode('utf-8', errors='ignore')
        except Exception:
            return None

    def _nsstring_at(self, ns_ptr):
        """Reads a compile-time NSString literal (__cfstring struct) into a Python str."""
        try:
            if not ns_ptr:
                return None
            if ns_ptr in self.strings:
                return self.strings[ns_ptr]
            data_ptr = struct.unpack('<I', bytes(self.uc.mem_read(ns_ptr + 8, 4)))[0]
            return self._cstring_at(data_ptr)
        except Exception:
            return None

    def _app_object(self):
        key = (0, 'sharedApplication')
        if key not in self._uikit_objects:
            self._uikit_objects[key] = self.objc_alloc(0)
        return self._uikit_objects[key]

    def _guest_for_nib_node(self, node, owner_guest):
        if not isinstance(node, dict):
            return 0
        if id(node) in self.node_guests:
            return self.node_guests[id(node)]
        if node.get('__class__') == 'UIProxyObject':
            ident = node.get('UIProxiedObjectIdentifier')
            if ident == 'IBFilesOwner':
                return owner_guest or 0
        return 0

    def _set_outlet(self, src, prop, dst):
        """Wire an IBOutlet: store dst into src's ivar named prop (or _prop)."""
        from objc_runtime import resolve_ivar_offset
        if not src or not dst or not prop:
            return False
        cls = self._class_of(src)
        if not cls:
            return False
        for name in (prop, '_' + prop):
            off = resolve_ivar_offset(self.classes_by_addr, cls, name)
            if off is not None:
                try:
                    self.uc.mem_write(src + off, struct.pack('<I', dst))
                    logging.info(f"[NIB] outlet {self.classes_by_addr[cls]['name']}.{name} = 0x{dst:08X}")
                    return True
                except Exception as e:
                    logging.warning(f"[NIB] failed writing outlet {name}: {e}")
        logging.debug(f"[NIB] outlet '{prop}' has no ivar on {self.classes_by_addr[cls]['name']} "
                      f"(framework-owned property)")
        return False

    def _load_nib(self, nib_name, owner_guest):
        from objc_runtime import resolve_method
        path = find_nib(self.app_resource_dir, nib_name)
        if not path:
            logging.warning(f"[NIB] '{nib_name}' not found anywhere in the app bundle")
            return None
        try:
            doc = NibDocument(path)
        except Exception as e:
            logging.error(f"[NIB] Could not parse {path}: {e} "
                          f"(non-keyed / classic nib formats aren't supported)", exc_info=True)
            return None
        logging.info(f"[NIB] Loaded {os.path.basename(path)}: {len(doc.views)} views, "
                     f"{len(doc.connections)} connections")
        doc.dump(logging.info)
        self.nib_docs.append(doc)

        # 1. Instantiate a guest object for every object in the nib.
        for node in doc.instances:
            if id(node) in self.node_guests:
                continue
            cls_name, custom = doc._class_info(node)
            class_addr = self.classes_by_name.get(custom, 0) if custom else 0
            guest = self.objc_alloc(class_addr)
            self.node_guests[id(node)] = guest
            view = doc.views.get(id(node))
            if view:
                view.guest = guest
                self.guest_views[guest] = view
                if custom and class_addr and resolve_method(self.classes_by_addr, class_addr, "drawRect:"):
                    view.custom_draw = True
                    self.view_hierarchy.append(guest)
                    logging.info(f"[NIB] {custom} implements drawRect: - will be drawn by app code")

        # 2. Wire connections: outlets -> ivars, actions -> buttons.
        for c in doc.connections:
            if not isinstance(c, dict):
                continue
            kind = c.get('__class__')
            label = c.get('UILabel')
            src_node, dst_node = c.get('UISource'), c.get('UIDestination')
            src = self._guest_for_nib_node(src_node, owner_guest)
            dst = self._guest_for_nib_node(dst_node, owner_guest)
            if kind == 'UIRuntimeOutletConnection':
                self._set_outlet(src, label, dst)
            elif kind == 'UIRuntimeEventConnection':
                view = doc.views.get(id(src_node)) if isinstance(src_node, dict) else None
                if view is not None and dst and label:
                    mask = int(c.get('UIEventMask') or 64)
                    view.actions.append((mask, dst, label))
                    logging.info(f"[NIB] action: {view.class_name} -> -[target {label}] (mask {mask})")

        self.nib_roots.extend(doc.roots)
        return doc

    def load_app_nibs(self):
        """Real UIApplicationMain loads NSMainNibFile from Info.plist before launching the delegate."""
        from objc_runtime import resolve_method
        if not self.app_resource_dir:
            return
        info = {}
        try:
            with open(os.path.join(self.app_resource_dir, 'Info.plist'), 'rb') as f:
                info = plistlib.load(f)
        except Exception as e:
            logging.warning(f"[NIB] Could not read Info.plist: {e}")
        self.status_bar_hidden = bool(info.get('UIStatusBarHidden', False))
        logging.info(f"[UIKit] UIStatusBarHidden = {self.status_bar_hidden} "
                     f"(applicationFrame origin y = {0 if self.status_bar_hidden else 20})")
        main_nib = info.get('NSMainNibFile') or info.get('NSMainNibFile~iphone')
        if not main_nib:
            logging.info("[NIB] Info.plist has no NSMainNibFile - app builds its UI in code.")
            return
        logging.info(f"[NIB] NSMainNibFile = {main_nib}")

        doc = self._load_nib(main_nib, self._app_object())
        if doc is None:
            return

        # Find the app delegate instance and any view controllers that name their own nib.
        for node in doc.instances:
            cls_name, custom = doc._class_info(node)
            if not custom:
                continue
            guest = self.node_guests.get(id(node))
            class_addr = self.classes_by_name.get(custom)
            if not guest or not class_addr:
                continue
            for sel in ("application:didFinishLaunchingWithOptions:", "applicationDidFinishLaunching:"):
                if self._nib_delegate is None and resolve_method(self.classes_by_addr, class_addr, sel):
                    self._nib_delegate = (guest, class_addr, sel)
                    logging.info(f"[NIB] App delegate from nib: {custom} ({sel})")
            vc_nib = node.get('UINibName')
            if isinstance(vc_nib, str) and vc_nib:
                logging.info(f"[NIB] View controller {custom} loads nib '{vc_nib}'")
                vc_doc = self._load_nib(vc_nib, guest)
                if vc_doc and vc_doc.roots:
                    self.vc_views[guest] = vc_doc.roots[0].guest
                    for sel in ("viewDidLoad",):
                        if resolve_method(self.classes_by_addr, class_addr, sel):
                            self._action_queue.append({'target': guest, 'sel': sel})
                    for sel in ("viewWillAppear:", "viewDidAppear:"):
                        if resolve_method(self.classes_by_addr, class_addr, sel):
                            self._action_queue.append({'target': guest, 'sel': sel, 'arg2': 0})

        if self._nib_delegate:
            guest, class_addr, _sel = self._nib_delegate
            if resolve_method(self.classes_by_addr, class_addr, "applicationDidBecomeActive:"):
                self._action_queue.append({'target': guest, 'sel': "applicationDidBecomeActive:",
                                           'arg2': self._app_object()})

    def _log_call(self, kind, cls_name, sel_name, imp):
        """
        Logs a dispatch into app code at INFO for the first few calls of each
        (class, selector), then DEBUG only. Accelerometer/timer-driven methods
        fire dozens of times a second; at INFO that is an endless wall of
        identical lines that buries everything else.
        """
        key = (cls_name, sel_name)
        n = self._call_log_counts.get(key, 0) + 1
        self._call_log_counts[key] = n
        msg = f"[{kind}] -[{cls_name} {sel_name}] -> IMP 0x{imp:08X}"
        if n <= 3:
            logging.info(msg)
        else:
            logging.debug(msg)

    # ---- running app code on demand (actions, timers, touches) ----
    def _drive_pending_actions(self, uc_engine):
        from objc_runtime import resolve_method
        cb, self._cb_pending = self._cb_pending, None
        if cb is not None:        # the previous guest call returned: hand its r0 to whoever asked (table datasource)
            cb(uc_engine.reg_read(arm_const.UC_ARM_REG_R0), uc_engine.reg_read(arm_const.UC_ARM_REG_R1))
        while self._action_queue:
            item = self._action_queue.pop(0)
            target, sel = item['target'], item['sel']
            cls = self._class_of(target) if target else None
            imp = resolve_method(self.classes_by_addr, cls, sel) if cls else None
            if not imp:
                logging.debug(f"[UIKit] no app implementation of {sel} on 0x{target:08X}; skipped")
                if item.get('on_return') is not None:
                    item['on_return'](0, 0)
                continue
            self._cb_pending = item.get('on_return')
            self._log_call('UIKit', self.classes_by_addr[cls]['name'], sel, imp)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, target)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R1, 0)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R2, item.get('arg2', 0))
            uc_engine.reg_write(arm_const.UC_ARM_REG_R3, item.get('arg3', 0))
            if item.get('stack'):         # 5th+ argument words live at the top of the stack
                words = item['stack']
                sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
                uc_engine.mem_write(sp, struct.pack(f'<{len(words)}I', *words))
            uc_engine.reg_write(arm_const.UC_ARM_REG_LR, self.stub_base + (15 * 4))
            uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
            return True
        return False

    def _custom_views_at(self, x, y):
        hits = []
        def rec(v):
            if v.hidden:
                return
            vx, vy, vw, vh = v.absolute_frame()
            if vx <= x < vx + vw and vy <= y < vy + vh and v.custom_class and v.guest:
                hits.append(v)
            for c in v.children:
                rec(c)
        for r in self.nib_roots:
            rec(r)
        return hits

    def queue_touch(self, phase, x, y):
        """Touch phases from touch.py: 'began' | 'moved' | 'ended' | 'cancelled'."""
        from objc_runtime import resolve_method
        if phase == 'began':
            self._touch_active = True
        elif not self._touch_active:
            return            # stray event with no touch in flight (e.g. the click that launched the app)
        elif phase in ('ended', 'cancelled'):
            self._touch_active = False
        self.touch_pos = (float(x), float(y))
        self._touch_phase = {'began': 0, 'moved': 1, 'ended': 3, 'cancelled': 4}[phase]
        if not self.touch_obj:
            self.touch_obj = self.objc_alloc(0)
        interactive, any_view = hit_test(self.nib_roots, x, y)

        if phase == 'began':
            self._touch_down_view = interactive
            self._touch_hit_view = any_view       # iOS delivers the whole gesture to the view first touched
            if interactive is not None:
                interactive.highlighted = interactive.class_name != 'UISwitch'
                for mask, target, sel in interactive.actions:
                    if mask & 1:   # UIControlEventTouchDown
                        self._action_queue.append({'target': target, 'sel': sel, 'arg2': interactive.guest})
        elif phase == 'ended':
            down, self._touch_down_view = self._touch_down_view, None
            if down is not None:
                down.highlighted = False
            if down is not None and down is interactive:
                if down.class_name == 'UISwitch':
                    down.on = not down.on          # a tap flips the switch, then UIControlEventValueChanged fires
                    down.on_set = True
                    for mask, target, sel in down.actions:
                        if mask & 4096:
                            self._action_queue.append({'target': target, 'sel': sel, 'arg2': down.guest})
                for mask, target, sel in down.actions:
                    if mask & 64:  # UIControlEventTouchUpInside
                        self._action_queue.append({'target': target, 'sel': sel, 'arg2': down.guest})
        elif phase == 'cancelled':
            if self._touch_down_view is not None:
                self._touch_down_view.highlighted = False
            self._touch_down_view = None

        # Raw touch delivery up the responder chain, starting at the view the gesture began on.
        sel_name = {'began': 'touchesBegan:withEvent:', 'moved': 'touchesMoved:withEvent:',
                    'ended': 'touchesEnded:withEvent:', 'cancelled': 'touchesCancelled:withEvent:'}[phase]
        v = self._touch_hit_view if phase != 'began' else any_view
        while v is not None:
            cls = self._class_of(v.guest) if v.guest else None
            if v.custom_class and cls and resolve_method(self.classes_by_addr, cls, sel_name):
                if phase == 'moved':   # coalesce: the app only needs the latest position
                    self._action_queue = [a for a in self._action_queue
                                          if not (a['sel'] == sel_name and a['target'] == v.guest)]
                self._action_queue.append({'target': v.guest, 'sel': sel_name,
                                           'arg2': self.touch_obj, 'arg3': self.touch_obj})
                break
            v = v.parent
        if phase in ('ended', 'cancelled'):
            self._touch_hit_view = None

        if phase == 'moved':
            logging.debug(f"[Input] touch moved ({x:.0f},{y:.0f})")
        else:
            logging.info(f"[Input] touch {phase} at ({x:.0f},{y:.0f}) -> "
                         f"{interactive.class_name + ' ' + str(interactive.image_name) if interactive else 'no interactive view'}")

    def _cg_log(self, msg):
        """First few CoreGraphics calls at INFO, then DEBUG only (drawRect: now runs continuously)."""
        if self._cg_log_left > 0:
            self._cg_log_left -= 1
            logging.info(msg)
        else:
            logging.debug(msg)

    def _request_redraw(self, now):
        """
        Stand-in for the app's own redraw trigger. -drawRect: geometry (the tilted
        beer surface) depends on state that -updateToInclinationInRadians: changes
        on every accelerometer sample, but nothing ever asked the QuartzView to
        draw again after its first pass - so the picture never followed the tilt.
        While the accelerometer is live, redraw every drawable view ~20x/s.
        """
        if not self.accel.delegate or now < self._next_redraw:
            return
        self._next_redraw = now + 0.05
        if self._dirty_views or self._draw_queue_index < len(self.view_hierarchy):
            return
        from objc_runtime import resolve_method
        for v in self.view_hierarchy:
            cls = self._class_of(v)
            if cls and resolve_method(self.classes_by_addr, cls, "drawRect:") and v not in self._dirty_views:
                self._dirty_views.append(v)

    def has_pending_work(self):
        """True while queued guest calls (timers, accelerometer, touches, redraws) are still waiting to run."""
        return bool(self._action_queue or self._dirty_views or self._draw_queue_index < len(self.view_hierarchy))

    def tick(self, now):
        """Called every frame by main.py: fire due timers and accelerometer updates."""
        for t in self.timers:
            if t['active'] and now >= t['next']:
                self._action_queue.append({'target': t['target'], 'sel': t['sel'],
                                           'arg2': t['obj'] if t['send_timer'] else t.get('arg', 0),
                                           'arg3': t.get('arg3', 0)})
                if t['repeats']:
                    t['next'] = max(t['next'] + t['interval'], now)
                else:
                    t['active'] = False
        self.timers = [t for t in self.timers if t['active']]
        self.anim_bridge.tick(now)
        self._tick_tables()
        self.accel.poll(now, self._action_queue, self.objc_alloc)
        self._request_redraw(now)

    def compose_frame(self, cg_surface):
        """One RGBA frame: nib views + the app's own drawRect: output, composited."""
        if not self.nib_roots:
            return cg_surface
        if self.renderer is None:
            self.renderer = UIKitRenderer(self.cg.width, self.cg.height, self._load_app_image)
        # The "overlay the raw CG surface full-frame" fallback is only for apps that draw with no custom view at
        # all. If the app has a custom-drawn view (the beer's QuartzView) that is currently detached - say the
        # fluid view was removed after the glass emptied - its last drawRect: output (slime.png over the whole
        # screen) is stale and must NOT be painted on top of the views that replaced it.
        has_custom = any(v.custom_draw for v in self.guest_views.values())
        return self.renderer.compose(self.nib_roots, cg_surface, allow_fallback=not has_custom)

    # ------------------------------------------------------------------
    # Foundation: NSString / NSArray / NSNumber / NSDate / NSUserDefaults / NSBundle / NSURL
    # ------------------------------------------------------------------
    _BENIGN_SELECTORS = frozenset({
        'init', 'retain', 'release', 'autorelease', 'dealloc', 'alloc', 'class', 'self', 'drain', 'copy',
        'mutableCopy', 'isKindOfClass:', 'respondsToSelector:', 'setNeedsLayout', 'setUserInteractionEnabled:',
        'setMultipleTouchEnabled:', 'setOpaque:', 'setClipsToBounds:', 'setContentMode:', 'setExclusiveTouch:',
        'setAutoresizingMask:', 'setAutoresizesSubviews:', 'initWithNibName:bundle:', 'superview',
        'setIdleTimerDisabled:', 'setContentVerticalAlignment:', 'setContentHorizontalAlignment:',
        'setAdjustsImageWhenDisabled:', 'setAdjustsImageWhenHighlighted:', 'CGImage', 'initWithURL:', 'loadRequest:',
    })
    _STRING_SELECTORS = frozenset({
        'UTF8String', 'cStringUsingEncoding:', 'fileSystemRepresentation', 'cString', 'length', 'intValue',
        'integerValue', 'floatValue', 'doubleValue', 'boolValue', 'isEqualToString:', 'stringByAppendingString:',
        'stringByAppendingPathComponent:', 'lastPathComponent', 'pathExtension', 'stringByDeletingPathExtension',
        'stringByDeletingLastPathComponent', 'hasPrefix:', 'hasSuffix:',
    })
    _FMT_RE = re.compile(r'%([-+ 0#]*)(\d+)?(?:\.(\d+))?(hh|h|ll|l|z|t|q)?([diouxXfFeEgGcs@%])')

    def _new_string(self, text):
        obj = self.objc_alloc(0)
        self.strings[obj] = text
        return obj

    def _str_of(self, ptr):
        return self._nsstring_at(ptr) if ptr else None

    def _cstr_ptr(self, text):
        ptr = self._string_cptrs.get(text)
        if ptr is None:
            raw = text.encode('utf-8', errors='ignore') + b'\x00'
            ptr = self._c_alloc(len(raw))
            self.uc.mem_write(ptr, raw)
            self._string_cptrs[text] = ptr
        return ptr

    def _ret_float(self, uc_engine, value):
        uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self._f32_bits(value))

    def _ret_double(self, uc_engine, value):
        lo, hi = struct.unpack('<II', struct.pack('<d', float(value)))
        uc_engine.reg_write(arm_const.UC_ARM_REG_R0, lo)
        uc_engine.reg_write(arm_const.UC_ARM_REG_R1, hi)

    def _arg_double(self, r2, r3):
        return struct.unpack('<d', struct.pack('<II', r2, r3))[0]

    def _collect_objects(self, uc_engine, first_words):
        """nil-terminated varargs: the first words ride in registers, the rest on the stack."""
        objs = []
        for w in first_words:
            if not w:
                return objs
            objs.append(w)
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        for i in range(4096):
            try:
                w = struct.unpack('<I', bytes(uc_engine.mem_read(sp + i * 4, 4)))[0]
            except Exception:
                break
            if w == 0:
                break
            objs.append(w)
        return objs

    def _format_string(self, uc_engine, fmt, first_slot_is_r3=True):
        """printf-style formatting with the ARM varargs layout (variadics start at r3, then the stack)."""
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        state = {'slot': 3}

        def word(slot):
            if slot == 3:
                return uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
            return struct.unpack('<I', bytes(uc_engine.mem_read(sp + (slot - 4) * 4, 4)))[0]

        def next_word():
            w = word(state['slot'])
            state['slot'] += 1
            return w

        def next_pair():                      # 64-bit values are 8-byte aligned
            if state['slot'] % 2:
                state['slot'] += 1
            lo, hi = word(state['slot']), word(state['slot'] + 1)
            state['slot'] += 2
            return lo, hi

        def sub(m):
            flags, width, prec, length, conv = m.groups()
            if conv == '%':
                return '%'
            spec = '%' + flags + (width or '') + ('.' + prec if prec is not None else '')
            try:
                if conv in 'fFeEgG':
                    lo, hi = next_pair()
                    return (spec + conv) % struct.unpack('<d', struct.pack('<II', lo, hi))[0]
                if conv == 'c':
                    return (spec + 's') % chr(next_word() & 0xFF)
                if conv == 's':
                    return (spec + 's') % (self._cstring_at(next_word()) or '(null)')
                if conv == '@':
                    ptr = next_word()
                    if not ptr:
                        text = '(null)'
                    elif ptr in self.numbers:
                        text = str(self.numbers[ptr])
                    else:
                        text = self._str_of(ptr) or f'<object 0x{ptr:08X}>'
                    return (spec + 's') % text
                if length in ('ll', 'q'):
                    lo, hi = next_pair()
                    val = lo | (hi << 32)
                    if conv in 'di' and val & (1 << 63):
                        val -= 1 << 64
                else:
                    val = next_word()
                    if conv in 'di' and val & 0x80000000:
                        val -= 0x100000000
                return (spec + ('d' if conv in 'di' else conv)) % val
            except Exception:
                return ''
        return self._FMT_RE.sub(sub, fmt)

    def _resolve_resource_path(self, path):
        """Device paths (/var/mobile/.../x.caf) don't exist here: re-resolve by basename inside the bundle."""
        if not path:
            return None
        if os.path.isfile(path):
            return path
        if self._resource_index is None:
            self._resource_index = self._build_resource_index()
        return self._resource_index.get(os.path.basename(path).lower())

    def _foundation_message(self, uc_engine, receiver, sel_name):
        """Framework behaviour for the Foundation classes apps lean on. Returns True if handled (R0 set)."""
        R0 = arm_const.UC_ARM_REG_R0
        r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
        r3 = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        is_class = receiver in self.classes_by_addr
        cls_addr = receiver if is_class else self.instance_isa.get(receiver)
        cls_name = self.classes_by_addr.get(cls_addr, {}).get('name') if cls_addr else None
        signed = lambda v: v - 0x100000000 if v & 0x80000000 else v

        # ---------------- NSArray / NSMutableArray ----------------
        if is_class and cls_name in ('NSArray', 'NSMutableArray'):
            if sel_name in ('array', 'arrayWithCapacity:'):
                obj = self.objc_alloc(0)
                self.arrays[obj] = []
            elif sel_name == 'arrayWithObject:':
                obj = self.objc_alloc(0)
                self.arrays[obj] = [r2]
            elif sel_name == 'arrayWithObjects:':
                obj = self.objc_alloc(0)
                self.arrays[obj] = self._collect_objects(uc_engine, [r2, r3])
            elif sel_name == 'arrayWithArray:':
                obj = self.objc_alloc(0)
                self.arrays[obj] = list(self.arrays.get(r2, []))
            else:
                obj = None
            if obj is not None:
                uc_engine.reg_write(R0, obj)
                return True
        if receiver in self.arrays or (not is_class and cls_name in ('NSArray', 'NSMutableArray')):
            items = self.arrays.setdefault(receiver, [])
            handled = True
            if sel_name in ('init', 'initWithCapacity:'):
                uc_engine.reg_write(R0, receiver)
            elif sel_name == 'initWithObjects:':
                items[:] = self._collect_objects(uc_engine, [r2, r3])
                uc_engine.reg_write(R0, receiver)
            elif sel_name == 'initWithArray:':
                items[:] = list(self.arrays.get(r2, []))
                uc_engine.reg_write(R0, receiver)
            elif sel_name == 'addObject:':
                items.append(r2)
            elif sel_name == 'addObjectsFromArray:':
                items.extend(self.arrays.get(r2, []))
            elif sel_name == 'insertObject:atIndex:':
                items.insert(min(max(0, r3), len(items)), r2)
            elif sel_name == 'replaceObjectAtIndex:withObject:':
                if r2 < len(items):
                    items[r2] = r3
            elif sel_name == 'removeObjectAtIndex:':
                if r2 < len(items):
                    del items[r2]
            elif sel_name == 'removeLastObject':
                if items:
                    items.pop()
            elif sel_name == 'removeAllObjects':
                items.clear()
            elif sel_name == 'count':
                uc_engine.reg_write(R0, len(items))
            elif sel_name == 'objectAtIndex:':
                uc_engine.reg_write(R0, items[r2] if r2 < len(items) else 0)
            elif sel_name == 'lastObject':
                uc_engine.reg_write(R0, items[-1] if items else 0)
            elif sel_name == 'firstObject':
                uc_engine.reg_write(R0, items[0] if items else 0)
            elif sel_name == 'containsObject:':
                uc_engine.reg_write(R0, 1 if r2 in items else 0)
            elif sel_name == 'indexOfObject:':
                uc_engine.reg_write(R0, items.index(r2) if r2 in items else 0x7FFFFFFF)
            elif sel_name in ('copy', 'mutableCopy'):
                obj = self.objc_alloc(0)
                self.arrays[obj] = list(items)
                uc_engine.reg_write(R0, obj)
            else:
                handled = False
            if handled:
                return True

        # ---------------- NSNumber ----------------
        if is_class and cls_name == 'NSNumber':
            val = None
            if sel_name in ('numberWithInt:', 'numberWithInteger:', 'numberWithLong:'):
                val = signed(r2)
            elif sel_name in ('numberWithUnsignedInt:', 'numberWithUnsignedInteger:', 'numberWithUnsignedLong:'):
                val = r2
            elif sel_name == 'numberWithBool:':
                val = r2 & 0xFF
            elif sel_name == 'numberWithFloat:':
                val = self._f32(r2)
            elif sel_name == 'numberWithDouble:':
                val = self._arg_double(r2, r3)
            if val is not None:
                obj = self.objc_alloc(0)
                self.numbers[obj] = val
                uc_engine.reg_write(R0, obj)
                return True
        if receiver in self.numbers:
            val = self.numbers[receiver]
            if sel_name in ('intValue', 'integerValue', 'longValue', 'unsignedIntValue', 'unsignedIntegerValue'):
                uc_engine.reg_write(R0, int(val) & 0xFFFFFFFF)
                return True
            if sel_name == 'boolValue':
                uc_engine.reg_write(R0, 1 if val else 0)
                return True
            if sel_name == 'floatValue':
                self._ret_float(uc_engine, val)
                return True
            if sel_name == 'doubleValue':
                self._ret_double(uc_engine, val)
                return True
            if sel_name == 'isEqualToNumber:':
                uc_engine.reg_write(R0, 1 if self.numbers.get(r2) == val else 0)
                return True

        # ---------------- NSDate ----------------
        if is_class and cls_name == 'NSDate' and sel_name in ('date', 'dateWithTimeIntervalSinceNow:', 'distantFuture', 'distantPast',
                                                             'dateWithTimeIntervalSince1970:', 'dateWithTimeIntervalSinceReferenceDate:'):
            obj = self.objc_alloc(0)
            now = time.time()
            if sel_name == 'dateWithTimeIntervalSinceNow:':
                now += self._arg_double(r2, r3)
            elif sel_name == 'dateWithTimeIntervalSince1970:':
                now = self._arg_double(r2, r3)
            elif sel_name == 'dateWithTimeIntervalSinceReferenceDate:':
                now = self._arg_double(r2, r3) + 978307200.0
            elif sel_name == 'distantFuture':
                now += 1e10
            elif sel_name == 'distantPast':
                now -= 1e10
            self.dates[obj] = now
            uc_engine.reg_write(R0, obj)
            return True
        if (receiver in self.dates or (not is_class and cls_name == 'NSDate')) and sel_name in (
                'init', 'timeIntervalSinceNow', 'timeIntervalSinceDate:', 'timeIntervalSince1970', 'compare:',
                'earlierDate:', 'laterDate:', 'isEqualToDate:', 'timeIntervalSinceReferenceDate', 'dateByAddingTimeInterval:'):
            when = self.dates.setdefault(receiver, time.time())
            if sel_name == 'init':
                uc_engine.reg_write(R0, receiver)
            elif sel_name == 'compare:':           # NSComparisonResult: -1 ascending, 0 same, 1 descending
                other = self.dates.get(r2, when)
                uc_engine.reg_write(R0, 0xFFFFFFFF if when < other else (1 if when > other else 0))
            elif sel_name in ('earlierDate:', 'laterDate:'):
                other = self.dates.get(r2, when)
                pick_self = (when <= other) if sel_name == 'earlierDate:' else (when >= other)
                uc_engine.reg_write(R0, receiver if pick_self else r2)
            elif sel_name == 'isEqualToDate:':
                uc_engine.reg_write(R0, 1 if self.dates.get(r2) == when else 0)
            elif sel_name == 'timeIntervalSinceReferenceDate':
                self._ret_double(uc_engine, when - 978307200.0)
            elif sel_name == 'dateByAddingTimeInterval:':
                obj = self.objc_alloc(0)
                self.dates[obj] = when + self._arg_double(r2, r3)
                uc_engine.reg_write(R0, obj)
            elif sel_name == 'timeIntervalSinceNow':
                self._ret_double(uc_engine, when - time.time())
            elif sel_name == 'timeIntervalSinceDate:':
                self._ret_double(uc_engine, when - self.dates.get(r2, time.time()))
            else:
                self._ret_double(uc_engine, when)
            return True

        # ---------------- NSUserDefaults (in-memory store) ----------------
        if is_class and cls_name == 'NSUserDefaults' and sel_name == 'standardUserDefaults':
            if not self._defaults_obj:
                self._defaults_obj = self.objc_alloc(receiver)
                self._load_defaults()
            uc_engine.reg_write(R0, self._defaults_obj)
            return True
        if self._defaults_obj and receiver == self._defaults_obj:
            def key_of(ptr):
                return self._str_of(ptr) or f'@0x{ptr:08X}'

            def num_for(key):
                v = self.defaults.get(key, 0)
                if isinstance(v, tuple):
                    v = self.numbers.get(v[1], 0)
                return v
            if sel_name in ('setBool:forKey:', 'setInteger:forKey:', 'setFloat:forKey:', 'setObject:forKey:'):
                logging.info(f"[NSUserDefaults] {sel_name} key={key_of(r3)!r}")
                if sel_name == 'setBool:forKey:':
                    self.defaults[key_of(r3)] = r2 & 0xFF
                elif sel_name == 'setInteger:forKey:':
                    self.defaults[key_of(r3)] = signed(r2)
                elif sel_name == 'setFloat:forKey:':
                    self.defaults[key_of(r3)] = self._f32(r2)
                else:
                    self.defaults[key_of(r3)] = ('obj', r2)
                self._save_defaults()
                return True
            if sel_name == 'setDouble:forKey:':
                (key_ptr,) = self._stack_words(uc_engine, sp, 1)
                self.defaults[key_of(key_ptr)] = self._arg_double(r2, r3)
                return True
            if sel_name == 'removeObjectForKey:':
                self.defaults.pop(key_of(r2), None)
                return True
            if sel_name == 'synchronize':
                self._save_defaults()
                uc_engine.reg_write(R0, 1)
                return True
            if sel_name == 'registerDefaults:':
                return True
            if sel_name == 'boolForKey:':
                uc_engine.reg_write(R0, 1 if num_for(key_of(r2)) else 0)
                return True
            if sel_name in ('integerForKey:', 'intForKey:'):
                uc_engine.reg_write(R0, int(num_for(key_of(r2))) & 0xFFFFFFFF)
                return True
            if sel_name == 'floatForKey:':
                self._ret_float(uc_engine, num_for(key_of(r2)))
                return True
            if sel_name == 'doubleForKey:':
                self._ret_double(uc_engine, num_for(key_of(r2)))
                return True
            if sel_name in ('objectForKey:', 'stringForKey:', 'valueForKey:'):
                v = self.defaults.get(key_of(r2))
                if isinstance(v, tuple):
                    ptr = v[1]
                elif v is None:
                    ptr = 0
                else:
                    ptr = self.objc_alloc(0)
                    self.numbers[ptr] = v
                uc_engine.reg_write(R0, ptr)
                return True

        # ---------------- NSBundle / NSURL ----------------
        if is_class and cls_name == 'NSBundle' and sel_name == 'mainBundle':
            if not self._bundle_obj:
                self._bundle_obj = self.objc_alloc(receiver)
            uc_engine.reg_write(R0, self._bundle_obj)
            return True
        if self._bundle_obj and receiver == self._bundle_obj:
            if sel_name in ('resourcePath', 'bundlePath'):
                uc_engine.reg_write(R0, self._new_string(self.app_resource_dir or ''))
                return True
            if sel_name in ('pathForResource:ofType:', 'pathForResource:ofType:inDirectory:',
                            'URLForResource:withExtension:'):
                name, ext = self._str_of(r2), self._str_of(r3)
                found = None
                if name:
                    if self._resource_index is None:
                        self._resource_index = self._build_resource_index()
                    found = self._resource_index.get((f"{name}.{ext}" if ext else name).lower())
                if not found:
                    logging.warning(f"[NSBundle] resource not found: {name!r} type {ext!r}")
                    uc_engine.reg_write(R0, 0)
                elif sel_name.startswith('URL'):
                    url = self.objc_alloc(0)
                    self.urls[url] = found
                    uc_engine.reg_write(R0, url)
                else:
                    uc_engine.reg_write(R0, self._new_string(found))
                return True
        if is_class and cls_name == 'NSURL' and sel_name in ('fileURLWithPath:', 'URLWithString:', 'fileURLWithPath:isDirectory:'):
            url = self.objc_alloc(0)
            self.urls[url] = self._str_of(r2) or ''
            uc_engine.reg_write(R0, url)
            return True
        if receiver in self.urls and sel_name in ('path', 'absoluteString', 'relativePath', 'relativeString'):
            uc_engine.reg_write(R0, self._new_string(self.urls[receiver]))
            return True
        if receiver in self.urls and sel_name == 'isFileURL':
            uc_engine.reg_write(R0, 1)
            return True

        # ---------------- NSIndexPath (table rows) ----------------
        if is_class and cls_name == 'NSIndexPath' and sel_name in ('indexPathForRow:inSection:', 'indexPathForItem:inSection:'):
            ip = self.objc_alloc(0)
            self.index_paths[ip] = (r3, r2)
            uc_engine.reg_write(R0, ip)
            return True
        if receiver in self.index_paths:
            sec_, row_ = self.index_paths[receiver]
            if sel_name in ('row', 'item'):
                uc_engine.reg_write(R0, row_)
                return True
            if sel_name == 'section':
                uc_engine.reg_write(R0, sec_)
                return True
            if sel_name == 'length':
                uc_engine.reg_write(R0, 2)
                return True
            if sel_name == 'indexAtPosition:':
                uc_engine.reg_write(R0, (sec_, row_)[r2] if r2 < 2 else 0)
                return True

        # ---------------- UIFont (only the point size matters here) ----------------
        if is_class and cls_name == 'UIFont' and sel_name in ('fontWithName:size:', 'systemFontOfSize:',
                                                              'boldSystemFontOfSize:', 'italicSystemFontOfSize:'):
            obj = self.objc_alloc(0)
            self.fonts[obj] = self._f32(r3 if sel_name == 'fontWithName:size:' else r2)
            uc_engine.reg_write(R0, obj)
            return True

        # ---------------- UIImage from a file path ----------------
        if is_class and cls_name == 'UIImage' and sel_name == 'imageWithContentsOfFile:':
            path = self._str_of(r2)
            surface = self._load_app_image(os.path.basename(path)) if path else None
            obj = self.objc_alloc(0)
            if surface is not None:
                self.image_objects[obj] = surface
            uc_engine.reg_write(R0, obj)
            return True

        # ---------------- NSString ----------------
        if is_class and cls_name in ('NSString', 'NSMutableString'):
            text = None
            if sel_name == 'stringWithFormat:':
                text = self._format_string(uc_engine, self._str_of(r2) or '')
            elif sel_name == 'stringWithUTF8String:':
                text = self._cstring_at(r2) or ''
            elif sel_name == 'stringWithString:':
                text = self._str_of(r2) or ''
            elif sel_name == 'string':
                text = ''
            if text is not None:
                uc_engine.reg_write(R0, self._new_string(text))
                return True
        text = self.strings.get(receiver)
        if text is None and sel_name in self._STRING_SELECTORS and not is_class and receiver not in self.instance_isa \
                and receiver not in self.guest_views:
            text = self._nsstring_at(receiver)       # compile-time @"literal" receivers
        if text is not None and sel_name in self._STRING_SELECTORS:
            if sel_name in ('UTF8String', 'cStringUsingEncoding:', 'fileSystemRepresentation', 'cString'):
                uc_engine.reg_write(R0, self._cstr_ptr(text))
            elif sel_name == 'length':
                uc_engine.reg_write(R0, len(text))
            elif sel_name in ('intValue', 'integerValue'):
                m = re.match(r'\s*[-+]?\d+', text)
                uc_engine.reg_write(R0, (int(m.group()) if m else 0) & 0xFFFFFFFF)
            elif sel_name in ('floatValue', 'doubleValue'):
                m = re.match(r'\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
                val = float(m.group()) if m else 0.0
                (self._ret_float if sel_name == 'floatValue' else self._ret_double)(uc_engine, val)
            elif sel_name == 'boolValue':
                uc_engine.reg_write(R0, 1 if text[:1] in 'YyTt1' and text else 0)
            elif sel_name == 'isEqualToString:':
                uc_engine.reg_write(R0, 1 if self._str_of(r2) == text else 0)
            elif sel_name == 'hasPrefix:':
                uc_engine.reg_write(R0, 1 if text.startswith(self._str_of(r2) or '\x00') else 0)
            elif sel_name == 'hasSuffix:':
                uc_engine.reg_write(R0, 1 if text.endswith(self._str_of(r2) or '\x00') else 0)
            elif sel_name == 'stringByAppendingString:':
                uc_engine.reg_write(R0, self._new_string(text + (self._str_of(r2) or '')))
            elif sel_name == 'stringByAppendingPathComponent:':
                uc_engine.reg_write(R0, self._new_string(text.rstrip('/') + '/' + (self._str_of(r2) or '')))
            elif sel_name == 'lastPathComponent':
                uc_engine.reg_write(R0, self._new_string(text.rstrip('/').rsplit('/', 1)[-1]))
            elif sel_name == 'pathExtension':
                uc_engine.reg_write(R0, self._new_string(os.path.splitext(text)[1].lstrip('.')))
            elif sel_name == 'stringByDeletingPathExtension':
                uc_engine.reg_write(R0, self._new_string(os.path.splitext(text)[0]))
            elif sel_name == 'stringByDeletingLastPathComponent':
                uc_engine.reg_write(R0, self._new_string(text.rstrip('/').rsplit('/', 1)[0] if '/' in text else ''))
            return True
        return False

    _WATCH_SELECTORS = frozenset({'calibratedAngleFromAngle:', 'updateToInclinationInRadians:',
                                  'setCurrentAngle:', 'setFluidHeight:'})

    def _watch(self, uc_engine, cls_name, sel_name, imp):
        """Tilt pipeline probe: what each stage is handed (as double and as float), and - once - its machine code."""
        counts = self.__dict__.setdefault('_watch_counts', {})
        n = counts.get(sel_name, 0)
        counts[sel_name] = n + 1
        r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
        r3 = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
        if n == 0:
            try:
                code = bytes(uc_engine.mem_read(imp & ~1, 192)).hex()
                logging.info(f"[Watch] code -[{cls_name} {sel_name}] @0x{imp:08X} ({'thumb' if imp & 1 else 'arm'}): {code}")
            except Exception:
                pass
        if n < 6 or n % 60 == 0:
            as_d = struct.unpack('<d', struct.pack('<II', r2, r3))[0]
            as_f = struct.unpack('<f', struct.pack('<I', r2))[0]
            logging.info(f"[Watch] -[{cls_name} {sel_name}] arg as double={as_d:.4f} ({math.degrees(as_d):.1f} deg) "
                         f"as float={as_f:.4f} ({math.degrees(as_f):.1f} deg)  #{n + 1}")

    # ------------------------------------------------------------------
    # UITableView: ask the app's datasource for its rows and lay them out (Settings screen, etc.)
    # ------------------------------------------------------------------
    def _on_screen(self, v):
        while v.parent is not None:
            v = v.parent
        return v in self.nib_roots

    def _tick_tables(self):
        for tv in list(self.guest_views.values()):
            if tv.class_name == 'UITableView' and tv.table_state == 'new' and self._on_screen(tv):
                self._load_table(tv)

    def _load_table(self, tv):
        from objc_runtime import resolve_method
        ds = tv.table_ds or tv.table_controller
        tv.table_state = 'done'
        if not ds:
            return
        tv.table_state = 'loading'
        tv_obj = tv.guest
        dscls = self._class_of(ds)
        has = lambda sel: bool(dscls and resolve_method(self.classes_by_addr, dscls, sel))
        st = {'n': 1, 'counts': [], 'plan': [], 'cells': []}

        def ask(sel, a2, a3, cb):
            self._action_queue.append({'target': ds, 'sel': sel, 'arg2': a2, 'arg3': a3, 'on_return': cb})

        def got_sections(n):
            st['n'] = max(1, min(int(n) if n < 0x80000000 else 1, 16))
            ask_count(0)

        def ask_count(sec):
            if sec >= st['n']:
                st['plan'] = [(s_, r_) for s_ in range(st['n']) for r_ in range(st['counts'][s_])]
                return ask_cell(0)
            if not has('tableView:numberOfRowsInSection:'):
                st['counts'].append(0)
                return ask_count(sec + 1)
            ask('tableView:numberOfRowsInSection:', tv_obj, sec, lambda r0, r1, sec=sec: (
                st['counts'].append(min(r0, 200) if r0 < 0x80000000 else 0), ask_count(sec + 1)))

        def ask_cell(i):
            if i >= len(st['plan']):
                return layout()
            sec, row = st['plan'][i]
            ip = self.objc_alloc(0)
            self.index_paths[ip] = (sec, row)
            ask('tableView:cellForRowAtIndexPath:', tv_obj, ip, lambda r0, r1, i=i: (st['cells'].append(r0), ask_cell(i + 1)))

        def layout():
            grouped = tv.table_style == 1
            width = tv.frame[2] or 320.0
            inset = 10.0 if grouped else 0.0
            y, idx = 0.0, 0
            for sec in range(st['n']):
                y += 10.0 if grouped else 0.0                       # grouped tables leave a 10pt gap above a group
                for _row in range(st['counts'][sec]):
                    cell = st['cells'][idx] if idx < len(st['cells']) else 0
                    idx += 1
                    cv = self.guest_views.get(cell) if cell else None
                    if cv is None:
                        y += tv.row_height
                        continue
                    self._attach_subview(tv.guest, cell)
                    cv.frame = (inset, y, width - 2 * inset, tv.row_height)
                    cv.class_name = 'UITableViewCell'
                    if cv.bg_color is None:
                        cv.bg_color = (255, 255, 255)
                    cv.text_inset = 15.0
                    for lab in (self._cell_labels.get((cell, 'textLabel')),):
                        lv = self.guest_views.get(lab) if lab else None
                        if lv is not None:
                            lv.frame = (15.0, 0.0, cv.frame[2] - 30.0, tv.row_height)
                    y += tv.row_height
                y += 10.0 if grouped else 0.0
            tv.table_state = 'done'
            logging.info(f"[UITableView] 0x{tv.guest:08X}: {len(st['plan'])} rows in {st['n']} section(s)")

        if has('numberOfSectionsInTableView:'):
            ask('numberOfSectionsInTableView:', tv_obj, 0, lambda r0, r1: got_sections(r0))
        else:
            got_sections(1)

    def _load_defaults(self):
        """Settings survive between launches like on a device (EMU_RESET_DEFAULTS=1 starts clean)."""
        import json
        if not self.app_resource_dir:
            return
        self._defaults_path = os.path.join(os.path.dirname(self.app_resource_dir.rstrip('/\\')), 'NSUserDefaults.json')
        if os.environ.get('EMU_RESET_DEFAULTS') == '1' or not os.path.isfile(self._defaults_path):
            return
        try:
            with open(self._defaults_path, 'r', encoding='utf-8') as f:
                self.defaults.update(json.load(f))
            logging.info(f"[NSUserDefaults] loaded {len(self.defaults)} saved settings from {self._defaults_path}")
        except Exception as e:
            logging.warning(f"[NSUserDefaults] could not read {self._defaults_path}: {e}")

    def _save_defaults(self):
        import json
        if not self._defaults_path:
            return
        try:
            plain = {k: v for k, v in self.defaults.items() if isinstance(v, (int, float))}
            with open(self._defaults_path, 'w', encoding='utf-8') as f:
                json.dump(plain, f, indent=1)
        except Exception as e:
            logging.warning(f"[NSUserDefaults] could not save settings: {e}")

    def dump_view_tree(self):
        from nib_loader import dump_tree
        return dump_tree(self.nib_roots)

    def _trace_msg(self, uc_engine, receiver, class_addr, sel_name, result, r2, r3):
        if self._trace is None:
            try:
                self._trace = open('trace.log', 'w', buffering=1, encoding='utf-8')
                logging.info("[Trace] EMU_TRACE=1: framework messages -> trace.log")
            except Exception:
                self._trace = False
        if not self._trace:
            return
        info = self.classes_by_addr.get(class_addr) if class_addr else None
        kind = '+' if receiver in self.classes_by_addr else '-'
        r0 = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
        text = self.strings.get(r2) if r2 in self.strings else None
        self._trace.write(f"{time.monotonic():10.3f} {kind}[{info['name'] if info else '?'} {sel_name}] "
                          f"recv=0x{receiver:08X} r2=0x{r2:08X}{'(' + repr(text) + ')' if text else ''} r3=0x{r3:08X} "
                          f"-> {result} r0=0x{r0:08X}\n")

    def _log_unhandled(self, receiver, class_addr, sel_name):
        """Report every framework selector nothing implements (once each) - it silently returns the receiver."""
        if sel_name in self._BENIGN_SELECTORS:
            return
        info = self.classes_by_addr.get(class_addr) if class_addr else None
        cls_name = info['name'] if info else 'unknown'
        kind = '+' if receiver in self.classes_by_addr else '-'
        key = (kind, cls_name, sel_name)
        if key in self._unhandled_seen:
            return
        self._unhandled_seen.add(key)
        logging.warning(f"[ObjC Unhandled] {kind}[{cls_name} {sel_name}] receiver=0x{receiver:08X} - nothing implements this; returned receiver")

    # ------------------------------------------------------------------
    # soft-float helpers + libm, by name (softfp ABI: floats in r0/r1, doubles in r0:r1 / r2:r3)
    # ------------------------------------------------------------------
    _LIBM1 = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'asin': math.asin, 'acos': math.acos,
              'atan': math.atan, 'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh, 'sqrt': math.sqrt,
              'exp': math.exp, 'log': math.log, 'log10': math.log10, 'log2': math.log2, 'floor': math.floor,
              'ceil': math.ceil, 'fabs': math.fabs, 'trunc': math.trunc,
              'round': lambda v: math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)}
    _LIBM2 = {'atan2': math.atan2, 'pow': math.pow, 'fmod': math.fmod, 'fmax': max, 'fmin': min, 'hypot': math.hypot}

    def _math_stub(self, uc_engine, name):
        base = name.lstrip('_')
        R = (arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R1, arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3)
        r = [uc_engine.reg_read(x) for x in R]
        f = lambda w: struct.unpack('<f', struct.pack('<I', w & 0xFFFFFFFF))[0]
        d = lambda lo, hi: struct.unpack('<d', struct.pack('<II', lo, hi))[0]

        def ret_f(v):
            uc_engine.reg_write(R[0], struct.unpack('<I', struct.pack('<f', v))[0])

        def ret_d(v):
            lo, hi = struct.unpack('<II', struct.pack('<d', v))
            uc_engine.reg_write(R[0], lo)
            uc_engine.reg_write(R[1], hi)
        try:
            if base.endswith('vfp'):
                core = base[:-3]
                for rel, fn in (('eq', lambda a, b: a == b), ('ne', lambda a, b: a != b), ('lt', lambda a, b: a < b),
                                ('le', lambda a, b: a <= b), ('gt', lambda a, b: a > b), ('ge', lambda a, b: a >= b)):
                    if core in (f'__{rel}sf2', f'__{rel}df2', f'{rel}sf2', f'{rel}df2'):
                        a, b = (f(r[0]), f(r[1])) if core.endswith('sf2') else (d(r[0], r[1]), d(r[2], r[3]))
                        uc_engine.reg_write(R[0], 1 if fn(a, b) else 0)
                        return True
                if core.lstrip('_') == 'floatsidf':
                    ret_d(float(r[0] - 0x100000000 if r[0] & 0x80000000 else r[0]))
                elif core.lstrip('_') == 'floatunssidf':
                    ret_d(float(r[0]))
                elif core.lstrip('_') == 'floatunssisf':
                    ret_f(float(r[0]))
                elif core.lstrip('_') == 'fixunssfsi':
                    uc_engine.reg_write(R[0], max(0, int(f(r[0]))) & 0xFFFFFFFF)
                elif core.lstrip('_') == 'fixunsdfsi':
                    uc_engine.reg_write(R[0], max(0, int(d(r[0], r[1]))) & 0xFFFFFFFF)
                else:
                    return False
                return True
            single = base.endswith('f') and base[:-1] in {**self._LIBM1, **self._LIBM2}
            key = base[:-1] if single else base
            if key in self._LIBM1:
                v = self._LIBM1[key](f(r[0]) if single else d(r[0], r[1]))
                (ret_f if single else ret_d)(float(v))
                return True
            if key in self._LIBM2:
                a, b = (f(r[0]), f(r[1])) if single else (d(r[0], r[1]), d(r[2], r[3]))
                (ret_f if single else ret_d)(float(self._LIBM2[key](a, b)))
                return True
        except (ValueError, OverflowError, ZeroDivisionError):
            (ret_f if base.endswith('f') else ret_d)(float('nan'))
            return True
        return False

    # ------------------------------------------------------------------
    # CATransform3D / CGAffineTransform C functions (struct returns go through the hidden r0 pointer)
    # ------------------------------------------------------------------
    def _transform_stub(self, uc_engine, name):
        """Returns True if `name` was a transform function and was emulated."""
        base = name.lstrip('_')
        if not base.startswith(('CATransform3D', 'CGAffineTransform')):
            return False
        regs = [uc_engine.reg_read(r) for r in (arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R1,
                                                 arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3)]
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        stack = self._stack_words(uc_engine, sp, 40)
        ret_ptr = regs[0]
        a = [self._f32(w) for w in regs[1:] + list(stack)]    # args after the hidden return pointer
        d = [self._f32(w) for w in regs + list(stack)]        # args when nothing is returned via memory
        m3 = aff = None
        boolean = None

        if base == 'CATransform3DMakeRotation':
            m3 = _m3_rotation(*a[0:4])
        elif base == 'CATransform3DMakeScale':
            m3 = _m3_scale(*a[0:3])
        elif base == 'CATransform3DMakeTranslation':
            m3 = _m3_translation(*a[0:3])
        elif base == 'CATransform3DRotate':
            m3 = _m3_mul(_m3_rotation(*a[16:20]), a[0:16])
        elif base == 'CATransform3DScale':
            m3 = _m3_mul(_m3_scale(*a[16:19]), a[0:16])
        elif base == 'CATransform3DTranslate':
            m3 = _m3_mul(_m3_translation(*a[16:19]), a[0:16])
        elif base == 'CATransform3DConcat':
            m3 = _m3_mul(a[0:16], a[16:32])
        elif base == 'CATransform3DInvert':
            try:
                m3 = [float(v) for v in np.linalg.inv(np.array(a[0:16], dtype=np.float64).reshape(4, 4)).reshape(-1)]
            except Exception:
                m3 = list(a[0:16])
        elif base == 'CATransform3DIsIdentity':
            boolean = all(abs(x - y) < 1e-6 for x, y in zip(d[0:16], _M3_IDENTITY))
        elif base == 'CATransform3DGetAffineTransform':
            aff = _aff_from_m3(a[0:16])
        elif base == 'CATransform3DMakeAffineTransform':
            m3 = _m3_from_aff(a[0:6])
        elif base == 'CGAffineTransformMakeRotation':
            c, s_ = math.cos(a[0]), math.sin(a[0])
            aff = (c, s_, -s_, c, 0.0, 0.0)
        elif base == 'CGAffineTransformMakeScale':
            aff = (a[0], 0.0, 0.0, a[1], 0.0, 0.0)
        elif base == 'CGAffineTransformMakeTranslation':
            aff = (1.0, 0.0, 0.0, 1.0, a[0], a[1])
        elif base == 'CGAffineTransformRotate':
            c, s_ = math.cos(a[6]), math.sin(a[6])
            aff = _aff_mul((c, s_, -s_, c, 0.0, 0.0), tuple(a[0:6]))
        elif base == 'CGAffineTransformScale':
            aff = _aff_mul((a[6], 0.0, 0.0, a[7], 0.0, 0.0), tuple(a[0:6]))
        elif base == 'CGAffineTransformTranslate':
            aff = _aff_mul((1.0, 0.0, 0.0, 1.0, a[6], a[7]), tuple(a[0:6]))
        elif base == 'CGAffineTransformConcat':
            aff = _aff_mul(tuple(a[0:6]), tuple(a[6:12]))
        elif base == 'CGAffineTransformInvert':
            ta, tb, tc, td, tx, ty = a[0:6]
            det = ta * td - tb * tc
            if abs(det) < 1e-12:
                aff = tuple(a[0:6])
            else:
                aff = (td / det, -tb / det, -tc / det, ta / det,
                       (tc * ty - td * tx) / det, (tb * tx - ta * ty) / det)
        elif base == 'CGAffineTransformIsIdentity':
            boolean = all(abs(x - y) < 1e-6 for x, y in zip(d[0:6], (1, 0, 0, 1, 0, 0)))
        else:
            return False

        try:
            if m3 is not None:
                uc_engine.mem_write(ret_ptr, struct.pack('<16f', *m3))
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, ret_ptr)
            elif aff is not None:
                uc_engine.mem_write(ret_ptr, struct.pack('<6f', *aff))
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, ret_ptr)
            else:
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, 1 if boolean else 0)
        except Exception as e:
            logging.warning(f"[Transform] {base}: could not write result: {e}")
        n = self.__dict__.get('_xf_logged', 0)
        if n < 8:
            self._xf_logged = n + 1
            logging.info(f"[Transform] {base} emulated")
        return True

    def _stret_shim(self, uc_engine, ret_ptr, receiver, sel_name):
        """Struct-returning framework selectors (CGPoint/CGRect...) - the hidden r0 return pointer."""
        try:
            if sel_name == 'locationInView:':
                uc_engine.mem_write(ret_ptr, struct.pack('<ff', *self.touch_pos))
                return
            if sel_name == 'size' and receiver in self.image_objects:
                img = self.image_objects[receiver]
                uc_engine.mem_write(ret_ptr, struct.pack('<2f', float(img.get_width()), float(img.get_height())))
                return
            if sel_name == 'applicationFrame':
                top = 0.0 if self.status_bar_hidden else 20.0
                uc_engine.mem_write(ret_ptr, struct.pack('<4f', 0.0, top, float(self.cg.width),
                                                         float(self.cg.height) - top))
                return
            if sel_name in ('transform', 'affineTransform'):
                owner = self._layer_owner.get(receiver)
                v = self.guest_views.get(owner if owner is not None else receiver)
                t = v.transform if v is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                if owner is not None and sel_name == 'transform':
                    uc_engine.mem_write(ret_ptr, struct.pack('<16f', *_m3_from_aff(t)))      # CALayer.transform
                else:
                    uc_engine.mem_write(ret_ptr, struct.pack('<6f', *t))                      # UIView.transform
                return
            view = self.guest_views.get(receiver)
            if sel_name in ('bounds', 'frame', 'applicationFrame', 'contentStretch'):
                if view is not None:
                    x, y, w, h = view.frame
                    if sel_name == 'bounds':
                        x, y = 0.0, 0.0
                else:
                    x, y, w, h = 0.0, 0.0, float(self.cg.width), float(self.cg.height)
                uc_engine.mem_write(ret_ptr, struct.pack('<4f', x, y, w, h))
                return
            if sel_name == 'center' and view is not None:
                x, y, w, h = view.frame
                ax, ay = getattr(view, 'anchor', (0.5, 0.5))
                uc_engine.mem_write(ret_ptr, struct.pack('<2f', x + ax * w, y + ay * h))
                return
            uc_engine.mem_write(ret_ptr, b'\x00' * 8)   # unknown struct: nil-messaging style zeros
        except Exception as e:
            logging.warning(f"[stret] {sel_name}: {e}")

    def _c_alloc(self, size):
        size = max(8, (size + 7) & ~7)
        ptr = self.cmalloc_next
        if ptr + size >= self.cmalloc_base + self.cmalloc_size:
            logging.warning("[libc] malloc arena exhausted - wrapping (contents will alias)")
            ptr = self.cmalloc_base
        self.cmalloc_next = ptr + size
        return ptr

    # ------------------------------------------------------------------
    # Dynamic UIView model: views the app creates in CODE (alloc/initWithFrame:/addSubview:...)
    # ------------------------------------------------------------------
    _UI_VIEW_NAMES = {
        'UIView', 'UIImageView', 'UIButton', 'UILabel', 'UIWindow', 'UIScrollView', 'UIWebView',
        'UIControl', 'UITextField', 'UITextView', 'UIToolbar', 'UINavigationBar', 'UISlider',
        'UISwitch', 'UIActivityIndicatorView', 'UITableView', 'UIRoundedRectButton',
    }
    _NAMED_COLORS = {
        'blackColor': (0, 0, 0), 'whiteColor': (255, 255, 255), 'redColor': (255, 0, 0),
        'greenColor': (0, 255, 0), 'blueColor': (0, 0, 255), 'grayColor': (128, 128, 128),
        'yellowColor': (255, 255, 0), 'orangeColor': (255, 128, 0), 'clearColor': None,
        'lightGrayColor': (170, 170, 170), 'darkGrayColor': (85, 85, 85), 'purpleColor': (128, 0, 128),
        'brownColor': (153, 102, 51), 'cyanColor': (0, 255, 255), 'magentaColor': (255, 0, 255),
        'darkTextColor': (0, 0, 0), 'lightTextColor': (255, 255, 255),
        'groupTableViewBackgroundColor': (197, 204, 212),
    }

    def _view_kind_for(self, obj):
        """(UIKit class name, app class name or None) for an object, walking the superclass chain."""
        from objc_runtime import resolve_method
        cls = self._class_of(obj)
        kind, custom, hops = 'UIView', None, 0
        while cls and hops < 12:
            info = self.classes_by_addr.get(cls)
            if not info:
                break
            if cls in self.external_class_addrs:
                if info['name'] in self._UI_VIEW_NAMES:
                    kind = info['name']
                break
            if custom is None:
                custom = info['name']
            cls = info.get('super', 0)
            hops += 1
        return kind, custom

    def _get_view(self, obj):
        v = self.guest_views.get(obj)
        if v is None and obj:
            from objc_runtime import resolve_method
            kind, custom = self._view_kind_for(obj)
            v = NibView({}, kind, custom)
            v.guest = obj
            if kind == 'UILabel':
                v.text_color = (0, 0, 0)          # UILabel's default text colour
            self.guest_views[obj] = v
            cls = self._class_of(obj)
            if custom and cls and resolve_method(self.classes_by_addr, cls, "drawRect:"):
                v.custom_draw = True
                if obj not in self.view_hierarchy:
                    self.view_hierarchy.append(obj)
        return v

    def _attach_subview(self, parent_obj, child_obj, index=None, above=None, below=None):
        """
        addSubview: / insertSubview:atIndex: / ...aboveSubview: / ...belowSubview:. Position in `children` IS the
        z-order (last = frontmost), so ignoring the index/sibling - as this used to - put every view on top.
        """
        pv, cv = self._get_view(parent_obj), self._get_view(child_obj)
        if pv is None or cv is None or pv is cv:
            return
        if cv.parent is not None and cv in cv.parent.children:
            cv.parent.children.remove(cv)
        if cv in self.nib_roots:
            self.nib_roots.remove(cv)
        cv.parent = pv
        pos = len(pv.children)
        if index is not None:
            pos = max(0, min(index, len(pv.children)))
        elif above is not None or below is not None:
            sib = self.guest_views.get(above if above is not None else below)
            if sib is not None and sib in pv.children:
                pos = pv.children.index(sib) + (1 if above is not None else 0)
        pv.children.insert(pos, cv)
        if cv.class_name == 'UISwitch' and not cv.on_set and os.environ.get('EMU_SWITCH_ON') == '1':
            cv.on = True          # experiment: a switch nobody has set() starts ON (see EMU_SWITCH_ON)

    def _regs(self, uc_engine):
        rd = uc_engine.reg_read
        return (rd(arm_const.UC_ARM_REG_R2), rd(arm_const.UC_ARM_REG_R3), rd(arm_const.UC_ARM_REG_SP))

    def _stack_words(self, uc_engine, sp, n):
        try:
            return struct.unpack('<%dI' % n, bytes(uc_engine.mem_read(sp, 4 * n)))
        except Exception:
            return (0,) * n

    def _view_message(self, uc_engine, receiver, sel_name):
        """
        Framework UIView/UIControl/UIWindow behaviour for views the app builds in code.
        Returns True if handled (R0 already set). Args follow the ARM ABI: a CGRect/CGPoint
        argument's first two floats ride in r2/r3, the rest of a CGRect spills to the stack.
        """
        R0 = arm_const.UC_ARM_REG_R0
        r2, r3, sp = self._regs(uc_engine)

        # --- UIColor class factories ---
        if sel_name in self._NAMED_COLORS and receiver in self.classes_by_addr:
            obj = self.objc_alloc(0)
            self.color_objs[obj] = self._NAMED_COLORS[sel_name]
            uc_engine.reg_write(R0, obj)
            return True
        if sel_name == 'colorWithWhite:alpha:' and receiver in self.classes_by_addr:
            w = int(max(0.0, min(1.0, self._f32(r2))) * 255)
            obj = self.objc_alloc(0)
            self.color_objs[obj] = (w, w, w) if self._f32(r3) > 0.0 else None
            uc_engine.reg_write(R0, obj)
            return True
        if sel_name == 'colorWithRed:green:blue:alpha:' and receiver in self.classes_by_addr:
            b, a = self._stack_words(uc_engine, sp, 2)
            rgb = tuple(int(max(0.0, min(1.0, self._f32(x))) * 255) for x in (r2, r3, b))
            obj = self.objc_alloc(0)
            self.color_objs[obj] = rgb if self._f32(a) > 0.0 else None
            uc_engine.reg_write(R0, obj)
            return True
        if sel_name in ('boolForKey:', 'integerForKey:', 'intForKey:', 'objectForKey:', 'stringForKey:',
                        'floatForKey:', 'valueForKey:'):
            uc_engine.reg_write(R0, 0)      # empty NSUserDefaults: never echo the receiver as a truthy value
            return True

        # --- UITableViewController / UITableView / UITableViewCell ---
        if sel_name == 'initWithStyle:' and receiver not in self.classes_by_addr \
                and self._inherits(self._class_of(receiver), 'UITableViewController'):
            self.table_styles[receiver] = r2
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name in ('initWithFrame:reuseIdentifier:', 'initWithStyle:reuseIdentifier:') and receiver not in self.classes_by_addr:
            cv_ = self._get_view(receiver)
            cv_.class_name = 'UITableViewCell'
            cv_.frame = (0.0, 0.0, 320.0, 44.0)
            cv_.bg_color = (255, 255, 255)
            cv_.text_inset = 15.0
            uc_engine.reg_write(R0, receiver)
            return True
        tvw = self.guest_views.get(receiver)
        if tvw is not None and tvw.class_name == 'UITableView':
            if sel_name == 'dequeueReusableCellWithIdentifier:':
                uc_engine.reg_write(R0, 0)                       # nothing to reuse: the app builds its cells
                return True
            if sel_name == 'setDataSource:':
                tvw.table_ds = r2
                return True
            if sel_name == 'setRowHeight:':
                tvw.row_height = self._f32(r2)
                return True
            if sel_name == 'reloadData':
                for c_ in [c for c in tvw.children if c.class_name == 'UITableViewCell']:
                    tvw.children.remove(c_)
                    c_.parent = None
                tvw.table_state = 'new'
                return True
            if sel_name in ('setDelegate:', 'setSeparatorStyle:', 'setSectionHeaderHeight:', 'setSectionFooterHeight:',
                            'setAllowsSelection:', 'setScrollEnabled:', 'setSeparatorColor:', 'setBackgroundView:'):
                return True
        if tvw is not None and tvw.class_name == 'UITableViewCell':
            if sel_name in ('textLabel', 'detailTextLabel'):
                key = (receiver, sel_name)
                if key not in self._cell_labels:
                    lab = self.objc_alloc(0)
                    lv = self._get_view(lab)
                    lv.class_name = 'UILabel'
                    lv.frame = (15.0, 0.0, max(10.0, tvw.frame[2] - 30.0), tvw.frame[3])
                    lv.text_align = 0
                    self._cell_labels[key] = lab
                    self._attach_subview(receiver, lab)
                uc_engine.reg_write(R0, self._cell_labels[key])
                return True
            if sel_name == 'contentView':
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setAccessoryView:' and r2 and r2 in self.guest_views:
                acc = self.guest_views[r2]
                self._attach_subview(receiver, r2)
                acc.frame = (tvw.frame[2] - acc.frame[2] - 12.0, (tvw.frame[3] - acc.frame[3]) / 2.0, acc.frame[2], acc.frame[3])
                return True
            if sel_name in ('setSelectionStyle:', 'setAccessoryType:', 'setSelectedBackgroundView:',
                            'setBackgroundView:', 'setEditingAccessoryType:'):
                return True

        # --- per-view properties: tag, UISwitch state, UIImageView frame animation, transforms ---
        pv = self.guest_views.get(receiver)
        if pv is None and receiver not in self.classes_by_addr and sel_name in (
                'setFont:', 'setTextAlignment:', 'setSelected:', 'setHighlighted:', 'setEnabled:', 'setTag:'):
            pv = self._get_view(receiver)          # a control configured before (or without) initWithFrame:
        if pv is not None:
            if sel_name == 'setTag:':
                pv.tag = r2 - 0x100000000 if r2 & 0x80000000 else r2
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'tag':
                uc_engine.reg_write(R0, pv.tag & 0xFFFFFFFF)
                return True
            if sel_name in ('setOn:', 'setOn:animated:'):
                pv.on = bool(r2 & 0xFF)
                pv.on_set = True
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'isOn':
                uc_engine.reg_write(R0, 1 if pv.on else 0)
                return True
            if sel_name == 'setAnimationImages:':
                frames = [self.image_objects[p] for p in self.arrays.get(r2, []) if p in self.image_objects]
                pv.anim_images = frames or None
                pv.anim_finished = False
                logging.info(f"[UIImageView] 0x{receiver:08X} animationImages = {len(frames)} frames")
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setAnimationDuration:':
                pv.anim_duration = struct.unpack('<d', struct.pack('<II', r2, r3))[0]
                return True
            if sel_name == 'setAnimationRepeatCount:':
                pv.anim_repeat = max(0, r2 - 0x100000000 if r2 & 0x80000000 else r2)
                return True
            if sel_name == 'startAnimating':
                if pv.anim_images:
                    pv.animating = True
                    pv.anim_finished = False
                    pv.anim_start = None            # the renderer stamps it the first frame the view is on screen
                    logging.info(f"[UIImageView] 0x{receiver:08X} startAnimating ({len(pv.anim_images)} frames, "
                                 f"duration {pv.anim_duration or len(pv.anim_images) / 30.0:.2f}s, "
                                 f"repeat {pv.anim_repeat or 'forever'})")
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'stopAnimating':
                pv.animating = False
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'isAnimating':
                uc_engine.reg_write(R0, 1 if pv.animating else 0)
                return True
            if sel_name == 'setTransform:':
                self.anim.set_view_prop(pv, 'transform', (self._f32(r2), self._f32(r3)) +
                                        tuple(self._f32(w) for w in self._stack_words(uc_engine, sp, 4)))
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'alpha':
                self._ret_float(uc_engine, pv.alpha)
                return True
            if sel_name in ('setSelected:', 'setHighlighted:', 'setEnabled:'):
                setattr(pv, {'setSelected:': 'selected', 'setHighlighted:': 'highlighted',
                             'setEnabled:': 'enabled'}[sel_name], bool(r2 & 0xFF))
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name in ('isSelected', 'isHighlighted', 'isEnabled'):
                uc_engine.reg_write(R0, 1 if getattr(pv, {'isSelected': 'selected', 'isHighlighted': 'highlighted',
                                                           'isEnabled': 'enabled'}[sel_name]) else 0)
                return True
            if sel_name == 'setFont:':
                pv.font_size = self.fonts.get(r2, pv.font_size)
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setTextAlignment:':
                pv.text_align = r2
                uc_engine.reg_write(R0, receiver)
                return True

        if sel_name in ('setStatusBarHidden:withAnimation:', 'setStatusBarHidden:animated:'):
            self.status_bar_hidden = bool(r2 & 0xFF)
            logging.info(f"[UIKit] setStatusBarHidden: {self.status_bar_hidden}")
            return True
        # --- window / view controller ---
        if sel_name == 'makeKeyAndVisible':
            v = self._get_view(receiver)
            v.hidden = False
            if v.parent is None and v not in self.nib_roots:
                self.nib_roots.insert(0, v)
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name == 'setView:' and r2:
            old_guest = self.vc_views.get(receiver)
            new_v = self._get_view(r2)
            if old_guest and old_guest in self.guest_views and new_v is not None:
                ov = self.guest_views[old_guest]
                if ov.parent is not None and ov in ov.parent.children:
                    ov.parent.children[ov.parent.children.index(ov)] = new_v
                    new_v.parent = ov.parent
                elif ov in self.nib_roots:
                    self.nib_roots[self.nib_roots.index(ov)] = new_v
            self.vc_views[receiver] = r2
            return True

        # --- CALayer handles: stable per view, so layer messages can reach their view ---
        if sel_name == 'layer' and receiver in self.guest_views:
            layer = self._view_layer.get(receiver)
            if layer is None:
                layer = self.objc_alloc(0)
                self._view_layer[receiver] = layer
                self._layer_owner[layer] = receiver
            uc_engine.reg_write(R0, layer)
            return True
        owner = self._layer_owner.get(receiver)
        if owner is not None:
            ov = self.guest_views.get(owner)
            if sel_name == 'setAnchorPoint:' and ov is not None:
                # CALayer.anchorPoint: which point of the view sits at `center` (its position).
                # Changing it keeps the position fixed, so the frame moves instead.
                ax, ay = self._f32(r2), self._f32(r3)
                oax, oay = getattr(ov, 'anchor', (0.5, 0.5))
                x, y, w, h = ov.frame
                px, py = x + oax * w, y + oay * h
                ov.anchor = (ax, ay)
                ov.frame = (px - ax * w, py - ay * h, w, h)
                logging.info(f"[UIKit] anchorPoint ({ax:.2f},{ay:.2f}) on 0x{owner:08X}; frame now "
                             f"({ov.frame[0]:.0f},{ov.frame[1]:.0f},{w:.0f},{h:.0f})")
                return True
            if sel_name == 'setTransform:' and ov is not None:      # CATransform3D: r2,r3 + 14 words on the stack
                m = [self._f32(w) for w in (r2, r3) + self._stack_words(uc_engine, sp, 14)]
                self.anim.set_view_prop(ov, 'transform', _aff_from_m3(m))
                n = self.__dict__.get('_layer_xf_logged', 0)
                if n < 4:
                    self._layer_xf_logged = n + 1
                    logging.info(f"[CALayer] setTransform on 0x{owner:08X}: rotation "
                                 f"{math.degrees(math.atan2(ov.transform[1], ov.transform[0])):.1f} deg, "
                                 f"scale ({math.hypot(ov.transform[0], ov.transform[1]):.2f}, "
                                 f"{math.hypot(ov.transform[2], ov.transform[3]):.2f})")
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setAffineTransform:' and ov is not None:
                self.anim.set_view_prop(ov, 'transform', (self._f32(r2), self._f32(r3)) +
                                        tuple(self._f32(w) for w in self._stack_words(uc_engine, sp, 4)))
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setOpacity:' and ov is not None:
                self.anim.set_view_prop(ov, 'alpha', self._f32(r2))
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setHidden:' and ov is not None:
                ov.hidden = bool(r2 & 0xFF)
                uc_engine.reg_write(R0, receiver)
                return True
            if sel_name == 'setPosition:' and ov is not None:
                cx, cy = self._f32(r2), self._f32(r3)
                ax, ay = getattr(ov, 'anchor', (0.5, 0.5))
                self.anim.set_view_prop(ov, 'frame', (cx - ax * ov.frame[2], cy - ay * ov.frame[3], ov.frame[2], ov.frame[3]))
                uc_engine.reg_write(R0, receiver)
                return True

        # --- view geometry / hierarchy ---
        if sel_name in ('initWithFrame:', 'setFrame:', 'setBounds:'):
            w, h = self._stack_words(uc_engine, sp, 2)
            v = self._get_view(receiver)
            x, y, w, h = self._f32(r2), self._f32(r3), self._f32(w), self._f32(h)
            if sel_name == 'initWithFrame:':
                v.frame = (x, y, w, h)
            elif sel_name == 'setFrame:':
                self.anim.set_view_prop(v, 'frame', (x, y, w, h))
            else:                                   # setBounds: resizes about the view's centre (position), not its corner
                ox, oy, ow, oh = v.frame
                ax, ay = getattr(v, 'anchor', (0.5, 0.5))
                new = (ox, oy, w, h) if ow == 0 and oh == 0 else (ox + ax * ow - ax * w, oy + ay * oh - ay * h, w, h)
                self.anim.set_view_prop(v, 'frame', new)
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name == 'initWithImage:':
            v = self._get_view(receiver)
            img = self.image_objects.get(r2)
            if img is not None:
                v.image_override = img
                v.frame = (0.0, 0.0, float(img.get_width()), float(img.get_height()))
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name == 'setCenter:':
            v = self._get_view(receiver)
            cx, cy = self._f32(r2), self._f32(r3)
            ax, ay = getattr(v, 'anchor', (0.5, 0.5))
            self.anim.set_view_prop(v, 'frame', (cx - ax * v.frame[2], cy - ay * v.frame[3], v.frame[2], v.frame[3]))
            return True
        if sel_name in ('addSubview:', 'insertSubview:atIndex:', 'insertSubview:aboveSubview:',
                        'insertSubview:belowSubview:'):
            if r2:
                self._attach_subview(receiver, r2,
                                     index=r3 if sel_name == 'insertSubview:atIndex:' else None,
                                     above=r3 if sel_name == 'insertSubview:aboveSubview:' else None,
                                     below=r3 if sel_name == 'insertSubview:belowSubview:' else None)
                cv_ = self._get_view(r2)
                where = '' if sel_name == 'addSubview:' else f" [{sel_name}{' 0x%08X' % r3 if r3 else ''}]"
                logging.info(f"[UIKit] addSubview: {cv_.class_name}{'(' + cv_.custom_class + ')' if cv_.custom_class else ''}"
                             f" (0x{r2:08X}) -> 0x{receiver:08X}{where}")
            return True
        if sel_name in ('bringSubviewToFront:', 'sendSubviewToBack:'):
            pv_, cv_ = self.guest_views.get(receiver), self.guest_views.get(r2)
            if pv_ is not None and cv_ is not None and cv_ in pv_.children:
                pv_.children.remove(cv_)
                if sel_name == 'bringSubviewToFront:':
                    pv_.children.append(cv_)
                else:
                    pv_.children.insert(0, cv_)
                logging.info(f"[UIKit] {sel_name} {cv_.class_name}{'(' + cv_.custom_class + ')' if cv_.custom_class else ''} "
                             f"(0x{r2:08X}) in 0x{receiver:08X}")
            return True
        if sel_name == 'exchangeSubviewAtIndex:withSubviewAtIndex:':
            pv_ = self.guest_views.get(receiver)
            if pv_ is not None and r2 < len(pv_.children) and r3 < len(pv_.children):
                pv_.children[r2], pv_.children[r3] = pv_.children[r3], pv_.children[r2]
            return True
        if sel_name == 'removeFromSuperview':
            v = self.guest_views.get(receiver)
            if v is not None and v.parent is not None and v in v.parent.children:
                logging.info(f"[UIKit] removeFromSuperview: {v.class_name}"
                             f"{'(' + v.custom_class + ')' if v.custom_class else ''} (0x{receiver:08X})")
                v.parent.children.remove(v)
                v.parent = None
            return True
        if sel_name in ('setHidden:', 'setAlpha:', 'setImage:', 'setImage:forState:', 'setBackgroundImage:forState:',
                        'setText:', 'setTitle:forState:', 'setBackgroundColor:', 'setTextColor:'):
            v = self._get_view(receiver)
            if sel_name == 'setHidden:':
                v.hidden = bool(r2 & 0xFF)
            elif sel_name == 'setAlpha:':
                self.anim.set_view_prop(v, 'alpha', self._f32(r2))
            elif sel_name.startswith('set') and 'Image' in sel_name:
                img = self.image_objects.get(r2)
                if img is not None:
                    if sel_name.endswith('forState:'):
                        # -setBackgroundImage:forState: / -setImage:forState: (r3 = UIControlState).
                        # The app sets normal, highlighted and disabled artwork back to back; only
                        # the NORMAL one (state 0) is what a resting button shows. Previously each
                        # call overwrote the last, so the DISABLED image ended up on screen.
                        states = getattr(v, 'state_images', None)
                        if states is None:
                            states = v.state_images = {}
                        states[r3] = img
                        v.image_override = states.get(0) if 0 in states else states[min(states)]
                    else:
                        v.image_override = img
            elif sel_name in ('setText:', 'setTitle:forState:'):
                text = self._nsstring_at(r2)
                if text is not None:
                    v.text = text
            elif sel_name == 'setBackgroundColor:':
                if v.class_name != 'UITableView':          # tables stay white, like the stock Settings page
                    v.bg_color = self.color_objs.get(r2)
            elif sel_name == 'setTextColor:':
                v.text_color = self.color_objs.get(r2) or v.text_color
            uc_engine.reg_write(R0, receiver)
            return True
        if sel_name == 'isHidden' and receiver in self.guest_views:
            uc_engine.reg_write(R0, 1 if self.guest_views[receiver].hidden else 0)
            return True

        # --- controls ---
        if sel_name == 'buttonWithType:' and receiver in self.classes_by_addr:
            obj = self.objc_alloc(receiver)
            self._get_view(obj).class_name = 'UIButton'
            uc_engine.reg_write(R0, obj)
            return True
        if sel_name == 'addTarget:action:forControlEvents:':
            (events,) = self._stack_words(uc_engine, sp, 1)
            sel = self._cstring_at(r3)
            v = self._get_view(receiver)
            if r2 and sel:
                v.actions.append((events or 64, r2, sel))
                logging.info(f"[UIKit] {v.class_name} 0x{receiver:08X} action {sel} -> 0x{r2:08X} (events {events})")
            return True
        return False

    def _finish_draw(self):
        """Snapshot what a just-finished -drawRect: painted into that view's own layer."""
        if self._drawing_view:
            v = self._get_view(self._drawing_view)
            if v is not None:
                v.draw_surface = self.cg.get_surface().copy()
            self._drawing_view = 0

    def _bootstrap_views_from_class_metadata(self):
        """
        Real UIKit builds the view hierarchy from MainWindow.xib, which we
        cannot load (no NSKeyedUnarchiver/NIB implementation here - that's a
        separate, much larger subsystem). As a heuristic fallback when the
        app never explicitly alloc'd a view in code (the NIB-driven case),
        treat every LOCAL class that implements drawRect: as something that
        was meant to be on screen, and instantiate it ourselves. This is a
        guess, not a real object graph - no outlet wiring - but we DO give
        it a real frame via -initWithFrame: if it defines one (see
        _pending_view_inits): objc_alloc hands back zeroed memory, and
        skipping real initialization here is exactly the same bug we
        already found and fixed for the app delegate - any ivar the view's
        own drawRect: geometry math depends on (cached bounds, a fill
        level, whatever else -initWithFrame: sets up) would otherwise just
        be reading zeroed heap as if it were real state.
        """
        from objc_runtime import resolve_method
        candidates = []
        for class_addr, cls in self.classes_by_addr.items():
            if not cls['name'] or cls['name'].startswith('ExternalClass_'):
                continue
            if resolve_method(self.classes_by_addr, class_addr, "drawRect:"):
                candidates.append((class_addr, cls['name']))
        if not candidates:
            logging.warning("[UIKit] No local class implements drawRect:; nothing to bootstrap.")
            return
        logging.info(f"[UIKit] View hierarchy empty (no NIB loader) - "
                     f"bootstrapping drawable classes: {[n for _, n in candidates]}")
        for class_addr, cls_name in candidates:
            obj = self.objc_alloc(class_addr)
            self.view_hierarchy.append(obj)
            init_imp = resolve_method(self.classes_by_addr, class_addr, "initWithFrame:")
            if init_imp:
                logging.info(f"[UIKit] {cls_name}: queuing real -initWithFrame: "
                             f"before first draw (frame 0,0,{self.cg.width},{self.cg.height})")
                self._pending_view_inits.append((obj, init_imp))

    def _drive_pending_view_init(self, uc_engine):
        """
        Dispatches one queued -initWithFrame: call for real, with an
        actual CGRect(0, 0, screen_width, screen_height) argument - AAPCS
        packs origin.x/origin.y into r2/r3 (self/`_cmd` already own r0/r1),
        and size.width/size.height overflow onto the stack at the incoming
        [sp+0]/[sp+4], same as any other 4-float struct arg with only two
        registers left to hold it. Returns True if a call was dispatched
        (caller must return immediately - PC/LR were redirected into real
        app code), False if the queue was empty.
        """
        if not self._pending_view_inits:
            return False
        obj, imp = self._pending_view_inits.pop(0)
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        width_bits = struct.unpack('I', struct.pack('f', float(self.cg.width)))[0]
        height_bits = struct.unpack('I', struct.pack('f', float(self.cg.height)))[0]
        uc_engine.mem_write(sp, struct.pack('<II', width_bits, height_bits))
        uc_engine.reg_write(arm_const.UC_ARM_REG_R0, obj)          # self
        uc_engine.reg_write(arm_const.UC_ARM_REG_R1, 0)            # _cmd (unused by our dispatch)
        uc_engine.reg_write(arm_const.UC_ARM_REG_R2, 0)            # origin.x = 0.0
        uc_engine.reg_write(arm_const.UC_ARM_REG_R3, 0)            # origin.y = 0.0
        uc_engine.reg_write(arm_const.UC_ARM_REG_LR, self.stub_base + (15 * 4))
        uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
        return True

    def _dispatch_draw(self, uc_engine, view):
        """Jump into a view's real -drawRect: (with a real bounds rect argument). True if dispatched."""
        from objc_runtime import resolve_method
        class_addr = self._class_of(view)
        if class_addr is None:
            return False
        imp = resolve_method(self.classes_by_addr, class_addr, "drawRect:")
        if not imp:
            return False
        cls_name = self.classes_by_addr[class_addr]['name']
        self._log_call('UIKit', cls_name, 'drawRect:', imp)
        self._drawing_view = view
        sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
        uc_engine.mem_write(sp, struct.pack('<2f', float(self.cg.width), float(self.cg.height)))
        uc_engine.reg_write(arm_const.UC_ARM_REG_R0, view)
        uc_engine.reg_write(arm_const.UC_ARM_REG_R1, 0)
        uc_engine.reg_write(arm_const.UC_ARM_REG_R2, 0)   # rect.origin.x
        uc_engine.reg_write(arm_const.UC_ARM_REG_R3, 0)   # rect.origin.y (size on the stack)
        uc_engine.reg_write(arm_const.UC_ARM_REG_LR, self.stub_base + (15 * 4))
        uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
        return True

    def drive_draw_cycle(self, uc_engine):
        """
        Stand-in for the run loop's redraw pass. Views the app marked dirty
        with setNeedsDisplay are redrawn first; then any view that has never
        been drawn gets its first drawRect:. One dispatch per park re-entry.
        Returns True if a draw was dispatched this call.
        """
        while self._dirty_views:
            if self._dispatch_draw(uc_engine, self._dirty_views.pop(0)):
                return True
        while self._draw_queue_index < len(self.view_hierarchy):
            view = self.view_hierarchy[self._draw_queue_index]
            self._draw_queue_index += 1
            if self._dispatch_draw(uc_engine, view):
                return True
        return False

    def objc_dispatch(self, uc_engine, receiver, sel_name, lr):
        """
        Try to dispatch to a REAL method implementation, or at least a
        sensible framework shim, instead of blindly echoing the receiver.

        Returns one of:
          'jumped'   - PC/LR redirected into real app code; caller must
                       return immediately without touching any registers.
          'handled'  - R0 already holds the correct result; caller must run
                       the normal return-to-LR epilogue but NOT touch R0.
          'fallback' - nothing resolved; caller should echo receiver into R0
                       (the old "every message returns self" fiction).
        """
        from objc_runtime import resolve_method

        if not self.objc_ready or not receiver:
            return 'fallback'

        class_addr = self._class_of(receiver)

        # +alloc is ours, not app code - it has no IMP in the binary.
        if sel_name == "alloc" and class_addr is not None:
            obj = self.objc_alloc(class_addr)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, obj)
            return 'handled'

        if class_addr is not None:
            imp = resolve_method(self.classes_by_addr, class_addr, sel_name)
            if imp:
                cls_name = self.classes_by_addr[class_addr]['name']
                self._log_call('ObjC Dispatch', cls_name, sel_name, imp)
                if sel_name in self._WATCH_SELECTORS and os.environ.get('EMU_WATCH') == '1':
                    self._watch(uc_engine, cls_name, sel_name, imp)
                if os.environ.get('EMU_TRACE') == '1':
                    self._trace_msg(uc_engine, receiver, class_addr, sel_name, f'APP IMP 0x{imp:08X}',
                                    uc_engine.reg_read(arm_const.UC_ARM_REG_R2), uc_engine.reg_read(arm_const.UC_ARM_REG_R3))
                # Enter the real method: LR returns to the original call site;
                # PC keeps bit0 for ARM/Thumb interworking.
                uc_engine.reg_write(arm_const.UC_ARM_REG_LR, lr)
                uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
                return 'jumped'

        # No app-defined IMP: fall back to the UIKit shim so framework calls
        # still yield usable objects instead of nil.
        tracing = os.environ.get('EMU_TRACE') == '1'
        if tracing:
            t2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            t3 = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
        if self._uikit_shim(uc_engine, receiver, sel_name):
            if tracing:
                self._trace_msg(uc_engine, receiver, class_addr, sel_name, 'handled', t2, t3)
            return 'handled'

        self._log_unhandled(receiver, class_addr, sel_name)
        if tracing:
            self._trace_msg(uc_engine, receiver, class_addr, sel_name, 'FALLBACK', t2, t3)
        return 'fallback'

    def _hook_code_execution(self, uc_engine, address, size, user_data):
        if self.stub_base <= address < self.stub_base + 0x20000:
            stub_index = (address - self.stub_base) // 4
            self.handle_api_stub(uc_engine, stub_index)

    def handle_api_stub(self, uc_engine, index):
        lr = uc_engine.reg_read(arm_const.UC_ARM_REG_LR)
        
        if index == 0: # _glClearColor
            try:
                r0 = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
                r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                r3 = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
                r_val = struct.unpack('f', struct.pack('I', r0))[0]
                g_val = struct.unpack('f', struct.pack('I', r1))[0]
                b_val = struct.unpack('f', struct.pack('I', r2))[0]
                a_val = struct.unpack('f', struct.pack('I', r3))[0]
                if self.gpu:
                    self.gpu.gl_clear_color(r_val, g_val, b_val, a_val)
            except Exception:
                if self.gpu:
                    self.gpu.gl_clear_color(0.0, 0.0, 0.0, 1.0)
                    
        elif index == 1: # _glClear
            mask = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            if self.gpu:
                self.gpu.gl_clear(mask)
                
        elif index == 2: # _glDrawArrays
            mode = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            first = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            count = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            if self.gpu:
                self.gpu.gl_draw_arrays(mode, first, count)
                
        elif index == 3: # _glDrawElements
            mode = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            count = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            dtype = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            indices = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
            if self.gpu:
                self.gpu.gl_draw_elements(mode, count, dtype, indices)
                
        elif index == 4: # _glBindTexture
            target = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            texture = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            if self.gpu:
                self.gpu.gl_bind_texture(target, texture)
                
        elif index == 5: # _glGenTextures
            n = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            textures_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            if self.gpu and textures_ptr:
                tex_id = self.gpu.gl_gen_textures(1)
                uc_engine.mem_write(textures_ptr, struct.pack('<I', tex_id))
                
        elif index == 11: # _objc_msgSend
            receiver = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            selector_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            sel_name = None
            if self.bridge:
                sel_name = self.bridge.resolve_msg_send(receiver, selector_ptr)
            result = self.objc_dispatch(uc_engine, receiver, sel_name, lr) if sel_name else 'fallback'
            if result == 'jumped':
                # Dispatched into real app code; that method's own epilogue
                # returns to lr. Skip the stub-return epilogue below.
                return
            if result == 'fallback':
                # Unresolvable (framework class, no metadata): keep the old
                # fiction of returning the receiver, which keeps init/retain/
                # release chains self-consistent.
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, receiver)
            # result == 'handled': R0 already holds the correct value
            # (a freshly alloc'd object, or a UIKit shim's return) - do
            # NOT overwrite it here, that was the bug that made every
            # +alloc silently collapse back into the class pointer.

        elif index == 13: # _NSLog
            format_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            try:
                mem_bytes = bytes(uc_engine.mem_read(format_ptr, 256))
                log_str = mem_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')
                logging.info(f"[iOS NSLog] {log_str}")
            except Exception:
                logging.error("[iOS NSLog] <string read error>")
                
        elif index == 14: # _presentRenderbuffer
            if self.gpu:
                self.gpu.eagl_present_renderbuffer()
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, 1)

        elif index == 15: # _UIApplicationMain
            # Real UIApplicationMain never returns - it becomes the run loop.
            # We approximate startup as a small state machine, since this
            # same stub is the re-entry point every time a called method
            # returns to us:
            #   0: not started -> alloc delegate, call its -init if it has
            #      one (skipping this was the bug: window/viewController
            #      ivars set up in -init never got set, so the launch method
            #      operated on a half-built delegate and everything downstream
            #      read back nil)
            #   1: -init done -> call the launch method for real
            #   2: launch done -> park / drive draw cycle
            stage = getattr(self, '_uiapp_stage', 0)

            if stage == 0:
                self._uiapp_stage = 1
                from objc_runtime import find_app_delegate, resolve_method
                if not self._nib_loaded:
                    self._nib_loaded = True
                    try:
                        self.load_app_nibs()
                    except Exception as nib_err:
                        logging.error(f"[NIB] load_app_nibs failed: {nib_err}", exc_info=True)
                self._delegate_init_ran = False
                self._delegate_class, self._delegate_launch_sel = find_app_delegate(self.classes_by_addr)
                if self._nib_delegate:
                    self._delegate_obj, self._delegate_class, self._delegate_launch_sel = self._nib_delegate
                if self._delegate_class:
                    cls_name = self.classes_by_addr[self._delegate_class]['name']
                    if not self._nib_delegate:
                        self._delegate_obj = self.objc_alloc(self._delegate_class)
                    init_imp = None if self._nib_delegate else resolve_method(self.classes_by_addr, self._delegate_class, "init")
                    if init_imp:
                        self._delegate_init_ran = True
                        logging.info(f"[UIApplicationMain] {cls_name}: calling -init before launch")
                        uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self._delegate_obj)
                        uc_engine.reg_write(arm_const.UC_ARM_REG_R1, 0)
                        uc_engine.reg_write(arm_const.UC_ARM_REG_LR, self.stub_base + (15 * 4))
                        uc_engine.reg_write(arm_const.UC_ARM_REG_PC, init_imp)
                        return
                    # No override of -init: nothing to run, fall straight
                    # through to the launch-method stage below.
                    stage = 1
                    self._uiapp_stage = 1
                else:
                    logging.warning("[UIApplicationMain] No app delegate found in class metadata; parking.")
                    stage = 2
                    self._uiapp_stage = 2

            if stage == 1:
                self._uiapp_stage = 2
                if self._delegate_class and self._delegate_launch_sel:
                    cls_name = self.classes_by_addr[self._delegate_class]['name']
                    logging.info(f"[UIApplicationMain] Calling -[{cls_name} {self._delegate_launch_sel}]")
                    imp = self.classes_by_addr[self._delegate_class]['methods'][self._delegate_launch_sel]
                    # -init may have returned a different object (rare, but
                    # follow convention); use whatever's in R0 if init ran,
                    # otherwise the object we allocated.
                    # Only trust R0 if -init actually ran; otherwise it's just
                    # leftover argc/junk from UIApplicationMain's own call.
                    self_obj = (uc_engine.reg_read(arm_const.UC_ARM_REG_R0) if self._delegate_init_ran else 0) \
                               or self._delegate_obj
                    uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self_obj)
                    uc_engine.reg_write(arm_const.UC_ARM_REG_R1, 0)
                    uc_engine.reg_write(arm_const.UC_ARM_REG_R2, self._app_object())  # (UIApplication *)application
                    uc_engine.reg_write(arm_const.UC_ARM_REG_R3, 0)
                    uc_engine.reg_write(arm_const.UC_ARM_REG_LR, self.stub_base + (15 * 4))
                    uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
                    return

            # Launch has completed and we're back at the park stub. Bootstrap
            # the view hierarchy from class metadata on first entry (this
            # also queues any real -initWithFrame: calls those views need -
            # see _bootstrap_views_from_class_metadata), then drain any
            # pending init before ever asking a view to draw: its geometry
            # math has no business running before its own init has, same as
            # we already require for the app delegate above.
            self._finish_draw()
            if not self._views_bootstrapped:
                self._views_bootstrapped = True
                if not self.view_hierarchy and not self.nib_roots:
                    self._bootstrap_views_from_class_metadata()
            if self._drive_pending_view_init(uc_engine):
                return
            # Queued IBActions / timers / touches / lifecycle calls run next,
            # before any redraw they may have triggered via setNeedsDisplay.
            if self._drive_pending_actions(uc_engine):
                return
            # Stand in for the run loop's redraw pass: ask the app's own
            # views to draw, one per re-entry, until the queue is exhausted.
            if self.drive_draw_cycle(uc_engine):
                return
            if self.gpu:
                self.gpu.eagl_present_renderbuffer()
            # Do NOT reg_write(PC, stub_base + 15*4) here - that's a
            # self-jump to the exact address this UC_HOOK_CODE callback is
            # already executing at. Unicorn re-invokes the code hook
            # synchronously when PC never advances out of it, which
            # recurses in libunicorn's native C call stack (invisible to
            # Python's own recursion limit) until it overflows and the
            # whole process dies with an access violation - no Python
            # exception, no traceback, just silently back at the shell
            # prompt. That crash previously landed on exactly the first
            # entry into this fallback (first frame's draw queue drained).
            #
            # PC is already sitting at this same stub address, which is
            # exactly where we want it for the next frame - so just stop
            # this emu_start() slice cleanly. run_cpu_slice() re-reads PC
            # next call (from outside any hook) and resumes here safely.
            uc_engine.emu_stop()
            return

        elif index == 18: # _AudioServicesPlaySystemSound(SystemSoundID inSystemSoundID)
            sound_id = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            if self.audio:
                self.audio.play_system_sound(sound_id)
                logging.debug(f"[Audio] PlaySystemSound({sound_id})")
            else:
                logging.debug(f"[Audio] PlaySystemSound({sound_id}) - no audio engine attached")

        elif index == 19: # _AudioServicesCreateSystemSoundID(CFURLRef inFileURL, SystemSoundID *outSystemSoundID) -> OSStatus
            file_url = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            out_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            path = self._cfurl_paths.get(file_url) or self._resolve_resource_path(self.urls.get(file_url))
            sound_id = 0
            if self.audio and path:
                sound_id = self.audio.create_system_sound_id(path)
            elif not path:
                logging.warning(f"[Audio] CreateSystemSoundID: unknown CFURLRef 0x{file_url:08X}")
            try:
                if out_ptr:
                    uc_engine.mem_write(out_ptr, struct.pack('<I', sound_id))
            except Exception as e:
                logging.warning(f"[Audio] Failed writing SystemSoundID out-param: {e}")
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, 0)  # OSStatus noErr

        elif index == 20: # _UIGraphicsGetCurrentContext
            if self.cg:
                # Real UIKit gives -drawRect: a freshly re-transformed,
                # re-clipped context on every redraw pass. This emulator
                # keeps one persistent QuartzContext for the app's whole
                # lifetime (no CGContextSaveGState/RestoreGState), so this
                # is the one per-frame reset point: the app calls
                # UIGraphicsGetCurrentContext() first thing every time it
                # wants to draw, so resetting here keeps a degenerate clip
                # from one frame (see QuartzContext.clip()) from
                # permanently blanking every frame after it.
                self.cg.reset()
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self.cg.CONTEXT_HANDLE)
                self._cg_log("[CG] UIGraphicsGetCurrentContext (context reset for new frame)")

        elif index == 21: # _CGContextTranslateCTM(ctx, tx, ty)
            if self.cg:
                r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                tx = struct.unpack('f', struct.pack('I', r1))[0]
                ty = struct.unpack('f', struct.pack('I', r2))[0]
                self.cg.translate_ctm(tx, ty)
                self._cg_log(f"[CG] TranslateCTM({tx:.2f}, {ty:.2f})")

        elif index == 22: # _CGContextScaleCTM(ctx, sx, sy)
            if self.cg:
                r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                sx = struct.unpack('f', struct.pack('I', r1))[0]
                sy = struct.unpack('f', struct.pack('I', r2))[0]
                self.cg.scale_ctm(sx, sy)
                self._cg_log(f"[CG] ScaleCTM({sx:.2f}, {sy:.2f})")

        elif index == 23: # _CGContextMoveToPoint(ctx, x, y)
            if self.cg:
                r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                x = struct.unpack('f', struct.pack('I', r1))[0]
                y = struct.unpack('f', struct.pack('I', r2))[0]
                self.cg.move_to(x, y)
                self._cg_log(f"[CG] MoveToPoint({x:.2f}, {y:.2f})")

        elif index == 24: # _CGContextAddLineToPoint(ctx, x, y)
            if self.cg:
                r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                x = struct.unpack('f', struct.pack('I', r1))[0]
                y = struct.unpack('f', struct.pack('I', r2))[0]
                self.cg.add_line_to(x, y)
                self._cg_log(f"[CG] AddLineToPoint({x:.2f}, {y:.2f})")

        elif index == 25: # _CGContextClosePath(ctx)
            if self.cg:
                self.cg.close_path()
                self._cg_log("[CG] ClosePath")

        elif index == 26: # _CGContextClip(ctx)
            if self.cg:
                path_before = [(round(px, 1), round(py, 1)) for px, py in self.cg.current_path]
                self.cg.clip()
                self._cg_log(f"[CG] Clip -> rect now {self.cg.clip_rect}")
                if path_before != getattr(self, '_last_clip_path', None):
                    self._last_clip_path = path_before
                    left = self.__dict__.setdefault('_clip_change_log_left', 25)
                    if left > 0:
                        self._clip_change_log_left = left - 1
                        logging.info(f"[CG] clip shape changed: {path_before[:4]} -> visible rect {self.cg.clip_rect}"
                                     f"{' (tilted: per-pixel mask)' if self.cg.clip_mask is not None else ''}")

        elif index == 27: # _CGContextDrawImage(ctx, rect, image)
            # CGRect is 4 floats (origin.x/y, size.w/h) = 16 bytes. With ctx
            # in r0, only 3 of those 4 floats fit in r1-r3; size.height and
            # the image pointer both overflow onto the stack at [SP+0] and
            # [SP+4] per AAPCS.
            if self.cg:
                try:
                    r1 = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
                    r2 = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
                    r3 = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
                    sp = uc_engine.reg_read(arm_const.UC_ARM_REG_SP)
                    stack_words = struct.unpack('<II', uc_engine.mem_read(sp, 8))
                    height_bits, image_ptr = stack_words
                    x = struct.unpack('f', struct.pack('I', r1))[0]
                    y = struct.unpack('f', struct.pack('I', r2))[0]
                    w = struct.unpack('f', struct.pack('I', r3))[0]
                    h = struct.unpack('f', struct.pack('I', height_bits))[0]
                    surface = self.image_objects.get(image_ptr)
                    self.cg.draw_image(x, y, w, h, surface=surface)
                    kind = "real artwork" if surface is not None else "placeholder"
                    self._cg_log(f"[CG] DrawImage rect=({x:.2f},{y:.2f},{w:.2f},{h:.2f}) [{kind}]")
                except Exception as e:
                    logging.warning(f"[CG] DrawImage failed to decode args: {e}")

        elif index == 28: # _CGImageRetain(image) -> image
            # Plain C function (not an objc message), reached via the same
            # GOT-slot/stub mechanism as _NSLog/_malloc. We don't track real
            # retain counts, so this is a pure passthrough: CFRetain/
            # CGImageRetain conventionally return the same pointer they
            # were given. Previously this symbol had no stub index at all,
            # so its GOT slot was left as raw unpatched file bytes - calling
            # through it jumped into garbage (the classic "lands at 0x1000"
            # crash we've seen for every other unresolved symbol).
            pass  # R0 already holds the image pointer; leave it as-is.

        elif 29 <= index <= 46:
            # The __*vfp / __*sf2 / __*df2 family: compiler-generated
            # soft-float runtime helpers for float/double arithmetic,
            # comparison, and conversion (AAPCS "softfp" ABI - values travel
            # in general-purpose registers even though VFP hardware exists).
            # Every one of these has been unresolved since the very first
            # log in this whole debugging session; any float/double math
            # anywhere in the app (which CG drawing code is full of) calls
            # straight through them, landing in unpatched GOT garbage.
            #
            # Register convention: a float is one register; a double is a
            # register PAIR, low 32 bits first (r0:r1 for the first double
            # arg, r2:r3 for the second).
            def _f(reg):
                return struct.unpack('f', struct.pack('I', uc_engine.reg_read(reg)))[0]

            def _d(lo_reg, hi_reg):
                lo = uc_engine.reg_read(lo_reg)
                hi = uc_engine.reg_read(hi_reg)
                return struct.unpack('d', struct.pack('II', lo, hi))[0]

            def _set_f(val):
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, struct.unpack('I', struct.pack('f', val))[0])

            def _set_d(val):
                lo, hi = struct.unpack('II', struct.pack('d', val))
                uc_engine.reg_write(arm_const.UC_ARM_REG_R0, lo)
                uc_engine.reg_write(arm_const.UC_ARM_REG_R1, hi)

            R0, R1, R2, R3 = (arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R1,
                              arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3)
            try:
                if index == 29:    # __addsf3vfp(float,float)->float
                    _set_f(_f(R0) + _f(R1))
                elif index == 30:  # __subsf3vfp
                    _set_f(_f(R0) - _f(R1))
                elif index == 31:  # __mulsf3vfp
                    _set_f(_f(R0) * _f(R1))
                elif index == 32:  # __divsf3vfp
                    b = _f(R1)
                    _set_f(_f(R0) / b if b != 0 else float('inf'))
                elif index == 33:  # __adddf3vfp(double,double)->double, args r0:r1, r2:r3
                    _set_d(_d(R0, R1) + _d(R2, R3))
                elif index == 34:  # __subdf3vfp
                    _set_d(_d(R0, R1) - _d(R2, R3))
                elif index == 35:  # __muldf3vfp
                    _set_d(_d(R0, R1) * _d(R2, R3))
                elif index == 36:  # __divdf3vfp
                    b = _d(R2, R3)
                    _set_d(_d(R0, R1) / b if b != 0 else float('inf'))
                elif index == 37:  # __floatsisfvfp(int)->float
                    val = uc_engine.reg_read(R0)
                    if val & 0x80000000:
                        val -= 0x100000000
                    _set_f(float(val))
                elif index == 38:  # __fixsfsivfp(float)->int (truncate toward zero)
                    result = int(_f(R0))
                    uc_engine.reg_write(R0, result & 0xFFFFFFFF)
                elif index == 39:  # __fixdfsivfp(double)->int
                    result = int(_d(R0, R1))
                    uc_engine.reg_write(R0, result & 0xFFFFFFFF)
                elif index == 40:  # __truncdfsf2vfp(double)->float
                    _set_f(_d(R0, R1))
                elif index == 41:  # __extendsfdf2vfp(float)->double
                    _set_d(_f(R0))
                # The Darwin "...vfp" comparison helpers (compiler-rt arm/*vfp.S) return exactly ONE when the
                # relation holds and ZERO otherwise - the caller then does `cmp r0,#0 ; bne`. __eqsf2vfp here
                # used to return 0 when the floats were EQUAL, which inverted every float == test in the app.
                elif index == 42:  # __eqsf2vfp(float,float)->int, 1 iff a == b
                    uc_engine.reg_write(R0, 1 if _f(R0) == _f(R1) else 0)
                elif index == 43:  # __ltsf2vfp, 1 iff a < b
                    uc_engine.reg_write(R0, 1 if _f(R0) < _f(R1) else 0)
                elif index == 44:  # __gtsf2vfp, 1 iff a > b
                    uc_engine.reg_write(R0, 1 if _f(R0) > _f(R1) else 0)
                elif index == 45:  # __nesf2vfp, 1 iff a != b
                    uc_engine.reg_write(R0, 1 if _f(R0) != _f(R1) else 0)
                elif index == 46:  # __gtdf2vfp(double,double), 1 iff a > b
                    uc_engine.reg_write(R0, 1 if _d(R0, R1) > _d(R2, R3) else 0)
            except Exception as e:
                logging.warning(f"[SoftFloat] stub index {index} failed: {e}")

        elif index == 47: # _CFRelease(obj) -> void
            # We don't track real CoreFoundation retain counts, so this is a
            # pure no-op. Left unresolved since the very first log in this
            # session - the classic "retain the new value, release the old
            # one" pattern right after CGImageRetain is almost certainly
            # what was calling through this GOT slot's raw file garbage.
            pass

        elif index == 48: # _sinf(float)->float
            x = struct.unpack('f', struct.pack('I', uc_engine.reg_read(arm_const.UC_ARM_REG_R0)))[0]
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, struct.unpack('I', struct.pack('f', math.sin(x)))[0])

        elif index == 49: # _cosf(float)->float
            x = struct.unpack('f', struct.pack('I', uc_engine.reg_read(arm_const.UC_ARM_REG_R0)))[0]
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, struct.unpack('I', struct.pack('f', math.cos(x)))[0])

        elif index == 50: # _memcpy(dst, src, n) -> dst
            dst = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            src = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            n = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            if n > 0:
                try:
                    data = bytes(uc_engine.mem_read(src, n))
                    uc_engine.mem_write(dst, data)
                except Exception as e:
                    logging.warning(f"[memcpy] failed dst=0x{dst:08X} src=0x{src:08X} n={n}: {e}")
            # R0 already holds dst, which is memcpy's correct return value.

        elif index == 51: # _CFURLCreateFromFileSystemRepresentation(alloc, buffer, bufLen, isDirectory) -> CFURLRef
            buffer_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            buf_len = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            handle = 0
            try:
                raw_path = bytes(uc_engine.mem_read(buffer_ptr, buf_len)).decode('utf-8', errors='ignore')
                # The app's own path is a real-device path (/var/mobile/...)
                # that doesn't exist here - re-resolve just the filename
                # against the whole extracted bundle, same as UIImage
                # imageNamed: does, rather than trying to use it literally.
                basename = os.path.basename(raw_path)
                if self._resource_index is None:
                    self._resource_index = self._build_resource_index()
                real_path = self._resource_index.get(basename.lower())
                if real_path:
                    handle = self._next_cfurl_handle
                    self._next_cfurl_handle += 0x10
                    self._cfurl_paths[handle] = real_path
                    logging.debug(f"[CFURL] '{raw_path}' -> {real_path} (handle 0x{handle:08X})")
                else:
                    logging.warning(f"[CFURL] Could not resolve '{raw_path}' (basename '{basename}') "
                                     f"anywhere in the app bundle")
            except Exception as e:
                logging.warning(f"[CFURL] CreateFromFileSystemRepresentation failed: {e}")
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, handle)

        elif index == 52: # _AudioServicesDisposeSystemSoundID(SystemSoundID inSystemSoundID) -> OSStatus
            sound_id = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            if self.audio:
                self.audio.dispose_system_sound(sound_id)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, 0)  # OSStatus noErr

        elif index == 53: # _objc_msgSend_stret(ret_ptr, receiver, sel, ...) - CGPoint/CGRect returns
            from objc_runtime import resolve_method
            ret_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            receiver = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            selector_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            sel_name = self.bridge.resolve_msg_send(receiver, selector_ptr) if self.bridge else None
            cls = self._class_of(receiver) if receiver else None
            imp = resolve_method(self.classes_by_addr, cls, sel_name) if (cls and sel_name) else None
            if imp:
                uc_engine.reg_write(arm_const.UC_ARM_REG_LR, lr)
                uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)   # regs untouched: IMP expects (ret, self, _cmd, ...)
                return
            if sel_name:
                self._stret_shim(uc_engine, ret_ptr, receiver, sel_name)

        elif index == 54: # _objc_msgSendSuper2(struct objc_super *super, SEL op, ...)
            from objc_runtime import resolve_method
            super_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            selector_ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R1)
            try:
                receiver, cur_class = struct.unpack('<II', bytes(uc_engine.mem_read(super_ptr, 8)))
            except Exception:
                receiver, cur_class = 0, 0
            sel_name = self.bridge.resolve_msg_send(receiver, selector_ptr) if self.bridge else None
            # Super2: the struct holds the CURRENT class; lookup starts at its superclass.
            start = self.classes_by_addr.get(cur_class, {}).get('super', 0)
            imp = resolve_method(self.classes_by_addr, start, sel_name) if (start and sel_name) else None
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, receiver)
            if imp:
                uc_engine.reg_write(arm_const.UC_ARM_REG_LR, lr)
                uc_engine.reg_write(arm_const.UC_ARM_REG_PC, imp)
                return
            # Framework superclass (UIView/UIViewController/NSObject): nothing to run except UIView
            # behaviour we model ([super initWithFrame:] gives the view its frame); returning
            # self keeps [super init]-style chains correct.
            if sel_name:
                self._view_message(uc_engine, receiver, sel_name)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, receiver)

        elif index == 55: # _objc_setProperty(self, _cmd, offset, newValue, atomic, shouldCopy)
            obj = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            offset = uc_engine.reg_read(arm_const.UC_ARM_REG_R2)
            value = uc_engine.reg_read(arm_const.UC_ARM_REG_R3)
            try:
                uc_engine.mem_write(obj + offset, struct.pack('<I', value))
            except Exception as e:
                logging.warning(f"[objc_setProperty] {e}")

        elif index == 56 or index == 63: # _malloc / operator new
            size = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, self._c_alloc(size))

        elif index == 57 or index == 64: # _free / operator delete
            pass

        elif index == 58: # _strlen
            ptr = uc_engine.reg_read(arm_const.UC_ARM_REG_R0)
            n = 0
            try:
                while n < 65536 and bytes(uc_engine.mem_read(ptr + n, 1)) != b'\x00':
                    n += 1
            except Exception:
                pass
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, n)

        elif index == 59 or index == 60: # _puts / _printf (format string only; varargs not expanded)
            text = self._cstring_at(uc_engine.reg_read(arm_const.UC_ARM_REG_R0))
            logging.info(f"[iOS stdout] {text}")
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, len(text or ''))

        elif index == 61: # _exit
            logging.warning("[libc] app called exit()")

        elif index == 62: # _atan2(double y, double x)
            r0, r1, r2, r3 = (uc_engine.reg_read(x) for x in (arm_const.UC_ARM_REG_R0, arm_const.UC_ARM_REG_R1,
                                                              arm_const.UC_ARM_REG_R2, arm_const.UC_ARM_REG_R3))
            y = struct.unpack('<d', struct.pack('<II', r0, r1))[0]
            x = struct.unpack('<d', struct.pack('<II', r2, r3))[0]
            result = math.atan2(y, x)
            n = self.__dict__.get('_atan2_logged', 0)
            if n < 6:
                self._atan2_logged = n + 1
                logging.info(f"[Accel] app atan2(y={y!r}, x={x!r}) = {result:.4f} rad")
            lo, hi = struct.unpack('<II', struct.pack('<d', result))
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, lo)
            uc_engine.reg_write(arm_const.UC_ARM_REG_R1, hi)

        elif index in self.unimplemented_symbols and self.unimplemented_symbols[index] in (
                '_CACurrentMediaTime', '_CFAbsoluteTimeGetCurrent'):
            self._ret_double(uc_engine, time.monotonic() if self.unimplemented_symbols[index] == '_CACurrentMediaTime'
                             else time.time() - 978307200.0)

        elif index in self.unimplemented_symbols and self.openal.handle(uc_engine, self.unimplemented_symbols[index]):
            pass            # OpenAL / AudioFile / SjLj unwind registration: emulated by name

        elif index in self.unimplemented_symbols and self._math_stub(uc_engine, self.unimplemented_symbols[index]):
            pass            # soft-float helpers / libm: emulated by name, result already in r0 (r0:r1 for doubles)

        elif index in self.unimplemented_symbols and self._transform_stub(uc_engine, self.unimplemented_symbols[index]):
            pass            # CATransform3D*/CGAffineTransform*: emulated, result already in the hidden return struct

        elif index in self.unimplemented_symbols:
            name = self.unimplemented_symbols[index]
            if name not in self._warned_unimplemented:
                self._warned_unimplemented.add(name)
                logging.warning(f"[Stub] app called unimplemented {name} - returning 0")
            if name.startswith('_CATransform3D'):
                try:
                    uc_engine.mem_write(uc_engine.reg_read(arm_const.UC_ARM_REG_R0),
                                        struct.pack('<16f', 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1))
                except Exception:
                    pass
            uc_engine.reg_write(arm_const.UC_ARM_REG_R0, 0)

        # This redirect happens INSIDE a live emu_start() call (mid-hook),
        # not via run_cpu_slice's own restart path. In that context Unicorn's
        # ARM engine reads the Thumb switch off bit0 of the value written to
        # PC itself (mirroring real hardware BX/BLX semantics) - manually
        # toggling CPSR.T while writing a masked/even PC does not reliably
        # take effect for the very next instruction fetched in the same slice.
        uc_engine.reg_write(arm_const.UC_ARM_REG_PC, lr)

    def _hook_interrupt(self, uc_engine, intno, user_data):
        pc = uc_engine.reg_read(arm_const.UC_ARM_REG_PC)
        self.uc.reg_write(arm_const.UC_ARM_REG_R0, 0)
        self.uc.reg_write(arm_const.UC_ARM_REG_PC, pc + 4)

    def _hook_unmapped_memory(self, uc_engine, access, address, size, value, user_data):
        page_mask = 0xFFFFF000  
        page_addr = address & page_mask
        map_size = 0x10000      
        
        try:
            uc_engine.mem_map(page_addr, map_size, uc.UC_PROT_ALL)
            if access & uc.UC_MEM_FETCH_UNMAPPED:
                safe_exit_code = b'\x1E\xFF\x2F\xE1' * (map_size // 4)
                uc_engine.mem_write(page_addr, safe_exit_code)
            else:
                uc_engine.mem_write(page_addr, b'\x00' * map_size)
            return True
        except Exception:
            return False

    def _is_thumb(self, addr):
        return (addr & 1) == 1

    def run_cpu_slice(self, code_base, count=2000):
        pc = self.uc.reg_read(arm_const.UC_ARM_REG_PC)
        cpsr = self.uc.reg_read(arm_const.UC_ARM_REG_CPSR)
        
        start_pc = pc
        if cpsr & (1 << 5):
            start_pc |= 1
        else:
            start_pc &= ~1
            
        try:
            self.uc.emu_start(start_pc, 0xFFFFFFFF, timeout=0, count=count)
            return True
        except uc.UcError as e:
            pc = self.uc.reg_read(arm_const.UC_ARM_REG_PC)
            lr = self.uc.reg_read(arm_const.UC_ARM_REG_LR)
            sp = self.uc.reg_read(arm_const.UC_ARM_REG_SP)
            logging.error(f"[CPU Trapped] Error: {e} | PC: 0x{pc:08X} | LR: 0x{lr:08X} | SP: 0x{sp:08X}")
            
            try:
                opcode_bytes = bytes(self.uc.mem_read(pc & ~1, 4))
                opcode = struct.unpack('<I', opcode_bytes)[0]
                
                if opcode == 0:
                    if lr != 0:
                        if (lr & 1) or self._is_thumb(lr):
                            cpsr |= (1 << 5)
                        else:
                            cpsr &= ~(1 << 5)
                        self.uc.reg_write(arm_const.UC_ARM_REG_CPSR, cpsr)
                        self.uc.reg_write(arm_const.UC_ARM_REG_PC, lr & ~1)
                        return True
            except Exception:
                pass

            is_thumb = (cpsr & (1 << 5)) != 0
            step = 2 if is_thumb else 4
            target_pc = (pc + step) & ~1

            self.uc.reg_write(arm_const.UC_ARM_REG_PC, target_pc)
            return False
