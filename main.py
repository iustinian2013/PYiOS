# main.py

import sys
import os
import faulthandler
import math
import time
import struct
import logging
import zipfile
import plistlib
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # no pygame banner on launch
import pygame
import unicorn as uc
from cpu import ARMCpuCore
from memory import SystemMemory
from gpu import PowerVRGPU
from audio import IOSAudioEngine
from touch import TouchInput
from homescreen import IOSHomeScreen
from bridge import IOSRuntimeBridge
from coregraphics import QuartzContext
from objc_runtime import parse_classes, bind_class_refs

os.environ["UC_IGNORE_REG_BREAK"] = "1"
faulthandler.enable()   # a real native crash (Unicorn/SDL/GL) prints a Python traceback instead of exiting silently

# Quiet console by default. For development only: EMU_DEBUG=1 prints the emulator log, EMU_DEBUG=2 adds the full trace.
_dbg = os.environ.get("EMU_DEBUG", "")
logging.basicConfig(
    level=logging.DEBUG if _dbg == "2" else (logging.INFO if _dbg else logging.ERROR),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def setup_unmapped_memory_hook(uc_engine):
    def hook_mem_invalid(uc_engine, access, address, size, value, user_data):
        page_addr = address & ~0xFFF
        try:
            uc_engine.mem_map(page_addr, 0x1000)
            
            if access & uc.UC_MEM_FETCH_UNMAPPED:
                cpsr = uc_engine.reg_read(uc.arm_const.UC_ARM_REG_CPSR)
                is_thumb = (cpsr & (1 << 5)) != 0
                if is_thumb:
                    safety_code = b'\x70\x47' * (0x1000 // 2)
                else:
                    safety_code = b'\x1E\xFF\x2F\xE1' * (0x1000 // 4)  # BX LR = 0xE12FFF1E, little-endian in memory
                
                uc_engine.mem_write(page_addr, safety_code)
                logging.warning(f"[Auto-Mapper] Trapped instruction fetch at unmapped 0x{address:08X} (Thumb={is_thumb}). Populated page with return stubs.")
            else:
                uc_engine.mem_write(page_addr, b'\x00' * 0x1000)
                logging.info(f"[Auto-Mapper] Dynamically mapped zero-filled data page at 0x{page_addr:08X} (Fault: 0x{address:08X})")
                
            return True
        except Exception as e:
            logging.error(f"[Auto-Mapper] Failed to map page at 0x{page_addr:08X}: {e}")
            return False

    uc_engine.hook_add(
        uc.UC_HOOK_MEM_READ_UNMAPPED | 
        uc.UC_HOOK_MEM_WRITE_UNMAPPED | 
        uc.UC_HOOK_MEM_FETCH_UNMAPPED, 
        hook_mem_invalid
    )

def setup_stub_safety_hook(uc_engine, stub_base, stub_count=64):
    def hook_stub_fallback(uc_inst, address, size, user_data):
        lr = uc_inst.reg_read(uc.arm_const.UC_ARM_REG_LR)
        logging.warning(f"[Stub Fallback] Trapped unhandled call at stub address 0x{address:08X}. Automatically returning to LR: 0x{lr:08X}")
        uc_inst.reg_write(uc.arm_const.UC_ARM_REG_PC, lr)

    try:
        uc_engine.hook_add(
            uc.UC_HOOK_CODE, 
            hook_stub_fallback, 
            begin=stub_base, 
            end=stub_base + (stub_count * 4)
        )
    except Exception as e:
        logging.error(f"[Stub Hook] Failed to add stub safety hook: {e}")

class RealMachoLoader:
    @staticmethod
    def load(cpu_core, memory_mgr, binary_path, bridge_instance=None):
        if not os.path.exists(binary_path):
            logging.error(f"[Loader Error] Target path not found: {binary_path}")
            return None, None

        with open(binary_path, "rb") as f:
            data = f.read()

        if data[:4] == b'PK\x03\x04':
            logging.info("[IPA Container] Unpacking ZIP archive container...")
            try:
                with zipfile.ZipFile(binary_path, 'r') as zf:
                    namelist = zf.namelist()
                    app_bundles = [n for n in namelist if n.startswith('Payload/') and n.endswith('.app/')]
                    if not app_bundles:
                        logging.error("[IPA Error] No valid .app bundle found inside Payload/ folder.")
                        return None, None
                    
                    app_dir = app_bundles[0]
                    plist_name = app_dir + "Info.plist"
                    exec_name = None
                    
                    if plist_name in namelist:
                        plist_data = plistlib.loads(zf.read(plist_name))
                        exec_name = plist_data.get('CFBundleExecutable')
                        logging.info(f"[Info.plist] Resolved executable name: {exec_name}")
                    
                    if not exec_name:
                        candidates = [n for n in namelist if n.startswith(app_dir) and n.count('/') == 2 and '.' not in n.split('/')[-1]]
                        if candidates:
                            exec_name = candidates[0].split('/')[-1]
                    
                    if exec_name:
                        target_path = app_dir + exec_name
                        if target_path in namelist:
                            data = zf.read(target_path)
                            logging.info(f"[IPA Container] Successfully loaded binary payload: {exec_name} ({len(data)} bytes)")
            except Exception as zip_err:
                logging.error(f"[IPA Extraction Error] Failed to parse container: {zip_err}")
                return None, None

        magic = struct.unpack('<I', data[:4])[0]
        if magic != 0xfeedface:
            logging.warning(f"[Loader Warning] Non-standard magic: 0x{magic:08X}. Using fallback layout.")
            default_base = 0x00008000
            map_size = (len(data) + 0x20000 + 0xFFF) & ~0xFFF
            memory_mgr.map_region(default_base, map_size)
            memory_mgr.write_payload(default_base, data)
            return default_base, default_base

        ncmds = struct.unpack('<I', data[16:20])[0]
        cmd_offset = 28
        entry_point = None
        code_base = 0x00001000

        symbol_map = {
            "_glClearColor": cpu_core.stub_base + (0 * 4),
            "_glClear": cpu_core.stub_base + (1 * 4),
            "_glDrawArrays": cpu_core.stub_base + (2 * 4),
            "_glDrawElements": cpu_core.stub_base + (3 * 4),
            "_glBindTexture": cpu_core.stub_base + (4 * 4),
            "_glGenTextures": cpu_core.stub_base + (5 * 4),
            "_glTexImage2D": cpu_core.stub_base + (6 * 4),
            "_glTexParameteri": cpu_core.stub_base + (7 * 4),
            "_glEnable": cpu_core.stub_base + (8 * 4),
            "_glDisable": cpu_core.stub_base + (9 * 4),
            "_glViewport": cpu_core.stub_base + (10 * 4),
            "_objc_msgSend": cpu_core.stub_base + (11 * 4),
            "_objc_msgSendSuper": cpu_core.stub_base + (12 * 4),
            "_NSLog": cpu_core.stub_base + (13 * 4),
            "_presentRenderbuffer": cpu_core.stub_base + (14 * 4),
            "_UIApplicationMain": cpu_core.stub_base + (15 * 4),
            "_UIGraphicsBeginImageContext": cpu_core.stub_base + (16 * 4),
            "_UIGraphicsEndImageContext": cpu_core.stub_base + (17 * 4),
            "_AudioServicesPlaySystemSound": cpu_core.stub_base + (18 * 4),
            "_AudioServicesCreateSystemSoundID": cpu_core.stub_base + (19 * 4),
            "_UIGraphicsGetCurrentContext": cpu_core.stub_base + (20 * 4),
            "_CGContextTranslateCTM": cpu_core.stub_base + (21 * 4),
            "_CGContextScaleCTM": cpu_core.stub_base + (22 * 4),
            "_CGContextMoveToPoint": cpu_core.stub_base + (23 * 4),
            "_CGContextAddLineToPoint": cpu_core.stub_base + (24 * 4),
            "_CGContextClosePath": cpu_core.stub_base + (25 * 4),
            "_CGContextClip": cpu_core.stub_base + (26 * 4),
            "_CGContextDrawImage": cpu_core.stub_base + (27 * 4),
            "_CGImageRetain": cpu_core.stub_base + (28 * 4),
            "___addsf3vfp": cpu_core.stub_base + (29 * 4),
            "___subsf3vfp": cpu_core.stub_base + (30 * 4),
            "___mulsf3vfp": cpu_core.stub_base + (31 * 4),
            "___divsf3vfp": cpu_core.stub_base + (32 * 4),
            "___adddf3vfp": cpu_core.stub_base + (33 * 4),
            "___subdf3vfp": cpu_core.stub_base + (34 * 4),
            "___muldf3vfp": cpu_core.stub_base + (35 * 4),
            "___divdf3vfp": cpu_core.stub_base + (36 * 4),
            "___floatsisfvfp": cpu_core.stub_base + (37 * 4),
            "___fixsfsivfp": cpu_core.stub_base + (38 * 4),
            "___fixdfsivfp": cpu_core.stub_base + (39 * 4),
            "___truncdfsf2vfp": cpu_core.stub_base + (40 * 4),
            "___extendsfdf2vfp": cpu_core.stub_base + (41 * 4),
            "___eqsf2vfp": cpu_core.stub_base + (42 * 4),
            "___ltsf2vfp": cpu_core.stub_base + (43 * 4),
            "___gtsf2vfp": cpu_core.stub_base + (44 * 4),
            "___nesf2vfp": cpu_core.stub_base + (45 * 4),
            "___gtdf2vfp": cpu_core.stub_base + (46 * 4),
            "_CFRelease": cpu_core.stub_base + (47 * 4),
            "_sinf": cpu_core.stub_base + (48 * 4),
            "_cosf": cpu_core.stub_base + (49 * 4),
            "_memcpy": cpu_core.stub_base + (50 * 4),
            "_CFURLCreateFromFileSystemRepresentation": cpu_core.stub_base + (51 * 4),
            "_AudioServicesDisposeSystemSoundID": cpu_core.stub_base + (52 * 4),
            "_objc_msgSend_stret": cpu_core.stub_base + (53 * 4),
            "_objc_msgSendSuper2": cpu_core.stub_base + (54 * 4),
            "_objc_setProperty": cpu_core.stub_base + (55 * 4),
            "_malloc": cpu_core.stub_base + (56 * 4),
            "_free": cpu_core.stub_base + (57 * 4),
            "_strlen": cpu_core.stub_base + (58 * 4),
            "_puts": cpu_core.stub_base + (59 * 4),
            "_printf": cpu_core.stub_base + (60 * 4),
            "_exit": cpu_core.stub_base + (61 * 4),
            "_atan2": cpu_core.stub_base + (62 * 4),
            "__Znwm": cpu_core.stub_base + (63 * 4),
            "__ZdlPv": cpu_core.stub_base + (64 * 4),
        }

        # setup_stub_safety_hook() intentionally NOT called: it registered a second
        # UC_HOOK_CODE over the same stub_base range that ARMCpuCore._hook_code_execution
        # already owns. Its fallback wrote PC = lr directly (no low-bit mask, no CPSR
        # Thumb-bit update) immediately after handle_api_stub had already done the
        # correct masked write - pure redundant duplicate control-flow, fragile and
        # unnecessary since handle_api_stub already has its own correct default-return
        # path for any stub index it doesn't specifically handle.

        symtab_cmd = None
        dysymtab_cmd = None
        segments = []

        cmd_scan_offset = 28
        for i in range(ncmds):
            if cmd_scan_offset >= len(data):
                break
            cmd, cmdsize = struct.unpack('<II', data[cmd_scan_offset:cmd_scan_offset+8])
            
            if cmd == 0x1: # LC_SEGMENT
                seg_name = data[cmd_scan_offset+8:cmd_scan_offset+24].rstrip(b'\x00').decode('utf-8', errors='ignore')
                vmaddr, vmsize, fileoff, filesize = struct.unpack('<IIII', data[cmd_scan_offset+24:cmd_scan_offset+40])
                nsects = struct.unpack('<I', data[cmd_scan_offset+48:cmd_scan_offset+52])[0]
                
                sections = []
                sect_offset = cmd_scan_offset + 56
                for s in range(nsects):
                    sect_name = data[sect_offset:sect_offset+16].rstrip(b'\x00').decode('utf-8', errors='ignore')
                    # section struct = sectname[16] + segname[16] + addr,size,offset,align,
                    # reloff,nreloc,flags,reserved1,reserved2 (9 x uint32). Both name fields
                    # (32 bytes total) must be skipped before addr starts - skipping only
                    # sectname (16 bytes) read segname's ASCII bytes as addr/size/offset/align,
                    # shifting every field after that by one slot too (including reserved1,
                    # the indirect-symbol-table start index the GOT patch pass depends on).
                    s_addr, s_size, s_offset, s_align, s_reloff, s_nreloc, s_flags, s_reserved1, s_reserved2 = struct.unpack('<IIIIIIIII', data[sect_offset+32:sect_offset+68])
                    sections.append({'name': sect_name, 'addr': s_addr, 'size': s_size, 'offset': s_offset, 'reserved1': s_reserved1})
                    sect_offset += 68
                
                segments.append({'name': seg_name, 'vmaddr': vmaddr, 'vmsize': vmsize, 'fileoff': fileoff, 'filesize': filesize, 'sections': sections})
            elif cmd == 0x2: # LC_SYMTAB
                symtab_cmd = struct.unpack('<IIII', data[cmd_scan_offset+8:cmd_scan_offset+24])
            elif cmd == 0xb: # LC_DYSYMTAB
                # Full struct is 18 uint32 fields after cmd/cmdsize (72 bytes).
                # indirectsymoff/nindirectsyms are fields 12/13, NOT 4/5 -
                # unpacking only 12 fields (as before) truncated the struct
                # before those fields even existed in the tuple.
                dysymtab_cmd = struct.unpack('<18I', data[cmd_scan_offset+8:cmd_scan_offset+8+18*4])
            elif cmd == 0x5: # LC_UNIXTHREAD
                flavor, count = struct.unpack('<II', data[cmd_scan_offset+8:cmd_scan_offset+16])
                pc_offset = cmd_scan_offset + 16 + (15 * 4)
                if pc_offset + 4 <= len(data):
                    entry_point = struct.unpack('<I', data[pc_offset:pc_offset+4])[0]
                
            cmd_scan_offset += cmdsize

        for seg in segments:
            if seg['vmsize'] > 0:
                aligned_vmsize = (seg['vmsize'] + 0xFFF) & ~0xFFF
                memory_mgr.map_region(seg['vmaddr'], aligned_vmsize)
                seg_data = bytearray(aligned_vmsize)
                if seg['filesize'] > 0 and seg['fileoff'] < len(data):
                    actual_size = min(seg['filesize'], len(data) - seg['fileoff'], seg['vmsize'])
                    seg_data[:actual_size] = data[seg['fileoff']:seg['fileoff']+actual_size]
                memory_mgr.write_payload(seg['vmaddr'], bytes(seg_data))

        reloc_entries = []
        if symtab_cmd and dysymtab_cmd:
            symoff, nsyms, stroff, strsize = symtab_cmd
            indirectsymoff, nindirectsyms = dysymtab_cmd[12], dysymtab_cmd[13]
            string_table = data[stroff:stroff+strsize]

            # External relocations = what dyld would bind at load: which framework
            # symbol goes at which address (class refs, superclass pointers...).
            extreloff, nextrel = dysymtab_cmd[14], dysymtab_cmd[15]
            for r in range(nextrel):
                roff = extreloff + r * 8
                if roff + 8 > len(data):
                    break
                r_address, r_word = struct.unpack('<iI', data[roff:roff + 8])
                if not (r_word >> 27) & 1:      # r_extern
                    continue
                nl = symoff + (r_word & 0xFFFFFF) * 12
                if nl + 12 > len(data):
                    continue
                strx = struct.unpack('<I', data[nl:nl + 4])[0]
                end = string_table.find(b'\x00', strx)
                reloc_entries.append((r_address, string_table[strx:end].decode('utf-8', errors='ignore')))

            # Data symbols (not functions): give them real backing memory instead of
            # leaving the slot pointing at garbage.
            memory_mgr.map_region(0x74000000, 0x1000)
            identity = struct.pack('<16f', 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)
            memory_mgr.write_payload(0x74000100, identity)
            memory_mgr.write_payload(0x74000200, struct.pack('<6f', 1, 0, 0, 1, 0, 0))   # CGAffineTransformIdentity
            data_symbol_map = {
                "_CGRectZero": 0x74000000, "_CGPointZero": 0x74000000, "_CGSizeZero": 0x74000000,
                "_kCFAllocatorDefault": 0x74000000, "_CATransform3DIdentity": 0x74000100,
                "_CGAffineTransformIdentity": 0x74000200,
            }
            cpu_core._next_unimplemented = getattr(cpu_core, '_next_unimplemented', 100)

            # Diagnostic: list every section so we can see if this binary has an
            # inline-stub section (commonly __symbol_stub4 / __picsymbolstub4)
            # that we don't patch at all - those hold trampoline CODE, not a
            # pointer slot, and a call through one would still crash even
            # after every __la_symbol_ptr/__nl_symbol_ptr entry is patched.
            for seg in segments:
                for sect in seg['sections']:
                    logging.debug(f"[Section Scan] {seg['name']}/{sect['name']} addr=0x{sect['addr']:08X} size=0x{sect['size']:X} reserved1={sect['reserved1']}")

            for seg in segments:
                for sect in seg['sections']:
                    if sect['name'] == '__la_symbol_ptr' or sect['name'] == '__nl_symbol_ptr':
                        reserved1 = sect['reserved1']
                        ptr_count = sect['size'] // 4
                        
                        for i in range(ptr_count):
                            indirect_sym_index_offset = indirectsymoff + ((reserved1 + i) * 4)
                            if indirect_sym_index_offset + 4 > len(data):
                                continue
                            
                            sym_idx = struct.unpack('<I', data[indirect_sym_index_offset:indirect_sym_index_offset+4])[0]
                            nlist_offset = symoff + (sym_idx * 12)
                            if nlist_offset + 12 > len(data):
                                continue
                            
                            strx = struct.unpack('<I', data[nlist_offset:nlist_offset+4])[0]
                            null_term = string_table.find(b'\x00', strx)
                            sym_name = string_table[strx:null_term].decode('utf-8', errors='ignore')
                            target_slot_addr = sect['addr'] + (i * 4)

                            if sym_name in symbol_map:
                                stub_addr = symbol_map[sym_name]
                                memory_mgr.write_payload(target_slot_addr, struct.pack('<I', stub_addr))
                                logging.debug(f"[GOT Patch] Bound {sym_name} at pointer slot 0x{target_slot_addr:08X} -> Stub 0x{stub_addr:08X}")
                            else:
                                # Not in symbol_map at all - this is a call we have
                                # zero emulation for. Logged so we can see exactly
                                # what's missing instead of guessing from a crash trace.
                                if sym_name in data_symbol_map:
                                    memory_mgr.write_payload(target_slot_addr, struct.pack('<I', data_symbol_map[sym_name]))
                                    logging.debug(f"[GOT Patch] Data symbol {sym_name} -> 0x{data_symbol_map[sym_name]:08X}")
                                else:
                                    # Bind to a per-symbol logging stub that returns 0, instead of
                                    # leaving raw file bytes to be jumped into.
                                    idx = cpu_core._next_unimplemented
                                    cpu_core._next_unimplemented += 1
                                    cpu_core.unimplemented_symbols[idx] = sym_name
                                    memory_mgr.write_payload(target_slot_addr, struct.pack('<I', cpu_core.stub_base + idx * 4))
                                    logging.debug(f"[GOT Patch] {sym_name} at 0x{target_slot_addr:08X} -> unimplemented stub #{idx}")

        if not entry_point or entry_point < 0x1000:
            entry_point = 0x00001000 + 0x100

        logging.info(f"[Loader] Mach-O parsed. Entry Point: 0x{entry_point:08X}")
        classlist_section = None
        classrefs_section = None
        for seg in segments:
            for sect in seg['sections']:
                if sect['name'] == '__objc_classlist':
                    classlist_section = sect
                elif sect['name'] == '__objc_classrefs':
                    classrefs_section = sect
        if classlist_section:
            classes_by_addr, classes_by_name = parse_classes(data, segments, classlist_section)

            # dyld normally binds these at load time; without it every
            # [SomeClass foo] send reads a nil receiver and can't dispatch.
            # Turn external relocations into class names. r_address is relative to a
            # base that varies by linker; pick whichever candidate lands the
            # _OBJC_CLASS_$_ relocs inside __objc_classrefs.
            classref_names, super_names = {}, {}
            prefix = '_OBJC_CLASS_$_'
            class_syms = [(a, n[len(prefix):]) for a, n in reloc_entries if n.startswith(prefix)]
            cr_lo = classrefs_section['addr'] if classrefs_section else 0
            cr_hi = cr_lo + (classrefs_section['size'] if classrefs_section else 0)
            first_seg = min((s['vmaddr'] for s in segments if s['vmsize'] > 0 and s['name'] != '__PAGEZERO'), default=0)
            best_base, best_hits = 0, -1
            for cand in (0, first_seg, next((s['vmaddr'] for s in segments if s['name'] == '__DATA'), 0)):
                hits = sum(1 for a, _n in class_syms if cr_lo <= cand + a < cr_hi)
                if hits > best_hits:
                    best_base, best_hits = cand, hits
            for a, name in class_syms:
                addr = best_base + a
                if cr_lo <= addr < cr_hi:
                    classref_names[addr] = name
                elif (addr - 4) in classes_by_addr:      # class_t.superclass field
                    super_names[addr - 4] = name
            logging.info(f"[ObjC Runtime] Named {len(classref_names)} framework class refs and "
                         f"{len(super_names)} superclasses from relocations (base 0x{best_base:X}): "
                         f"{sorted(set(classref_names.values()))[:12]}")

            patches, synthetic = bind_class_refs(data, segments, classrefs_section, classes_by_addr,
                                                 classref_names=classref_names, super_names=super_names)
            classes_by_addr.update(synthetic)
            for a, c in synthetic.items():
                classes_by_name.setdefault(c['name'], a)
            cpu_core.external_class_addrs = set(synthetic.keys())
            for slot_addr, class_ptr in patches:
                memory_mgr.write_payload(slot_addr, struct.pack('<I', class_ptr))
            logging.info(f"[ObjC Runtime] Bound {len(patches)} class refs "
                         f"({len(synthetic)} external/synthetic).")

            cpu_core.install_objc_runtime(classes_by_addr, classes_by_name)
        else:
            logging.warning("[ObjC Runtime] No __objc_classlist section; dispatch will fall back to stubs.")

        return code_base, entry_point


class IOSEmulatorCoordinator:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        self.width, self.height = 320, 480
        self.window = pygame.display.set_mode((self.width * 2, self.height * 2))
        pygame.display.set_caption("PYiOS - iOS Emulator")
        
        self.clock = pygame.time.Clock()
        self._build_machine()

        self.homescreen = IOSHomeScreen(self.width, self.height)
        self._ignore_mouse_until = 0.0
        self.audio = None

        self.state = "HOME"
        self.running_app = None
        self.code_base = 0x00001000
        self.entry_point = 0x00001000
        logging.info("[Coordinator] Emulator Debug Environment Initialized.")

    def _build_machine(self):
        """
        Fresh CPU/GPU/memory/CG/touch state. Called once at startup and again
        on every app launch: ARMCpuCore.setup_registers() mem_map()s a fixed
        string page and all UIKit/NIB/lifecycle state lives on the core, so
        re-launching on the same core (ESC -> tap an icon again) raised
        UC_ERR_MAP and bounced straight back to the home screen. The GL
        context is also recreated by set_display_mode("APP"), so the old
        GPU object's texture ids / initialized flag would be stale anyway.
        """
        # Unicorn's mem_map(0xFF000000, ...) inside ARMCpuCore() trips a first-chance
        # access violation on Windows that Unicorn itself handles. faulthandler's
        # handler sees it before Unicorn does and prints a scary (harmless)
        # "Windows fatal exception" traceback, so keep it off just for construction.
        faulthandler.disable()
        try:
            self._build_machine_inner()
        finally:
            faulthandler.enable()

    def _build_machine_inner(self):
        self.gpu = PowerVRGPU(self.width, self.height)
        self.bridge = IOSRuntimeBridge(None, self.gpu)
        self.cg = QuartzContext(self.width, self.height)
        self.cpu = ARMCpuCore(gpu_instance=self.gpu, bridge_instance=self.bridge, cg_instance=self.cg)
        self.memory = SystemMemory(self.cpu.uc)
        setup_unmapped_memory_hook(self.cpu.uc)
        self.touch = TouchInput(self.cpu, (self.width, self.height))
        self.audio = None

    def set_display_mode(self, mode_type):
        if mode_type == "HOME":
            self.window = pygame.display.set_mode((self.width * 2, self.height * 2))
            pygame.display.set_caption("PYiOS - iOS Emulator")
        elif mode_type == "APP":
            # GL attributes must be set BEFORE set_mode() creates the context -
            # SDL only reads them at context-creation time. Setting them
            # afterward (as gpu.py's init_gl_context() used to do, called
            # after this window already existed) was a silent no-op, leaving
            # whatever default profile the driver handed back - quite
            # possibly a core profile that doesn't support the legacy
            # fixed-function GL calls (glMatrixMode, glVertexPointer,
            # glEnableClientState, etc.) this emulator relies on throughout.
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            self.window = pygame.display.set_mode((self.width, self.height), flags)

    def launch_app(self, launched_app):
        try:
            self._build_machine()
            self.running_app = launched_app
            binary_path = launched_app.get('binary_path', '')
            app_folder = os.path.dirname(binary_path) if binary_path else ""
            app_title = launched_app.get('name', 'iOS App')

            self.set_display_mode("APP")
            pygame.display.set_caption(f"{app_title}")

            logging.info(f"[Runtime Debug] Booting sandbox for: {app_title} (Target: {binary_path})")
            
            stack_base = 0x50000000
            self.memory.map_region(stack_base, 0x10000)

            self.bridge.uc = self.cpu.uc

            self.cpu.app_resource_dir = app_folder
            resolved_base, resolved_entry = RealMachoLoader.load(self.cpu, self.memory, binary_path, self.bridge)
            if resolved_base:
                self.code_base = resolved_base
                self.entry_point = resolved_entry

            self.cpu.setup_registers(self.entry_point, stack_base + 0xFF00, app_name=app_title)
            self.gpu.eagl_create_context()
            self.audio = IOSAudioEngine(app_folder)
            self.cpu.audio = self.audio  # so _AudioServicesPlaySystemSound/CreateSystemSoundID have something to call into
            logging.info(f"[Runtime Debug] CPU execution context primed at PC: 0x{self.entry_point:08X}")
        except Exception as e:
            logging.critical(f"[Critical Error] Failed to launch app: {e}", exc_info=True)
            self.set_display_mode("HOME")
            self.state = "HOME"

    def run(self):
        running = True
        while running:
            try:
                for event in pygame.event.get():
                    if self.state == "APP" and self.touch.handle_event(event, self.window.get_size()):
                        continue
                    if event.type == pygame.QUIT:
                        logging.info("[Runtime Debug] Window close requested (pygame.QUIT) - shutting down.")
                        running = False
                    elif event.type == pygame.FINGERDOWN and self.state == "HOME":
                        # Touchscreen tap on the home screen (SDL's synthesized mouse click is ignored below).
                        w, h = self.window.get_size()
                        launched_app = self.homescreen.handle_click((int(event.x * w), int(event.y * h)))
                        self._ignore_mouse_until = time.monotonic() + 0.4
                        if launched_app:
                            self.touch.cancel_all()
                            self.launch_app(launched_app)
                            self.state = "APP"
                    elif event.type == pygame.MOUSEBUTTONDOWN and self.state == "HOME":
                        if time.monotonic() < self._ignore_mouse_until:
                            continue
                        launched_app = self.homescreen.handle_click(event.pos)
                        if launched_app:
                            self.touch.cancel_all()
                            self.launch_app(launched_app)
                            self.state = "APP"
                    elif event.type == pygame.KEYDOWN and self.state == "APP":
                        if event.key == pygame.K_ESCAPE:
                            self.touch.cancel_all()
                            logging.info("[Runtime Debug] User returned to Home screen.")
                            self.set_display_mode("HOME")
                            self.state = "HOME"

                if self.state == "APP":
                    try:
                        pc = self.cpu.uc.reg_read(uc.arm_const.UC_ARM_REG_PC)
                        sp = self.cpu.uc.reg_read(uc.arm_const.UC_ARM_REG_SP)
                        lr = self.cpu.uc.reg_read(uc.arm_const.UC_ARM_REG_LR)
                        
                        try:
                            insn_bytes = self.cpu.uc.mem_read(pc, 4)
                            if pc != getattr(self, '_last_traced_pc', None):
                                logging.debug(f"[CPU Trace] PC=0x{pc:08X} | SP=0x{sp:08X} | LR=0x{lr:08X} | Opcode: {insn_bytes.hex()}")
                                self._last_traced_pc = pc
                        except Exception:
                            if pc != getattr(self, '_last_traced_pc', None):
                                logging.debug(f"[CPU Trace] PC=0x{pc:08X} (Memory unmapped or invalid)")
                                self._last_traced_pc = pc

                        self.cpu.accel.poll_input()
                        self.cpu.tick(time.monotonic())
                        # One slice runs one queued guest call (a redraw, an accelerometer sample, a timer...).
                        # Running only one per frame capped the app at <= fps calls/s in total, which is why tilt
                        # felt sluggish: keep going while work is queued, within a small time budget.
                        budget_end = time.perf_counter() + 0.012
                        for _slice in range(8):
                            success = self.cpu.run_cpu_slice(self.cpu.uc.reg_read(uc.arm_const.UC_ARM_REG_PC), count=20000)
                            if not success:
                                logging.warning(f"[Runtime CPU Warning] Slice execution halted or trapped at PC 0x{pc:08X}")
                                break
                            if not self.cpu.has_pending_work() or time.perf_counter() > budget_end:
                                break
                    except uc.UcError as uc_err:
                        logging.error(f"[Unicorn Trap] Emulation exception at PC 0x{pc:08X}: {uc_err}", exc_info=True)
                        self.state = "HOME"
                        self.set_display_mode("HOME")

                if self.state == "HOME":
                    self.window.fill((0, 0, 0))
                    self.homescreen.draw(self.window)
                elif self.state == "APP":
                    self.gpu.render_frame(self.window, active_app_name=(self.running_app['name'] if self.running_app else ""))
                    frame = self.cpu.compose_frame(self.cg.get_surface())
                    if frame is not self.cg.get_surface():
                        self.touch.draw_overlay(frame)
                    self.gpu.draw_cg_surface(frame)

                pygame.display.flip()
                self.clock.tick(60)
                if self.state == "APP" and logging.getLogger().isEnabledFor(logging.INFO):
                    self._perf_frames = getattr(self, '_perf_frames', 0) + 1
                    now_p = time.perf_counter()
                    t0 = getattr(self, '_perf_t0', now_p)
                    self._perf_t0 = t0
                    if now_p - t0 >= 5.0:
                        acc = self.cpu.accel
                        d0 = getattr(self, '_perf_d0', 0)
                        logging.info(f"[Perf] {self._perf_frames / (now_p - t0):.1f} fps | accelerometer "
                                     f"{(acc.delivered - d0) / (now_p - t0):.1f} samples/s (interval {acc.interval:.3f}s) "
                                     f"| tilt roll {math.degrees(acc.roll):+.0f} pitch {math.degrees(acc.pitch):+.0f} deg")
                        self._perf_frames, self._perf_t0, self._perf_d0 = 0, now_p, acc.delivered
            except Exception as e:
                logging.error(f"[Runtime Error] Unhandled exception in loop: {e}", exc_info=True)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        emulator = IOSEmulatorCoordinator()
        emulator.run()
    except Exception as fatal:
        logging.critical(f"[Fatal Crash] Startup failure: {fatal}", exc_info=True)
        sys.exit(1)