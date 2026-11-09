# libc.py
"""
A C library for emulated iPhone OS binaries (the games / apps that don't go through UIKit for everything):
string + memory routines, ctype, stdlib, stdio and POSIX file I/O mapped onto the app bundle and a private
Documents sandbox, printf/scanf families (including va_list forms), 64-bit integer division, pthread / semaphore
stand-ins, a minimal std::string and the usual C++ allocation operators.

handle(uc, name) is called with a symbol name (leading underscores ignored) for every import the loader couldn't
bind to a built-in stub, and returns True when it emulated the call (result already in r0 / r0:r1).

ARM softfp conventions apply: int args in r0-r3 then the stack, doubles in even register pairs, 64-bit results in r0:r1.
"""
import logging
import os
import random
import struct
import time

from unicorn import arm_const as A

STDOUT, STDERR, STDIN = 0x74002010, 0x74002020, 0x74002030     # fake FILE* values (see main.py data symbols)
RUNE_LOCALE_ADDR, STD_VARS_ADDR = 0x74001000, 0x74002000

# Darwin <runetype.h> classes
_A, _C, _D, _G, _L, _P, _S, _U, _X, _B, _R, _I, _T, _Q = (0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000,
                                                           0x8000, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x200000)


def rune_locale_bytes():
    """Contents of _DefaultRuneLocale for the 8-bit range (runetype / maplower / mapupper tables)."""
    runetype, lower, upper = [], [], []
    for c in range(256):
        t = 0
        ch = chr(c) if c < 128 else ''
        if ch.isalpha():
            t |= _A | _R | _G | _Q
            t |= _U if ch.isupper() else _L
        if ch.isdigit():
            t |= _D | _R | _G | _Q | _X
        if ch and ch in 'abcdefABCDEF':
            t |= _X
        if ch and ch in ' \t\n\v\f\r':
            t |= _S
        if ch == ' ' or ch == '\t':
            t |= _B
        if 32 <= c < 127:
            t |= _R | _G | _Q if ch != ' ' else _R | _Q
            if not ch.isalnum() and ch != ' ':
                t |= _P
        if c < 32 or c == 127:
            t |= _C
        if ch in ' \t':
            t |= _B
        runetype.append(t)
        lower.append(ord(ch.lower()) if ch else c)
        upper.append(ord(ch.upper()) if ch else c)
    head = b'RuneMagA' + b'NONE'.ljust(32, b'\x00') + struct.pack('<III', 0, 0, 0xFFFD)
    return head + struct.pack('<256I', *runetype) + struct.pack('<256I', *lower) + struct.pack('<256I', *upper)


class _Args:
    """Variadic argument reader: either straight from registers/stack, or from a va_list pointer."""

    def __init__(self, uc, first_slot=None, va_list=None):
        self.uc = uc
        self.sp = uc.reg_read(A.UC_ARM_REG_SP)
        self.slot = first_slot
        self.va = va_list

    def _slot_word(self, slot):
        if slot < 4:
            return self.uc.reg_read(A.UC_ARM_REG_R0 + slot)
        return struct.unpack('<I', bytes(self.uc.mem_read(self.sp + (slot - 4) * 4, 4)))[0]

    def word(self):
        if self.va is not None:
            w = struct.unpack('<I', bytes(self.uc.mem_read(self.va, 4)))[0]
            self.va += 4
            return w
        w = self._slot_word(self.slot)
        self.slot += 1
        return w

    def dword(self):
        if self.va is not None:
            self.va = (self.va + 7) & ~7
            lo, hi = struct.unpack('<II', bytes(self.uc.mem_read(self.va, 8)))
            self.va += 8
            return lo | (hi << 32)
        if self.slot % 2:
            self.slot += 1
        lo, hi = self._slot_word(self.slot), self._slot_word(self.slot + 1)
        self.slot += 2
        return lo | (hi << 32)


class LibC:
    def __init__(self, cpu):
        self.cpu = cpu
        self.files = {}          # FILE* -> {'f': pyfile, 'eof': bool, 'path': str}
        self.fds = {}            # int fd -> pyfile
        self._next_fd = 3
        self.errno_ptr = 0
        self._strtok_state = 0
        self._rand = random.Random(1)
        self._alloc_sizes = {}
        self._empty_rep = 0
        self.sandbox = None

    # ------------------------------------------------------------------ helpers
    def _sandbox_dir(self):
        if self.sandbox is None:
            root = self.cpu.app_resource_dir or os.getcwd()
            self.sandbox = os.path.join(os.path.dirname(os.path.dirname(root.rstrip('/\\'))) if self.cpu.app_resource_dir else root,
                                        'Sandbox')
            for sub in ('Documents', 'Library', 'tmp'):
                os.makedirs(os.path.join(self.sandbox, sub), exist_ok=True)
        return self.sandbox

    def documents_dir(self):
        return os.path.join(self._sandbox_dir(), 'Documents')

    def map_path(self, path, writing=False):
        """Device paths (/var/mobile/Applications/UUID/App.app/x, .../Documents/y) -> real files on this machine."""
        if not path:
            return path
        p = path.replace('\\', '/')
        app_dir = (self.cpu.app_resource_dir or '').replace('\\', '/').rstrip('/')
        if app_dir and p.startswith(app_dir):
            return path
        marker = '.app/'
        if marker in p:
            rel = p.split(marker, 1)[1]
            return os.path.join(self.cpu.app_resource_dir or '', *rel.split('/')) if rel else (self.cpu.app_resource_dir or '')
        for sub in ('Documents', 'Library', 'tmp'):
            key = f'/{sub}/'
            if key in p or p.endswith(f'/{sub}'):
                rel = p.split(key, 1)[1] if key in p else ''
                return os.path.join(self._sandbox_dir(), sub, *[x for x in rel.split('/') if x])
        if not os.path.isabs(p) and not (len(p) > 1 and p[1] == ':'):
            # relative path: the app's cwd is its bundle
            return os.path.join(self.cpu.app_resource_dir or '', *p.split('/'))
        if os.path.exists(path):
            return path
        found = self.cpu._resolve_resource_path(path)
        if found:
            return found
        return os.path.join(self._sandbox_dir(), 'Documents', os.path.basename(p)) if writing else path

    def cbytes(self, uc, ptr, limit=1 << 20):
        out = bytearray()
        while len(out) < limit:
            try:
                chunk = bytes(uc.mem_read(ptr + len(out), 256))
            except Exception:
                chunk = b''
                for i in range(256):
                    try:
                        chunk += bytes(uc.mem_read(ptr + len(out) + i, 1))
                    except Exception:
                        break
                if not chunk:
                    break
            nul = chunk.find(b'\x00')
            if nul >= 0:
                out += chunk[:nul]
                return bytes(out)
            out += chunk
        return bytes(out)

    def text(self, uc, ptr):
        return self.cbytes(uc, ptr).decode('utf-8', errors='replace') if ptr else ''

    def _errno(self, uc, value):
        if not self.errno_ptr:
            self.errno_ptr = self.cpu._c_alloc(4)
        uc.mem_write(self.errno_ptr, struct.pack('<I', value & 0xFFFFFFFF))

    # ------------------------------------------------------------------ printf engine
    def c_format(self, uc, fmt, args):
        out = bytearray()
        i, n = 0, len(fmt)
        while i < n:
            ch = fmt[i:i + 1]
            if ch != b'%':
                j = fmt.find(b'%', i)
                j = n if j < 0 else j
                out += fmt[i:j]
                i = j
                continue
            i += 1
            flags = ''
            while i < n and chr(fmt[i]) in '-+ 0#':
                flags += chr(fmt[i])
                i += 1
            width = ''
            if i < n and fmt[i:i + 1] == b'*':
                w = args.word()
                w = w - (1 << 32) if w & 0x80000000 else w
                if w < 0:
                    flags += '-'
                    w = -w
                width = str(w)
                i += 1
            else:
                while i < n and 48 <= fmt[i] <= 57:
                    width += chr(fmt[i])
                    i += 1
            prec = None
            if i < n and fmt[i:i + 1] == b'.':
                i += 1
                prec = ''
                if i < n and fmt[i:i + 1] == b'*':
                    prec = str(max(0, args.word()))
                    i += 1
                else:
                    while i < n and 48 <= fmt[i] <= 57:
                        prec += chr(fmt[i])
                        i += 1
                prec = prec or '0'
            length = ''
            while i < n and chr(fmt[i]) in 'hlLqjzt':
                length += chr(fmt[i])
                i += 1
            if i >= n:
                break
            conv = chr(fmt[i])
            i += 1
            spec = '%' + flags + width + ('.' + prec if prec is not None else '')
            try:
                piece = ''
                if conv == '%':
                    piece = '%'
                elif conv in 'di':
                    if length in ('ll', 'q', 'j'):
                        v = args.dword()
                        v = v - (1 << 64) if v & (1 << 63) else v
                    else:
                        v = args.word()
                        v = v - (1 << 32) if v & 0x80000000 else v
                        if length == 'h':
                            v = ((v + 0x8000) & 0xFFFF) - 0x8000
                        elif length == 'hh':
                            v = ((v + 0x80) & 0xFF) - 0x80
                    piece = (spec + 'd') % v
                elif conv in 'uoxX':
                    v = args.dword() if length in ('ll', 'q', 'j') else args.word()
                    if length == 'h':
                        v &= 0xFFFF
                    elif length == 'hh':
                        v &= 0xFF
                    piece = (spec + ('d' if conv == 'u' else conv)) % v
                elif conv == 'p':
                    piece = '0x%x' % args.word()
                elif conv in 'fFeEgGaA':
                    v = struct.unpack('<d', struct.pack('<Q', args.dword()))[0]
                    piece = (spec + (conv if conv not in 'aA' else 'g')) % v
                elif conv == 'c':
                    out += bytes([args.word() & 0xFF])
                    continue
                elif conv == 's':
                    ptr = args.word()
                    sv = self.cbytes(uc, ptr) if ptr else b'(null)'
                    if prec is not None:
                        sv = sv[:int(prec)]
                    piece = (spec.split('.')[0] + 's') % sv.decode('latin-1')
                elif conv == '@':
                    ptr = args.word()
                    sv = (self.cpu._str_of(ptr) or '(null)') if ptr else '(null)'
                    out += ((spec.split('.')[0] + 's') % sv).encode('utf-8', errors='replace')
                    continue
                elif conv == 'n':
                    args.word()
                else:
                    piece = '%' + conv
                out += piece.encode('latin-1', errors='replace')
            except Exception:
                pass
        return bytes(out)

    # ------------------------------------------------------------------ scanf engine (strings only)
    def c_scan(self, uc, text, fmt, args):
        pos, count = 0, 0
        i = 0
        while i < len(fmt):
            c = fmt[i]
            if c.isspace():
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                i += 1
                continue
            if c != '%':
                if pos < len(text) and text[pos] == c:
                    pos += 1
                    i += 1
                    continue
                break
            i += 1
            suppress = False
            if i < len(fmt) and fmt[i] == '*':
                suppress = True
                i += 1
            width = ''
            while i < len(fmt) and fmt[i].isdigit():
                width += fmt[i]
                i += 1
            length = ''
            while i < len(fmt) and fmt[i] in 'hlLqz':
                length += fmt[i]
                i += 1
            if i >= len(fmt):
                break
            conv = fmt[i]
            i += 1
            if conv != 'c':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
            if pos >= len(text) and conv != 'n':
                return count if count else -1
            maxw = int(width) if width else 1 << 30
            if conv in 'diuxXo':
                base = {'x': 16, 'X': 16, 'o': 8}.get(conv, 10)
                j = pos
                if j < len(text) and text[j] in '+-':
                    j += 1
                digits = '0123456789abcdefABCDEF'[:base if base <= 10 else 22]
                while j < len(text) and j - pos < maxw and text[j] in digits:
                    j += 1
                tok = text[pos:j]
                if not tok or tok in '+-':
                    break
                val = int(tok, base)
                pos = j
                if not suppress:
                    ptr = args.word()
                    size = 2 if length == 'h' else (1 if length == 'hh' else 4)
                    uc.mem_write(ptr, (val & ((1 << (8 * size)) - 1)).to_bytes(size, 'little'))
                    count += 1
            elif conv in 'feEgG':
                j = pos
                while j < len(text) and j - pos < maxw and (text[j] in '+-.eE' or text[j].isdigit()):
                    j += 1
                try:
                    val = float(text[pos:j])
                except ValueError:
                    break
                pos = j
                if not suppress:
                    ptr = args.word()
                    uc.mem_write(ptr, struct.pack('<d' if 'l' in length else '<f', val))
                    count += 1
            elif conv == 's':
                j = pos
                while j < len(text) and j - pos < maxw and not text[j].isspace():
                    j += 1
                tok = text[pos:j]
                pos = j
                if not suppress:
                    ptr = args.word()
                    uc.mem_write(ptr, tok.encode('latin-1', errors='replace') + b'\x00')
                    count += 1
            elif conv == 'c':
                tok = text[pos:pos + (int(width) if width else 1)]
                pos += len(tok)
                if not suppress:
                    ptr = args.word()
                    uc.mem_write(ptr, tok.encode('latin-1', errors='replace'))
                    count += 1
            else:
                break
        return count

    # ------------------------------------------------------------------ main entry
    def handle(self, uc, name):
        base = name.lstrip('_')
        cpu = self.cpu
        R = [A.UC_ARM_REG_R0, A.UC_ARM_REG_R1, A.UC_ARM_REG_R2, A.UC_ARM_REG_R3]
        r0, r1, r2, r3 = (uc.reg_read(x) for x in R)
        sp = uc.reg_read(A.UC_ARM_REG_SP)

        def ret(v):
            uc.reg_write(A.UC_ARM_REG_R0, v & 0xFFFFFFFF)

        def ret64(v):
            v &= 0xFFFFFFFFFFFFFFFF
            uc.reg_write(A.UC_ARM_REG_R0, v & 0xFFFFFFFF)
            uc.reg_write(A.UC_ARM_REG_R1, v >> 32)

        def retd(v):
            lo, hi = struct.unpack('<II', struct.pack('<d', float(v)))
            uc.reg_write(A.UC_ARM_REG_R0, lo)
            uc.reg_write(A.UC_ARM_REG_R1, hi)

        def sword(w):
            return w - (1 << 32) if w & 0x80000000 else w

        def wr(addr, data):
            if addr:
                uc.mem_write(addr, data)

        # ===== no-op / trivially successful =====
        if base in ('pthread_mutex_init', 'pthread_mutex_lock', 'pthread_mutex_unlock', 'pthread_mutex_destroy',
                    'sem_post', 'sem_wait', 'sem_trywait', 'sem_close', 'sem_unlink', 'usleep', 'sleep', 'atexit',
                    '_cxa_atexit', 'setbuf', 'fflush', 'ferror', 'objc_enumerationMutation', 'ZdaPv',
                    'DNSServiceRefDeallocate', 'AudioSessionSetActive', 'AudioSessionInitialize', 'AudioSessionSetProperty',
                    'AudioSessionGetProperty', 'freeifaddrs', 'if_freenameindex', 'Unwind_SjLj_Register',
                    'Unwind_SjLj_Unregister', 'ZNSsD2Ev'):
            ret(0)
            return True
        if base == 'sem_open':
            ret(cpu._c_alloc(8))
            return True
        if base == 'pthread_create':                       # no real threads: report success, don't run the routine
            wr(r0, struct.pack('<I', 1))
            logging.info(f"[libc] pthread_create(start=0x{r2:08X}) not run (single-threaded emulation)")
            ret(0)
            return True
        if base in ('isatty', 'select', 'socket', 'bind', 'recvfrom', 'sendto', 'fcntl', 'getifaddrs', 'if_nameindex',
                    'DNSServiceBrowse', 'DNSServiceProcessResult', 'DNSServiceQueryRecord', 'DNSServiceRegister',
                    'DNSServiceResolve', 'DNSServiceRefSockFD'):
            ret(0xFFFFFFFF if base in ('socket', 'bind', 'recvfrom', 'sendto', 'select', 'getifaddrs') else 0)
            return True
        if base == 'mmap':
            ret(0xFFFFFFFF)                                 # MAP_FAILED: callers fall back to read()
            return True
        if base == 'getenv':
            ret(0)
            return True
        if base in ('abort', 'assert_rtn'):
            logging.error(f"[libc] {base}() called{(': ' + self.text(uc, r3)) if base == 'assert_rtn' else ''}")
            ret(0)
            return True
        if base in ('setjmp', '_setjmp'):
            ret(0)
            return True
        if base in ('longjmp', '_longjmp'):
            logging.warning("[libc] longjmp is not supported")
            ret(1)
            return True
        if base == 'CFRunLoopGetMain':
            ret(0x74003000)
            return True
        if base == 'error':
            if not self.errno_ptr:
                self.errno_ptr = cpu._c_alloc(4)
            ret(self.errno_ptr)
            return True

        # ===== integer division helpers (ARMv6 has no divide instruction) =====
        if base == 'divsi3':
            a, b = sword(r0), sword(r1)
            ret(int(a / b) if b else 0)
            return True
        if base == 'modsi3':
            a, b = sword(r0), sword(r1)
            ret(a - int(a / b) * b if b else 0)
            return True
        if base == 'udivsi3':
            ret(r0 // r1 if r1 else 0)
            return True
        if base == 'umodsi3':
            ret(r0 % r1 if r1 else 0)
            return True
        s64 = lambda v: v - (1 << 64) if v & (1 << 63) else v
        if base in ('divdi3', 'udivdi3', 'moddi3', 'umoddi3'):
            a, b = r0 | (r1 << 32), r2 | (r3 << 32)
            if base in ('divdi3', 'moddi3'):
                a, b = s64(a), s64(b)
                if b == 0:
                    ret64(0)
                else:
                    q = abs(a) // abs(b)
                    q = -q if (a < 0) != (b < 0) else q
                    ret64(q if base == 'divdi3' else a - q * b)       # C truncates toward zero
            else:
                ret64((a // b if base == 'udivdi3' else a % b) if b else 0)
            return True
        if base in ('cmpdi2', 'ucmpdi2'):                      # libgcc: 0 = less, 1 = equal, 2 = greater
            a, b = r0 | (r1 << 32), r2 | (r3 << 32)
            if base == 'cmpdi2':
                a, b = s64(a), s64(b)
            ret(0 if a < b else (1 if a == b else 2))
            return True
        if base in ('floatdidf', 'floatundidf', 'floatdisf'):
            v = r0 | (r1 << 32)
            v = float(s64(v) if base != 'floatundidf' else v)
            if base == 'floatdisf':
                ret(struct.unpack('<I', struct.pack('<f', v))[0])
            else:
                retd(v)
            return True
        if base in ('fixdfdi', 'fixunsdfdi'):
            v = struct.unpack('<d', struct.pack('<II', r0, r1))[0]
            try:
                ret64(int(v) if base == 'fixdfdi' else max(0, int(v)))
            except (ValueError, OverflowError):
                ret64(0)
            return True
        if base in ('ashldi3', 'lshrdi3', 'ashrdi3'):
            a, n = r0 | (r1 << 32), r2 & 63
            ret64((a << n) if base == 'ashldi3' else ((a >> n) if base == 'lshrdi3' else (s64(a) >> n)))
            return True
        if base == 'muldi3':
            ret64((r0 | (r1 << 32)) * (r2 | (r3 << 32)))
            return True
        if base == 'mach_absolute_time':                       # nanoseconds (timebase below is 1/1)
            ret64(time.perf_counter_ns())
            return True
        if base == 'mach_timebase_info':
            wr(r0, struct.pack('<II', 1, 1))
            ret(0)
            return True
        if base == 'arc4random':
            ret(self._rand.getrandbits(32))
            return True

        # ===== ctype =====
        if base == 'maskrune':
            c = r0
            rt = struct.unpack_from('<I', rune_locale_bytes(), 52 + 4 * c)[0] if c < 256 else 0
            ret(1 if rt & r1 else 0)
            return True
        if base in ('tolower', 'toupper'):
            c = r0
            if c < 128:
                ret(ord(chr(c).lower() if base == 'tolower' else chr(c).upper()))
            else:
                ret(c)
            return True

        # ===== strings / memory =====
        if base == 'strcpy':
            wr(r0, self.cbytes(uc, r1) + b'\x00')
            ret(r0)
            return True
        if base == 'strncpy':
            s = self.cbytes(uc, r1, r2)[:r2]
            wr(r0, s.ljust(r2, b'\x00'))
            ret(r0)
            return True
        if base == 'strcat':
            d = self.cbytes(uc, r0)
            wr(r0 + len(d), self.cbytes(uc, r1) + b'\x00')
            ret(r0)
            return True
        if base == 'strncat':
            d = self.cbytes(uc, r0)
            wr(r0 + len(d), self.cbytes(uc, r1)[:r2] + b'\x00')
            ret(r0)
            return True
        if base in ('strcmp', 'strncmp', 'strcasecmp', 'strncasecmp'):
            a, b = self.cbytes(uc, r0), self.cbytes(uc, r1)
            if base.startswith('strn'):
                a, b = a[:r2], b[:r2]
            if 'case' in base:
                a, b = a.lower(), b.lower()
            ret(0 if a == b else (0xFFFFFFFF if a < b else 1))
            return True
        if base in ('strchr', 'strrchr'):
            s = self.cbytes(uc, r0)
            c = r1 & 0xFF
            idx = (s + b'\x00').find(bytes([c])) if base == 'strchr' else ((s + b'\x00').rfind(bytes([c])))
            ret(r0 + idx if idx >= 0 else 0)
            return True
        if base == 'strstr':
            idx = self.cbytes(uc, r0).find(self.cbytes(uc, r1))
            ret(r0 + idx if idx >= 0 else 0)
            return True
        if base == 'strdup':
            s = self.cbytes(uc, r0) + b'\x00'
            p = cpu._c_alloc(len(s))
            wr(p, s)
            ret(p)
            return True
        if base == 'strerror':
            msg = b'Unknown error\x00'
            p = cpu._c_alloc(len(msg))
            wr(p, msg)
            ret(p)
            return True
        if base == 'strtok':
            if r0:
                self._strtok_state = r0
            ptr = self._strtok_state
            delims = self.cbytes(uc, r1)
            if not ptr:
                ret(0)
                return True
            s = self.cbytes(uc, ptr)
            start = 0
            while start < len(s) and s[start] in delims:
                start += 1
            if start >= len(s):
                self._strtok_state = 0
                ret(0)
                return True
            end = start
            while end < len(s) and s[end] not in delims:
                end += 1
            if end < len(s):
                wr(ptr + end, b'\x00')
                self._strtok_state = ptr + end + 1
            else:
                self._strtok_state = 0
            ret(ptr + start)
            return True
        if base == 'strsep':
            sp_ptr = struct.unpack('<I', bytes(uc.mem_read(r0, 4)))[0]
            if not sp_ptr:
                ret(0)
                return True
            delims = self.cbytes(uc, r1)
            s = self.cbytes(uc, sp_ptr)
            end = 0
            while end < len(s) and s[end] not in delims:
                end += 1
            if end < len(s):
                wr(sp_ptr + end, b'\x00')
                wr(r0, struct.pack('<I', sp_ptr + end + 1))
            else:
                wr(r0, struct.pack('<I', 0))
            ret(sp_ptr)
            return True
        if base == 'memcmp':
            a, b = bytes(uc.mem_read(r0, r2)) if r2 else b'', bytes(uc.mem_read(r1, r2)) if r2 else b''
            ret(0 if a == b else (0xFFFFFFFF if a < b else 1))
            return True
        if base in ('memmove', 'memcpy_chk'):
            if r2:
                wr(r0, bytes(uc.mem_read(r1, r2)))
            ret(r0)
            return True
        if base == 'memset':
            if r2:
                wr(r0, bytes([r1 & 0xFF]) * r2)
            ret(r0)
            return True
        if base == 'bzero':
            if r1:
                wr(r0, b'\x00' * r1)
            return True
        if base == 'memchr':
            data = bytes(uc.mem_read(r0, r2)) if r2 else b''
            idx = data.find(bytes([r1 & 0xFF]))
            ret(r0 + idx if idx >= 0 else 0)
            return True
        if base in ('atoi', 'atol'):
            try:
                s = self.text(uc, r0).strip()
                num = ''
                for k, ch in enumerate(s):
                    if ch.isdigit() or (k == 0 and ch in '+-'):
                        num += ch
                    else:
                        break
                ret(int(num) if num not in ('', '+', '-') else 0)
            except Exception:
                ret(0)
            return True
        if base == 'atof':
            import re
            m = re.match(r'\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?', self.text(uc, r0))
            retd(float(m.group()) if m else 0.0)
            return True
        if base in ('strtol', 'strtoul'):
            s = self.text(uc, r0)
            b = r2 or 10
            k = 0
            while k < len(s) and s[k].isspace():
                k += 1
            j = k
            if j < len(s) and s[j] in '+-':
                j += 1
            if b in (0, 16) and s[j:j + 2].lower() == '0x':
                j += 2
                b = 16
            elif b == 0:
                b = 8 if s[j:j + 1] == '0' else 10
            digs = '0123456789abcdefghijklmnopqrstuvwxyz'[:b]
            e = j
            while e < len(s) and s[e].lower() in digs:
                e += 1
            try:
                val = int(s[k:e].replace('0x', '').replace('0X', '') or '0', b) if e > j else 0
                if s[k:k + 1] == '-' and val > 0:
                    val = -val
            except ValueError:
                val = 0
            if r1:
                wr(r1, struct.pack('<I', r0 + (e if e > j else 0)))
            ret(val)
            return True
        if base == 'abs':
            ret(abs(sword(r0)))
            return True
        if base == 'labs':
            ret(abs(sword(r0)))
            return True
        if base == 'rand':
            ret(self._rand.randrange(0, 0x7FFFFFFF))
            return True
        if base == 'srand':
            self._rand.seed(r0)
            return True
        if base == 'realloc':
            if r1 == 0:
                ret(0)
                return True
            new = cpu._c_alloc(r1)
            if r0:
                old_size = self._alloc_sizes.get(r0, r1)
                try:
                    wr(new, bytes(uc.mem_read(r0, min(old_size, r1))))
                except Exception:
                    pass
            self._alloc_sizes[new] = r1
            ret(new)
            return True
        if base in ('Znam', 'Znwm', 'malloc_array'):
            p = cpu._c_alloc(r0)
            self._alloc_sizes[p] = r0
            ret(p)
            return True
        if base == 'gettimeofday':
            now = time.time()
            wr(r0, struct.pack('<II', int(now), int((now % 1) * 1e6)))
            ret(0)
            return True
        if base == 'qsort':
            logging.warning("[libc] qsort(): comparator callbacks aren't supported yet - array left unsorted")
            return True
        if base == 'NSSearchPathForDirectoriesInDomains':
            arr = cpu.objc_alloc(0)
            cpu.arrays[arr] = [cpu._new_string(self.documents_dir())]
            ret(arr)
            return True

        # ===== std::string (libstdc++ copy-on-write layout: the object is one pointer to the characters) =====
        if base == 'ZNSsC2Ev':
            if not self._empty_rep:
                self._empty_rep = cpu._c_alloc(16)
                wr(self._empty_rep, struct.pack('<III', 0, 0, 0xFFFFFFFF) + b'\x00')
            wr(r0, struct.pack('<I', self._empty_rep + 12))
            ret(r0)
            return True
        if base == 'ZNSs6assignEPKcm':                       # std::string::assign(const char*, size_t)
            data = bytes(uc.mem_read(r1, r2)) if r2 else b''
            rep = cpu._c_alloc(12 + r2 + 1)
            wr(rep, struct.pack('<III', r2, r2, 0) + data + b'\x00')
            wr(r0, struct.pack('<I', rep + 12))
            ret(r0)
            return True

        # ===== printf / scanf =====
        if base in ('printf', 'vprintf', 'puts', 'putchar', 'fprintf', 'vfprintf', 'perror'):
            if base == 'puts':
                text = self.cbytes(uc, r0)
            elif base == 'putchar':
                text = bytes([r0 & 0xFF])
            elif base == 'perror':
                text = self.cbytes(uc, r0) + b': error'
            elif base == 'printf':
                text = self.c_format(uc, self.cbytes(uc, r0), _Args(uc, first_slot=1))
            elif base == 'vprintf':
                text = self.c_format(uc, self.cbytes(uc, r0), _Args(uc, va_list=r1))
            elif base == 'fprintf':
                text = self.c_format(uc, self.cbytes(uc, r1), _Args(uc, first_slot=2))
            else:
                text = self.c_format(uc, self.cbytes(uc, r1), _Args(uc, va_list=r2))
            target = r0 if base in ('fprintf', 'vfprintf') else STDOUT
            if target in self.files:
                self.files[target]['f'].write(text)
            else:
                logging.info(f"[iOS stdout] {text.decode('utf-8', errors='replace').rstrip()}")
            ret(len(text))
            return True
        if base in ('sprintf', 'snprintf', 'vsnprintf', 'vsprintf'):
            if base == 'sprintf':
                out = self.c_format(uc, self.cbytes(uc, r1), _Args(uc, first_slot=2))
                limit = None
            elif base == 'snprintf':
                out = self.c_format(uc, self.cbytes(uc, r2), _Args(uc, first_slot=3))
                limit = r1
            elif base == 'vsprintf':
                out = self.c_format(uc, self.cbytes(uc, r1), _Args(uc, va_list=r2))
                limit = None
            else:
                out = self.c_format(uc, self.cbytes(uc, r2), _Args(uc, va_list=r3))
                limit = r1
            if limit is not None:
                clipped = out[:max(0, limit - 1)] if limit else b''
                if limit:
                    wr(r0, clipped + b'\x00')
            else:
                wr(r0, out + b'\x00')
            ret(len(out))
            return True
        if base in ('sscanf', 'fscanf'):
            if base == 'sscanf':
                ret(self.c_scan(uc, self.text(uc, r0), self.text(uc, r1), _Args(uc, first_slot=2)))
            else:
                ret(0xFFFFFFFF)
            return True

        # ===== stdio =====
        if base == 'fopen':
            path, mode = self.text(uc, r0), self.text(uc, r1)
            writing = any(m in mode for m in 'wa+')
            real = self.map_path(path, writing)
            pymode = mode.replace('b', '').replace('t', '')
            pymode = {'r': 'rb', 'w': 'wb', 'a': 'ab', 'r+': 'r+b', 'w+': 'w+b', 'a+': 'a+b'}.get(pymode, 'rb')
            try:
                if writing:
                    os.makedirs(os.path.dirname(real) or '.', exist_ok=True)
                f = open(real, pymode)
            except OSError as e:
                logging.info(f"[libc] fopen({path!r}, {mode!r}) failed: {e.strerror}")
                self._errno(uc, 2)
                ret(0)
                return True
            handle = cpu._c_alloc(16)
            self.files[handle] = {'f': f, 'eof': False, 'path': real}
            ret(handle)
            return True
        if base == 'fclose':
            fl = self.files.pop(r0, None)
            if fl:
                fl['f'].close()
            ret(0)
            return True
        if base == 'fread':
            fl = self.files.get(r3)
            if fl is None or r1 == 0 or r2 == 0:
                ret(0)
                return True
            data = fl['f'].read(r1 * r2)
            if len(data) < r1 * r2:
                fl['eof'] = True
            wr(r0, data)
            ret(len(data) // r1)
            return True
        if base == 'fwrite':
            fl = self.files.get(r3)
            data = bytes(uc.mem_read(r0, r1 * r2)) if r1 * r2 else b''
            if fl is None:
                if r3 in (STDOUT, STDERR):
                    logging.info(f"[iOS stdout] {data.decode('utf-8', errors='replace').rstrip()}")
                    ret(r2)
                else:
                    ret(0)
                return True
            fl['f'].write(data)
            ret(r2)
            return True
        if base == 'fseek':
            fl = self.files.get(r0)
            if fl is None:
                ret(0xFFFFFFFF)
                return True
            try:
                fl['f'].seek(sword(r1), r2)
            except (OSError, ValueError):
                self._errno(uc, 22)
                ret(0xFFFFFFFF)
                return True
            fl['eof'] = False
            ret(0)
            return True
        if base == 'ftell':
            fl = self.files.get(r0)
            ret(fl['f'].tell() if fl else 0xFFFFFFFF)
            return True
        if base == 'feof':
            fl = self.files.get(r0)
            ret(1 if fl and fl['eof'] else 0)
            return True
        if base == 'fgetc':
            fl = self.files.get(r0)
            b = fl['f'].read(1) if fl else b''
            if not b:
                if fl:
                    fl['eof'] = True
                ret(0xFFFFFFFF)
            else:
                ret(b[0])
            return True
        if base == 'fgets':
            fl = self.files.get(r2)
            if fl is None or r1 <= 1:
                ret(0)
                return True
            line = fl['f'].readline(r1 - 1)
            if not line:
                fl['eof'] = True
                ret(0)
                return True
            wr(r0, line + b'\x00')
            ret(r0)
            return True
        if base == 'fputc':
            fl = self.files.get(r1)
            if fl:
                fl['f'].write(bytes([r0 & 0xFF]))
            ret(r0 & 0xFF)
            return True
        if base == 'fputs':
            fl = self.files.get(r1)
            data = self.cbytes(uc, r0)
            if fl:
                fl['f'].write(data)
            else:
                logging.info(f"[iOS stdout] {data.decode('utf-8', errors='replace').rstrip()}")
            ret(0)
            return True
        if base == 'remove':
            try:
                os.remove(self.map_path(self.text(uc, r0), True))
                ret(0)
            except OSError:
                ret(0xFFFFFFFF)
            return True

        # ===== POSIX files =====
        if base in ('stat', 'fstat', 'access'):
            if base == 'fstat':
                f = self.fds.get(r0)
                real = f.name if f else None
            else:
                real = self.map_path(self.text(uc, r0))
            if not real or not os.path.exists(real):
                self._errno(uc, 2)
                ret(0xFFFFFFFF)
                return True
            if base != 'access':
                st = os.stat(real)
                mode = (0x4000 | 0o755) if os.path.isdir(real) else (0x8000 | 0o644)
                buf = bytearray(96)
                struct.pack_into('<iIHH', buf, 0, 1, 1, mode, 1)
                struct.pack_into('<q', buf, 48, st.st_size)
                struct.pack_into('<q', buf, 56, (st.st_size + 511) // 512)
                struct.pack_into('<i', buf, 64, 4096)
                wr(r1, bytes(buf))
            ret(0)
            return True
        if base == 'open':
            path, flags = self.text(uc, r0), r1
            writing = bool(flags & 3) or bool(flags & 0x200)
            real = self.map_path(path, writing)
            try:
                if flags & 0x200 and not os.path.exists(real):
                    os.makedirs(os.path.dirname(real) or '.', exist_ok=True)
                    open(real, 'wb').close()
                mode = 'r+b' if flags & 3 == 2 else ('r+b' if flags & 3 == 1 else 'rb')
                f = open(real, mode)
                if flags & 0x400:
                    f.truncate(0)
            except OSError:
                self._errno(uc, 2)
                ret(0xFFFFFFFF)
                return True
            fd = self._next_fd
            self._next_fd += 1
            self.fds[fd] = f
            ret(fd)
            return True
        if base == 'close':
            f = self.fds.pop(r0, None)
            if f:
                f.close()
            ret(0)
            return True
        if base == 'read':
            f = self.fds.get(r0)
            if f is None:
                ret(0)
                return True
            data = f.read(r2)
            wr(r1, data)
            ret(len(data))
            return True
        if base == 'write':
            f = self.fds.get(r0)
            data = bytes(uc.mem_read(r1, r2)) if r2 else b''
            if f is not None:
                f.write(data)
            elif r0 in (1, 2):
                logging.info(f"[iOS stdout] {data.decode('utf-8', errors='replace').rstrip()}")
            ret(r2)
            return True
        if base == 'lseek':
            # Apple's ARM ABI does NOT 8-byte-align 64-bit arguments (AAPCS does): lseek(fd, off_t off, int whence)
            # arrives as r0=fd, r1:r2=off, r3=whence. Fall back to the AAPCS layout (off in r2:r3, whence on the
            # stack) only when r3 isn't a valid whence.
            f = self.fds.get(r0)
            if r3 in (0, 1, 2):
                off, whence = r1 | (r2 << 32), r3
            else:
                (whence,) = cpu._stack_words(uc, sp, 1)
                off = r2 | (r3 << 32)
            off = off - (1 << 64) if off & (1 << 63) else off
            if f is None or whence not in (0, 1, 2):
                self._errno(uc, 9 if f is None else 22)
                ret64(0xFFFFFFFFFFFFFFFF)
                return True
            try:
                ret64(f.seek(off, whence))
            except (OSError, ValueError):
                self._errno(uc, 22)
                ret64(0xFFFFFFFFFFFFFFFF)
            return True
        return False
