# objc_runtime.py
import struct


def _addr_to_fileoff(addr, segments):
    for seg in segments:
        if seg['vmaddr'] <= addr < seg['vmaddr'] + seg['vmsize']:
            file_off = seg['fileoff'] + (addr - seg['vmaddr'])
            if file_off < seg['fileoff'] + seg['filesize']:
                return file_off
    return None


def _read_u32(data, fileoff):
    if fileoff is None or fileoff + 4 > len(data):
        return None
    return struct.unpack('<I', data[fileoff:fileoff + 4])[0]


def _read_cstring(data, fileoff, max_len=256):
    if fileoff is None or fileoff >= len(data):
        return ""
    end = data.find(b'\x00', fileoff, fileoff + max_len)
    if end == -1:
        end = min(fileoff + max_len, len(data))
    return data[fileoff:end].decode('utf-8', errors='ignore')


def _read_ptr_at_addr(data, addr, segments):
    return _read_u32(data, _addr_to_fileoff(addr, segments))


def _read_cstring_at_addr(data, addr, segments):
    return _read_cstring(data, _addr_to_fileoff(addr, segments))


def _parse_method_list(data, addr, segments):
    """Returns {selector_name: imp_addr} for one method_list_t at `addr`."""
    methods = {}
    if not addr:
        return methods
    off = _addr_to_fileoff(addr, segments)
    if off is None or off + 8 > len(data):
        return methods
    entsize, count = struct.unpack('<II', data[off:off + 8])
    entsize &= 0x0000FFFF  # low 16 bits are the real per-entry size
    if entsize == 0:
        entsize = 12
    entry_off = off + 8
    for _ in range(count):
        if entry_off + 12 > len(data):
            break
        name_ptr, types_ptr, imp_addr = struct.unpack('<III', data[entry_off:entry_off + 12])
        sel_name = _read_cstring_at_addr(data, name_ptr, segments)
        if sel_name:
            methods[sel_name] = imp_addr
        entry_off += entsize
    return methods


def _parse_ivar_list(data, addr, segments):
    """Returns {ivar_name: byte_offset_in_instance} for one ivar_list_t at `addr`."""
    ivars = {}
    if not addr:
        return ivars
    off = _addr_to_fileoff(addr, segments)
    if off is None or off + 8 > len(data):
        return ivars
    entsize, count = struct.unpack('<II', data[off:off + 8])
    if entsize < 20:
        entsize = 20
    entry_off = off + 8
    for _ in range(count):
        if entry_off + 20 > len(data):
            break
        offset_ptr, name_ptr, _types, _align, _size = struct.unpack('<IIIII', data[entry_off:entry_off + 20])
        name = _read_cstring_at_addr(data, name_ptr, segments)
        ivar_offset = _read_ptr_at_addr(data, offset_ptr, segments)
        if name and ivar_offset is not None:
            ivars[name] = ivar_offset
        entry_off += entsize
    return ivars


def parse_classes(data, segments, classlist_section):
    """
    Parses __objc_classlist into real class/method metadata. NOT a general
    runtime: no categories, no protocols, no message forwarding, no
    metaclass (class-method) resolution. Just enough to let _objc_msgSend
    dispatch to real app-defined instance methods instead of faking every
    call, and to let UIApplicationMain locate the app delegate.

    Returns (classes_by_addr, classes_by_name):
      classes_by_addr[class_t_addr] = {'name': str, 'methods': {sel: imp}, 'super': class_t_addr_or_0}
      classes_by_name[name] = class_t_addr
    """
    classes_by_addr = {}
    classes_by_name = {}
    if not classlist_section:
        return classes_by_addr, classes_by_name

    base = classlist_section['addr']
    count = classlist_section['size'] // 4

    for i in range(count):
        entry_addr = base + i * 4
        class_addr = _read_ptr_at_addr(data, entry_addr, segments)
        if not class_addr:
            continue

        class_off = _addr_to_fileoff(class_addr, segments)
        if class_off is None or class_off + 20 > len(data):
            continue
        isa, superclass, cache, vtable, ro_addr = struct.unpack('<IIIII', data[class_off:class_off + 20])

        ro_off = _addr_to_fileoff(ro_addr, segments)
        if ro_off is None or ro_off + 40 > len(data):
            continue
        (flags, inst_start, inst_size, ivar_layout, name_ptr,
         base_methods, base_protocols, ivars, weak_ivar_layout, base_props) = \
            struct.unpack('<IIIIIIIIII', data[ro_off:ro_off + 40])

        class_name = _read_cstring_at_addr(data, name_ptr, segments)
        methods = _parse_method_list(data, base_methods, segments)
        ivar_map = _parse_ivar_list(data, ivars, segments)

        classes_by_addr[class_addr] = {'name': class_name, 'methods': methods, 'super': superclass, 'ivars': ivar_map, 'inst_size': inst_size}
        if class_name:
            classes_by_name[class_name] = class_addr

    return classes_by_addr, classes_by_name


def resolve_method(classes_by_addr, class_addr, selector_name, max_hops=8):
    """Walk the superclass chain looking for selector_name. Returns an IMP address or None."""
    hops = 0
    while class_addr and class_addr in classes_by_addr and hops < max_hops:
        cls = classes_by_addr[class_addr]
        if selector_name in cls['methods']:
            return cls['methods'][selector_name]
        class_addr = cls['super']
        hops += 1
    return None


def bind_class_refs(data, segments, classrefs_section, classes_by_addr, synth_base=0x71000000,
                    classref_names=None, super_names=None):
    """
    Real dyld fills every __objc_classrefs slot with a pointer to the class
    object at load time. We never do, so every `[SomeClass foo]` send reads
    a zero receiver and dispatch can't resolve anything.

    For classes defined in this binary the slot already holds the real
    class_t address. For framework classes (UIImageView, UIButton...) we
    mint one synthetic class object PER NAME, so identity is stable and
    `alloc` / class-method sends land on something identifiable.

    classref_names {slot_addr: 'UIImageView'} and super_names
    {local_class_addr: 'UIView'} come from the binary's external
    relocation table (what dyld would have bound); when absent we fall
    back to anonymous per-slot names.

    Returns (patches, synthetic_classes).
    """
    patches = []
    synthetic = {}
    classref_names = classref_names or {}
    super_names = super_names or {}
    name_to_synth = {}
    state = {'next': synth_base}

    def synth_for(name, fallback):
        key = name or fallback
        if key not in name_to_synth:
            addr = state['next']
            state['next'] += 0x10
            name_to_synth[key] = addr
            synthetic[addr] = {'name': key, 'methods': {}, 'super': 0, 'ivars': {}, 'inst_size': 0}
        return name_to_synth[key]

    if classrefs_section:
        base = classrefs_section['addr']
        count = classrefs_section['size'] // 4
        for i in range(count):
            slot_addr = base + i * 4
            existing = _read_ptr_at_addr(data, slot_addr, segments)
            if existing and existing in classes_by_addr:
                patches.append((slot_addr, existing))
                continue
            patches.append((slot_addr, synth_for(classref_names.get(slot_addr), f"ExternalClass_{i}")))

    # Superclass links for local classes whose superclass is a framework class.
    for class_addr, name in super_names.items():
        if class_addr in classes_by_addr:
            classes_by_addr[class_addr]['super'] = synth_for(name, name)

    return patches, synthetic


def find_app_delegate(classes_by_addr):
    """
    Heuristic: the app delegate is whichever local class implements one of
    the standard launch-lifecycle selectors. Real UIApplicationMain finds it
    via the principal-class/delegate mechanism (Info.plist / nib); we don't
    have that pipeline, so we just look for the telltale method instead.
    Returns (class_addr, selector_name) or (None, None).
    """
    candidates = (
        "application:didFinishLaunchingWithOptions:",
        "applicationDidFinishLaunching:",
    )
    for class_addr, cls in classes_by_addr.items():
        for sel in candidates:
            if sel in cls['methods']:
                return class_addr, sel
    return None, None

def resolve_ivar_offset(classes_by_addr, class_addr, ivar_name, max_hops=8):
    """Walk the superclass chain for an ivar's byte offset. Returns int or None."""
    hops = 0
    while class_addr and class_addr in classes_by_addr and hops < max_hops:
        cls = classes_by_addr[class_addr]
        if ivar_name in cls.get('ivars', {}):
            return cls['ivars'][ivar_name]
        class_addr = cls['super']
        hops += 1
    return None