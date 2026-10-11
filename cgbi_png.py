# cgbi_png.py
"""
Decodes Apple's CgBI-mangled PNGs - the device-optimized PNG format Xcode's
PNG-crush step produces for app bundle resources - into plain RGBA bytes.

Standard libpng (what pygame.image.load uses) refuses these outright
("CgBI: unhandled critical chunk"): they're deliberately non-conformant.
Three differences from a normal PNG:
  - a private "CgBI" chunk is inserted right after IHDR
  - pixel data is BGRA (not RGBA) and premultiplied by alpha
  - the IDAT stream is deflated RAW - no 2-byte zlib header, no adler32
    trailer - instead of a standard zlib stream
Chunk structure and per-scanline filtering (None/Sub/Up/Average/Paeth) are
otherwise ordinary PNG. See http://iphonedevwiki.net/index.php/CgBI_file_format.

This only handles 8-bit, non-interlaced RGB/RGBA (color types 2 and 6) -
that covers the overwhelming majority of sprite/UI art in these older app
bundles, which is all this emulator needs it for.
"""
import io
import struct
import zlib

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def is_cgbi_png(data):
    """True if `data` is a PNG with a CgBI chunk (must appear before IDAT)."""
    if data[:8] != PNG_SIGNATURE:
        return False
    pos = 8
    while pos + 8 <= len(data):
        length, = struct.unpack('>I', data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        if ctype == b'CgBI':
            return True
        if ctype == b'IDAT':
            return False  # CgBI always precedes IDAT in a real CgBI file
        pos += 8 + length + 4  # length + type + data + crc
    return False


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw, width, height, bpp):
    """Reverses PNG per-scanline filtering. `raw` is the inflated IDAT stream."""
    stride = width * bpp
    out = bytearray(stride * height)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride

        if ftype == 0:  # None
            pass
        elif ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xFF
        else:
            raise ValueError(f"Unknown PNG filter type {ftype} on row {y}")

        out[y * stride:(y + 1) * stride] = line
        prev = line
    return bytes(out)


def _decode_cgbi_png_slow(data):
    """
    Returns (width, height, rgba_or_rgb_bytes, channels) with pixels
    converted from Apple's premultiplied BGRA/BGR back to straight
    RGBA/RGB, ready for pygame.image.frombuffer().
    """
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("Not a PNG file")

    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat_chunks = []

    while pos + 8 <= len(data):
        length, = struct.unpack('>I', data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]

        if ctype == b'IHDR':
            width, height, bit_depth, color_type, _comp, _filt, interlace = \
                struct.unpack('>IIBBBBB', chunk_data)
        elif ctype == b'IDAT':
            idat_chunks.append(chunk_data)
        elif ctype == b'IEND':
            break

        pos += 8 + length + 4

    if width is None:
        raise ValueError("No IHDR chunk found")
    if interlace:
        raise ValueError("Interlaced CgBI PNGs not supported")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"Unsupported CgBI PNG: bit_depth={bit_depth}, color_type={color_type}")

    channels = 4 if color_type == 6 else 3
    compressed = b''.join(idat_chunks)

    # RAW deflate - wbits=-15 tells zlib to expect no header/trailer at all,
    # which is exactly what Apple's crush step leaves out.
    decompressor = zlib.decompressobj(-15)
    raw = decompressor.decompress(compressed) + decompressor.flush()

    pixels = bytearray(_unfilter(raw, width, height, channels))

    if channels == 4:
        # BGRA premultiplied -> RGBA straight alpha
        for i in range(0, len(pixels), 4):
            b, g, r, a = pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]
            if a and a != 255:
                r = min(255, (r * 255) // a)
                g = min(255, (g * 255) // a)
                b = min(255, (b * 255) // a)
            pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3] = r, g, b, a
    else:
        # BGR -> RGB (no alpha channel to premultiply against)
        for i in range(0, len(pixels), 3):
            pixels[i], pixels[i + 2] = pixels[i + 2], pixels[i]

    return width, height, bytes(pixels), channels


def _chunk(ctype, payload):
    return (struct.pack('>I', len(payload)) + ctype + payload
            + struct.pack('>I', zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def _decode_cgbi_png_fast(data):
    """
    Rebuilds a standard PNG from the CgBI one (drop the CgBI chunk, re-wrap
    the raw-deflate stream in a real zlib stream) so libpng - via pygame -
    does the per-scanline unfiltering in C, then swaps B/R and
    un-premultiplies with numpy. The pure-Python path below spends ~0.5s per
    320x480 image; this is a few ms.
    """
    import pygame
    if not _HAS_NUMPY:
        raise RuntimeError("numpy unavailable")

    pos = 8
    ihdr = None
    idat = []
    while pos + 8 <= len(data):
        length, = struct.unpack('>I', data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if ctype == b'IHDR':
            ihdr = payload
        elif ctype == b'IDAT':
            idat.append(payload)
        elif ctype == b'IEND':
            break
        pos += 8 + length + 4
    if ihdr is None:
        raise ValueError("No IHDR chunk found")
    width, height, bit_depth, color_type, _c, _f, interlace = struct.unpack('>IIBBBBB', ihdr)
    if interlace:
        raise ValueError("Interlaced CgBI PNGs not supported")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"Unsupported CgBI PNG: bit_depth={bit_depth}, color_type={color_type}")

    d = zlib.decompressobj(-15)
    raw = d.decompress(b''.join(idat)) + d.flush()
    std = (PNG_SIGNATURE + _chunk(b'IHDR', ihdr)
           + _chunk(b'IDAT', zlib.compress(raw, 1)) + _chunk(b'IEND', b''))

    surf = pygame.image.load(io.BytesIO(std), 'cgbi.png')
    channels = 4 if color_type == 6 else 3
    buf = pygame.image.tostring(surf, 'RGBA' if channels == 4 else 'RGB')
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, channels).copy()

    # libpng handed back the file's B,G,R[,A] bytes labelled as R,G,B[,A].
    arr[:, [0, 2]] = arr[:, [2, 0]]
    if channels == 4:
        a = arr[:, 3].astype(np.uint32)
        m = (a > 0) & (a < 255)
        if m.any():
            for c in range(3):
                col = arr[m, c].astype(np.uint32)
                arr[m, c] = np.minimum(255, (col * 255) // a[m]).astype(np.uint8)
    return width, height, arr.tobytes(), channels


def decode_cgbi_png(data):
    """Returns (width, height, pixels, channels): straight-alpha RGBA/RGB."""
    try:
        return _decode_cgbi_png_fast(data)
    except Exception:
        return _decode_cgbi_png_slow(data)