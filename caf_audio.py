# caf_audio.py
"""
Decodes Apple's CAF (Core Audio File) container into a standard WAV file
pygame.mixer/SDL_mixer can load directly. SDL_mixer has no CAF support at
all, so without this, every .caf sound an app tries to play just fails to
load.

CAF is a CONTAINER, not a codec - but AudioServicesCreateSystemSoundID
(the one API this whole emulator's audio path exists to serve) is
documented to accept exactly two encodings inside it: linear PCM (any of
8/16/24/32-bit int or 32/64-bit float, big- or little-endian) and IMA4
ADPCM. That's the complete input space for this API, not an arbitrary
subset - so covering both is a complete fix for this use case, not a
partial one.
"""
import struct

CAF_MAGIC = b'caff'


def is_caf(data):
    return data[:4] == CAF_MAGIC


def _read_chunks(data):
    """Yields (chunk_type: bytes, chunk_data: bytes) for each top-level CAF chunk."""
    pos = 8  # 'caff' magic(4) + file version(2) + file flags(2)
    while pos + 12 <= len(data):
        chunk_type = data[pos:pos + 4]
        chunk_size = struct.unpack('>q', data[pos + 4:pos + 12])[0]
        pos += 12
        if chunk_size < 0:  # rare "rest of file" marker for streamed data
            chunk_size = len(data) - pos
        chunk_data = data[pos:pos + chunk_size]
        yield chunk_type, chunk_data
        pos += chunk_size


# Standard IMA ADPCM tables (identical ones QuickTime/CAF IMA4 uses).
_IMA_INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
_IMA_STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]


def _decode_ima4_packet(packet_32_bytes, predictor, step_index):
    """One packet's 32 data bytes = 64 nibbles = 64 samples for one channel."""
    samples = [0] * 64
    i = 0
    for byte in packet_32_bytes:
        for nibble in (byte & 0x0F, (byte >> 4) & 0x0F):
            step = _IMA_STEP_TABLE[step_index]
            diff = step >> 3
            if nibble & 4:
                diff += step
            if nibble & 2:
                diff += step >> 1
            if nibble & 1:
                diff += step >> 2
            if nibble & 8:
                predictor -= diff
            else:
                predictor += diff
            predictor = max(-32768, min(32767, predictor))
            step_index = max(0, min(88, step_index + _IMA_INDEX_TABLE[nibble]))
            samples[i] = predictor
            i += 1
    return samples


def _decode_ima4(data, channels):
    """
    IMA4 packs 64 samples per channel into a 34-byte packet: a 2-byte
    preamble (top 9 bits = initial predictor, low 7 bits = step table
    index) followed by 32 bytes / 64 nibbles of deltas. Multi-channel CAF
    interleaves one whole packet per channel before repeating.
    """
    packet_size = 34
    channel_samples = [[] for _ in range(channels)]
    pos = 0
    while pos + packet_size * channels <= len(data):
        for ch in range(channels):
            preamble = struct.unpack('>H', data[pos:pos + 2])[0]
            predictor = preamble & 0xFF80
            if predictor & 0x8000:
                predictor -= 0x10000
            step_index = preamble & 0x7F
            step_index = max(0, min(88, step_index))
            packet_data = data[pos + 2:pos + packet_size]
            channel_samples[ch].extend(_decode_ima4_packet(packet_data, predictor, step_index))
            pos += packet_size
    if channels == 1:
        return channel_samples[0]
    interleaved = []
    for i in range(len(channel_samples[0])):
        for ch in range(channels):
            interleaved.append(channel_samples[ch][i])
    return interleaved


def decode_caf_to_wav_bytes(data):
    """Parses a CAF file and returns an equivalent standard PCM16 little-endian WAV, as bytes."""
    if not is_caf(data):
        raise ValueError("Not a CAF file")

    desc = None
    audio_data = None
    for chunk_type, chunk_data in _read_chunks(data):
        if chunk_type == b'desc' and len(chunk_data) >= 32:
            (sample_rate, format_id, format_flags, bytes_per_packet,
             frames_per_packet, channels, bits_per_channel) = struct.unpack('>d4sIIIII', chunk_data[:32])
            desc = {
                'sample_rate': sample_rate, 'format_id': format_id, 'format_flags': format_flags,
                'channels': channels, 'bits_per_channel': bits_per_channel,
            }
        elif chunk_type == b'data':
            audio_data = chunk_data[4:]  # first 4 bytes are an "edit count", not audio

    if desc is None or audio_data is None:
        raise ValueError("CAF file missing 'desc' or 'data' chunk")

    channels = desc['channels'] or 1
    sample_rate = int(desc['sample_rate'])
    format_id = desc['format_id']

    if format_id == b'ima4':
        samples = _decode_ima4(audio_data, channels)
        pcm16 = struct.pack(f'<{len(samples)}h', *samples)

    elif format_id == b'lpcm':
        is_float = bool(desc['format_flags'] & 0x1)
        is_little_endian = bool(desc['format_flags'] & 0x2)  # kCAFLinearPCMFormatFlagIsLittleEndian - SET means little-endian; clear means CAF's native big-endian
        is_big_endian = not is_little_endian
        bits = desc['bits_per_channel']
        endian = '>' if is_big_endian else '<'

        if is_float:
            fmt_char = 'f' if bits == 32 else 'd'
            sz = struct.calcsize(fmt_char)
            count = len(audio_data) // sz
            floats = struct.unpack(f'{endian}{count}{fmt_char}', audio_data[:count * sz])
            samples = [max(-32768, min(32767, int(f * 32767.0))) for f in floats]
            pcm16 = struct.pack(f'<{len(samples)}h', *samples)
        elif bits == 16:
            count = len(audio_data) // 2
            samples = struct.unpack(f'{endian}{count}h', audio_data[:count * 2])
            pcm16 = struct.pack(f'<{count}h', *samples)
        elif bits == 8:
            # CAF 8-bit linear PCM is unsigned; WAV's 8-bit convention is
            # also unsigned, but we standardize everything to 16-bit here
            # for one simple downstream path.
            samples = [(b - 128) * 256 for b in audio_data]
            pcm16 = struct.pack(f'<{len(samples)}h', *samples)
        elif bits == 24:
            count = len(audio_data) // 3
            samples = []
            for i in range(count):
                b0, b1, b2 = audio_data[i * 3:i * 3 + 3]
                if is_big_endian:
                    val = (b0 << 16) | (b1 << 8) | b2
                else:
                    val = (b2 << 16) | (b1 << 8) | b0
                if val & 0x800000:
                    val -= 0x1000000
                samples.append(max(-32768, min(32767, val >> 8)))
            pcm16 = struct.pack(f'<{len(samples)}h', *samples)
        elif bits == 32:
            count = len(audio_data) // 4
            samples = struct.unpack(f'{endian}{count}i', audio_data[:count * 4])
            samples = [max(-32768, min(32767, s >> 16)) for s in samples]
            pcm16 = struct.pack(f'<{len(samples)}h', *samples)
        else:
            raise ValueError(f"Unsupported lpcm bit depth: {bits}")
    else:
        raise ValueError(f"Unsupported CAF audio format: {format_id!r} "
                          f"(AudioServicesCreateSystemSoundID only ever produces lpcm or ima4)")

    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm16)
    header = (
        b'RIFF' + struct.pack('<I', 36 + data_size) + b'WAVE'
        + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
        + b'data' + struct.pack('<I', data_size)
    )
    return header + pcm16
