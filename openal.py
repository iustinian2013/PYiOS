# openal.py
"""
OpenAL + AudioFile emulation (the C audio stack iBeer's looping sounds use).

The app's sound engine is Apple's classic "oalTouch" pattern:

    AudioFileOpenURL(url, ..., &file)
    AudioFileGetProperty(file, 'dfmt', &size, &asbd)     -> channels / bit depth / sample rate
    AudioFileGetProperty(file, 'bcnt', &size, &bytes)    -> number of PCM bytes
    AudioFileReadBytes(file, 0, 0, &n, buffer)           -> raw PCM into guest memory
    alGenBuffers / alBufferData(buffer, format, data, size, freq)
    alGenSources / alSourcei(src, AL_BUFFER, buf) / alSourcei(src, AL_LOOPING, 1) / alSourcePlay(src)

Before this module every one of those was an "unimplemented stub returning 0", so alcOpenDevice() failed,
the app printed "... returned error" and never made a sound with its loop engine (the pour / fluid loop).

Files are decoded with caf_audio (CAF lpcm / ima4 -> PCM16) or the stdlib `wave` module, handed to the guest as
16-bit little-endian PCM, and OpenAL buffers/sources are mapped onto pygame.mixer.Sound / Channel
(looping, gain, play/stop/pause/rewind, source state queries).

Everything is keyed by plain integer handles; handle(uc, name) returns True when it emulated `name`.
"""
import io
import logging
import os
import struct
import wave

import pygame
from unicorn import arm_const as A

from caf_audio import is_caf, decode_caf_to_wav_bytes

# AL enums
AL_PITCH, AL_GAIN, AL_LOOPING, AL_BUFFER, AL_SOURCE_STATE = 0x1003, 0x100A, 0x1007, 0x1009, 0x1010
AL_INITIAL, AL_PLAYING, AL_PAUSED, AL_STOPPED = 0x1011, 0x1012, 0x1013, 0x1014
AL_FORMATS = {0x1100: (1, 8), 0x1101: (1, 16), 0x1102: (2, 8), 0x1103: (2, 16)}

# AudioFile property IDs (four-char codes)
K_DATA_FORMAT = 0x64666D74        # 'dfmt'  AudioStreamBasicDescription
K_BYTE_COUNT = 0x62636E74         # 'bcnt'  UInt64
K_PACKET_COUNT = 0x70636E74       # 'pcnt'  UInt64
K_FORMAT_LPCM = 0x6C70636D        # 'lpcm'
K_ERR_UNSUPPORTED = 0x7479703F    # 'typ?'
K_ERR_PROPERTY = 0x70726F70       # 'prop'


class OpenALEmulator:
    def __init__(self, cpu):
        self.cpu = cpu
        self.files = {}        # AudioFileID -> {'pcm': bytes, 'channels': int, 'rate': int, 'path': str}
        self.buffers = {}      # ALuint -> {'pcm': bytes, 'channels': int, 'bits': int, 'rate': int, 'sound': Sound|None}
        self.sources = {}      # ALuint -> {'buffer': int, 'loop': bool, 'gain': float, 'channel': Channel|None, 'paused': bool}
        self._next_file = 0x0AF10001
        self._next_buf = 1
        self._next_src = 1
        self.device = 0
        self.context = 0
        self.current = 0

    # ---- helpers -------------------------------------------------------------
    def _decode(self, path):
        """Any supported file -> (pcm16le bytes, channels, rate), or None."""
        with open(path, 'rb') as f:
            raw = f.read()
        wav_bytes = decode_caf_to_wav_bytes(raw) if is_caf(raw) else raw
        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            pcm = w.readframes(w.getnframes())
        if width == 1:                                   # unsigned 8-bit -> signed 16-bit
            pcm = b''.join(struct.pack('<h', (b - 128) * 256) for b in pcm)
        elif width != 2:
            return None
        return pcm, ch, rate

    @staticmethod
    def _wav(pcm, channels, bits, rate):
        out = io.BytesIO()
        with wave.open(out, 'wb') as w:
            w.setnchannels(channels)
            w.setsampwidth(bits // 8)
            w.setframerate(rate)
            w.writeframes(pcm)
        return out.getvalue()

    def _sound_for(self, buf):
        if buf['sound'] is None and pygame.mixer.get_init():
            try:
                buf['sound'] = pygame.mixer.Sound(io.BytesIO(self._wav(buf['pcm'], buf['channels'], buf['bits'], buf['rate'])))
            except Exception as e:
                logging.warning(f"[OpenAL] could not build a sound from buffer: {e}")
        return buf['sound']

    def _state(self, src):
        if src['channel'] is not None and src['channel'].get_busy():
            return AL_PAUSED if src['paused'] else AL_PLAYING
        if src['paused']:
            return AL_PAUSED
        return AL_STOPPED if src.get('played') else AL_INITIAL

    # ---- the stubs -----------------------------------------------------------
    def handle(self, uc, name):
        base = name.lstrip('_')
        cpu = self.cpu
        rd = uc.reg_read
        R0, R1, R2, R3 = rd(A.UC_ARM_REG_R0), rd(A.UC_ARM_REG_R1), rd(A.UC_ARM_REG_R2), rd(A.UC_ARM_REG_R3)
        sp = rd(A.UC_ARM_REG_SP)

        def ret(v):
            uc.reg_write(A.UC_ARM_REG_R0, v & 0xFFFFFFFF)

        def w32(addr, val):
            if addr:
                uc.mem_write(addr, struct.pack('<I', val & 0xFFFFFFFF))

        f32 = lambda w: struct.unpack('<f', struct.pack('<I', w & 0xFFFFFFFF))[0]

        # --- C++ exception frames (the app's .mm code registers setjmp/longjmp unwind contexts) ---
        if base in ('Unwind_SjLj_Register', 'Unwind_SjLj_Unregister'):
            return True

        # ================= AudioFile =================
        if base == 'AudioFileOpenURL':                     # (url, perm, typeHint, &file)
            path = cpu._cfurl_paths.get(R0) or cpu._resolve_resource_path(cpu.urls.get(R0))
            if not path:
                logging.warning(f"[AudioFile] AudioFileOpenURL: unknown CFURLRef 0x{R0:08X}")
                ret(K_ERR_UNSUPPORTED)
                return True
            try:
                decoded = self._decode(path)
            except Exception as e:
                logging.warning(f"[AudioFile] cannot decode {path}: {e}")
                decoded = None
            if decoded is None:
                ret(K_ERR_UNSUPPORTED)
                return True
            handle = self._next_file
            self._next_file += 1
            self.files[handle] = {'pcm': decoded[0], 'channels': decoded[1], 'rate': decoded[2], 'path': path}
            w32(R3, handle)
            logging.info(f"[AudioFile] opened {os.path.basename(path)}: {decoded[1]}ch {decoded[2]}Hz, {len(decoded[0])} PCM bytes")
            ret(0)
            return True
        if base == 'AudioFileClose':
            self.files.pop(R0, None)
            ret(0)
            return True
        if base == 'AudioFileGetProperty':                 # (file, propID, &ioSize, outData)
            fl = self.files.get(R0)
            if fl is None:
                ret(K_ERR_PROPERTY)
                return True
            if R1 == K_DATA_FORMAT:
                ch = fl['channels']
                uc.mem_write(R3, struct.pack('<dIIIIIIII', float(fl['rate']), K_FORMAT_LPCM, 0x0C, 2 * ch, 1, 2 * ch, ch, 16, 0))
                w32(R2, 40)
            elif R1 == K_BYTE_COUNT:
                uc.mem_write(R3, struct.pack('<Q', len(fl['pcm'])))
                w32(R2, 8)
            elif R1 == K_PACKET_COUNT:
                uc.mem_write(R3, struct.pack('<Q', len(fl['pcm']) // (2 * fl['channels'])))
                w32(R2, 8)
            else:
                ret(K_ERR_PROPERTY)
                return True
            ret(0)
            return True
        if base == 'AudioFileReadBytes':                   # (file, useCache, SInt64 start [r2:r3], &ioNumBytes, outBuffer)
            fl = self.files.get(R0)
            num_ptr, out_ptr = cpu._stack_words(uc, sp, 2)
            if fl is None:
                ret(K_ERR_PROPERTY)
                return True
            start = R2 | (R3 << 32)
            want = struct.unpack('<I', bytes(uc.mem_read(num_ptr, 4)))[0]
            chunk = fl['pcm'][start:start + want]
            uc.mem_write(out_ptr, chunk)
            w32(num_ptr, len(chunk))
            ret(0)
            return True

        # ================= ALC (device / context) =================
        if base == 'alcOpenDevice':
            self.device = 0x0A1D0001
            ret(self.device)
            return True
        if base == 'alcCreateContext':
            self.context = 0x0A1C0001
            ret(self.context)
            return True
        if base == 'alcMakeContextCurrent':
            self.current = R0
            ret(1)
            return True
        if base == 'alcGetCurrentContext':
            ret(self.current)
            return True
        if base in ('alcDestroyContext', 'alcCloseDevice', 'alcProcessContext', 'alcSuspendContext'):
            ret(1 if base == 'alcCloseDevice' else 0)
            return True
        if base in ('alcGetError', 'alGetError'):
            ret(0)
            return True

        # ================= AL: listener / misc (nothing to spatialise) =================
        if base in ('alListener3f', 'alListenerf', 'alListenerfv', 'alListeneri', 'alDistanceModel', 'alDopplerFactor',
                    'alSpeedOfSound', 'alSource3f', 'alSourcefv', 'alSourceQueueBuffers', 'alSourceUnqueueBuffers'):
            return True

        # ================= AL: buffers =================
        if base == 'alGenBuffers':                          # (n, ids*)
            for i in range(R0):
                bid = self._next_buf
                self._next_buf += 1
                self.buffers[bid] = {'pcm': b'', 'channels': 1, 'bits': 16, 'rate': 22050, 'sound': None}
                w32(R1 + 4 * i, bid)
            return True
        if base == 'alDeleteBuffers':
            for i in range(R0):
                self.buffers.pop(struct.unpack('<I', bytes(uc.mem_read(R1 + 4 * i, 4)))[0], None)
            return True
        if base in ('alBufferData', 'alBufferDataStatic'):  # (buffer, format, data*, size, freq[stack])
            (freq,) = cpu._stack_words(uc, sp, 1)
            buf = self.buffers.get(R0)
            if buf is not None and R1 in AL_FORMATS and R2 and R3:
                ch, bits = AL_FORMATS[R1]
                pcm = bytes(uc.mem_read(R2, R3))
                if bits == 8:
                    pcm = b''.join(struct.pack('<h', (b - 128) * 256) for b in pcm)
                    bits = 16
                buf.update(pcm=pcm, channels=ch, bits=bits, rate=freq or 22050, sound=None)
            return True
        if base == 'alIsBuffer':
            ret(1 if R0 in self.buffers else 0)
            return True

        # ================= AL: sources =================
        if base == 'alGenSources':                          # (n, ids*)
            for i in range(R0):
                sid = self._next_src
                self._next_src += 1
                self.sources[sid] = {'buffer': 0, 'loop': False, 'gain': 1.0, 'channel': None, 'paused': False}
                w32(R1 + 4 * i, sid)
            return True
        if base == 'alDeleteSources':
            for i in range(R0):
                sid = struct.unpack('<I', bytes(uc.mem_read(R1 + 4 * i, 4)))[0]
                src = self.sources.pop(sid, None)
                if src and src['channel'] is not None:
                    src['channel'].stop()
            return True
        if base == 'alIsSource':
            ret(1 if R0 in self.sources else 0)
            return True
        if base in ('alSourcei', 'alSourcef'):              # (src, param, value)
            src = self.sources.get(R0)
            if src is not None:
                if R1 == AL_BUFFER:
                    src['buffer'] = R2
                elif R1 == AL_LOOPING:
                    src['loop'] = bool(R2)
                elif R1 == AL_GAIN:
                    src['gain'] = max(0.0, min(1.0, f32(R2) if base == 'alSourcef' else float(R2)))
                    if src['channel'] is not None:
                        src['channel'].set_volume(src['gain'])
            return True
        if base == 'alGetSourcei':                          # (src, param, &value)
            src = self.sources.get(R0)
            val = 0
            if src is not None:
                val = self._state(src) if R1 == AL_SOURCE_STATE else (src['buffer'] if R1 == AL_BUFFER else int(src['loop']))
            w32(R2, val)
            return True
        if base == 'alGetBufferi':
            w32(R2, 0)
            return True
        if base in ('alSourcePlay', 'alSourceStop', 'alSourcePause', 'alSourceRewind'):
            src = self.sources.get(R0)
            if src is None:
                return True
            if base == 'alSourcePlay':
                if src['paused'] and src['channel'] is not None:
                    src['channel'].unpause()
                    src['paused'] = False
                else:
                    buf = self.buffers.get(src['buffer'])
                    snd = self._sound_for(buf) if buf else None
                    if snd is not None:
                        if src['channel'] is not None:
                            src['channel'].stop()
                        src['channel'] = snd.play(loops=-1 if src['loop'] else 0)
                        if src['channel'] is not None:
                            src['channel'].set_volume(src['gain'])
                        src['played'] = True
                        src['paused'] = False
                        logging.info(f"[OpenAL] source {R0} play ({'loop' if src['loop'] else 'once'})")
            elif base == 'alSourcePause':
                if src['channel'] is not None and src['channel'].get_busy():
                    src['channel'].pause()
                    src['paused'] = True
            else:
                if src['channel'] is not None:
                    src['channel'].stop()
                src['paused'] = False
                src['played'] = base == 'alSourceStop'
            return True
        return False
