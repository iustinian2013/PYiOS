# audio.py
import io
import logging
import pygame
from caf_audio import is_caf, decode_caf_to_wav_bytes


class IOSAudioEngine:
    """
    Real (if partial) implementation of the one AudioServices/AudioToolbox
    pattern old apps like this actually use: load a bundle sound file once
    via AudioServicesCreateSystemSoundID, then fire-and-forget play it via
    AudioServicesPlaySystemSound. Previously this class was a single print
    statement, and cpu.py never even routed those two calls anywhere - no
    app sound has ever played. cpu.py resolves the app's CFURLRef/path and
    hands this class a real, already-resolved file path; this class only
    owns loading and playback.

    .caf (Core Audio Format) files - what AudioServices actually requires -
    are decoded by caf_audio.py (linear PCM and IMA4) into WAV in memory,
    since SDL_mixer can't read CAF at all. Everything else (.wav/.mp3/.ogg)
    goes straight to pygame.mixer.
    """

    def __init__(self, app_folder=""):
        self.app_folder = app_folder
        self._sounds = {}          # SystemSoundID (int) -> pygame.mixer.Sound
        self._next_sound_id = 1
        self._mixer_ok = False
        try:
            pygame.mixer.init()
            self._mixer_ok = True
            logging.info("[Audio Engine] Core audio subsystem initialized.")
        except Exception as e:
            logging.warning(f"[Audio Engine] pygame.mixer.init() failed - audio disabled: {e}")

    def create_system_sound_id(self, path):
        """
        Loads a sound file and returns a new SystemSoundID for it, or 0 on
        failure - matching AudioServicesCreateSystemSoundID's out-param
        closely enough for cpu.py's stub to hand straight back to the app.
        """
        if not self._mixer_ok or not path:
            return 0
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            if is_caf(raw):
                sound = pygame.mixer.Sound(io.BytesIO(decode_caf_to_wav_bytes(raw)))
            else:
                sound = pygame.mixer.Sound(io.BytesIO(raw))
        except Exception as e:
            logging.warning(f"[Audio Engine] Could not load '{path}': {e}")
            return 0
        sound_id = self._next_sound_id
        self._next_sound_id += 1
        self._sounds[sound_id] = sound
        logging.info(f"[Audio Engine] Loaded system sound {sound_id} <- {path}")
        return sound_id

    def play_system_sound(self, sound_id):
        sound = self._sounds.get(sound_id)
        if sound is None:
            logging.debug(f"[Audio Engine] play_system_sound: unknown id {sound_id}")
            return
        try:
            sound.play()
        except Exception as e:
            logging.warning(f"[Audio Engine] Playback failed for sound {sound_id}: {e}")

    def dispose_system_sound(self, sound_id):
        self._sounds.pop(sound_id, None)