# touch.py
"""
Input layer: turns mouse AND real touchscreen events into UIKit-style touch
phases (began / moved / ended / cancelled) for the emulated app.

Windows touchscreens reach pygame two ways: native FINGERDOWN/FINGERMOTION/
FINGERUP events (normalized 0..1 coordinates) and SDL's synthesized mouse
events for the same touch (flagged event.touch == True). Handling both would
deliver every tap twice, so finger events win, and touch-synthesized mouse
events are only used as a fallback on drivers that never send finger events.

iOS apps get one UITouch per finger. This emulator hands the app a single
touch object, so the FIRST finger drives the app; extra fingers are tracked
(so lifting the wrong one doesn't end the touch) but not delivered.
"""
import logging

import pygame

_FINGERDOWN = getattr(pygame, 'FINGERDOWN', None)
_FINGERUP = getattr(pygame, 'FINGERUP', None)
_FINGERMOTION = getattr(pygame, 'FINGERMOTION', None)
_FOCUS_LOST = getattr(pygame, 'WINDOWFOCUSLOST', None)

MOVE_THRESHOLD = 1.5   # logical points; below this a "move" is just finger jitter


class TouchInput:
    def __init__(self, cpu, logical_size=(320, 480)):
        self.cpu = cpu
        self.lw, self.lh = logical_size
        self._primary = None            # ('finger', id) or ('mouse',)
        self._primary_synthetic = False # primary began as a touch-synthesized mouse event
        self._fingers = {}              # finger_id -> (x, y) in logical points
        self._finger_events_seen = False
        self._last_pos = (0.0, 0.0)
        self.show_indicator = False

    # ---- coordinate helpers ----
    def _from_normalized(self, nx, ny):
        return (max(0.0, min(1.0, nx)) * self.lw, max(0.0, min(1.0, ny)) * self.lh)

    def _from_pixels(self, pos, window_size):
        w, h = window_size
        return (pos[0] * self.lw / max(1, w), pos[1] * self.lh / max(1, h))

    # ---- dispatch into the emulated app ----
    def _send(self, phase, pos):
        self._last_pos = pos
        self.cpu.queue_touch(phase, pos[0], pos[1])

    def _moved(self, pos):
        if abs(pos[0] - self._last_pos[0]) < MOVE_THRESHOLD and abs(pos[1] - self._last_pos[1]) < MOVE_THRESHOLD:
            return
        self._send('moved', pos)

    def cancel_all(self):
        """Window lost focus / app switched: end any in-flight touch the way iOS does (touchesCancelled)."""
        if self._primary is not None:
            self._send('cancelled', self._last_pos)
        self._primary = None
        self._primary_synthetic = False
        self._fingers.clear()

    # ---- pygame event entry point ----
    def handle_event(self, event, window_size):
        """Returns True if the event was input this layer owns (so main.py should skip it)."""
        t = event.type

        if _FINGERDOWN is not None and t == _FINGERDOWN:
            if not self._finger_events_seen:
                self._finger_events_seen = True
                logging.info("[Input] touchscreen detected (finger events); touch-synthesized mouse events ignored")
            pos = self._from_normalized(event.x, event.y)
            self._fingers[event.finger_id] = pos
            if self._primary is None:
                self._primary = ('finger', event.finger_id)
                self._primary_synthetic = False
                self._send('began', pos)
            elif self._primary == ('mouse',) and self._primary_synthetic:
                # The synthesized mouse-down for this same touch got here first: adopt it, don't re-begin.
                self._primary = ('finger', event.finger_id)
                self._primary_synthetic = False
            else:
                logging.debug(f"[Input] extra finger {event.finger_id} tracked, not delivered (single-touch app model)")
            return True

        if _FINGERMOTION is not None and t == _FINGERMOTION:
            pos = self._from_normalized(event.x, event.y)
            self._fingers[event.finger_id] = pos
            if self._primary == ('finger', event.finger_id):
                self._moved(pos)
            return True

        if _FINGERUP is not None and t == _FINGERUP:
            pos = self._from_normalized(event.x, event.y)
            self._fingers.pop(event.finger_id, None)
            if self._primary == ('finger', event.finger_id):
                self._primary = None
                self._send('ended', pos)
            return True

        if t in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            synthetic = bool(getattr(event, 'touch', False))
            if synthetic and self._finger_events_seen:
                return True                         # duplicate of a finger event we already handled
            pos = self._from_pixels(event.pos, window_size)
            if t == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self._primary is None:
                    self._primary = ('mouse',)
                    self._primary_synthetic = synthetic
                    self._send('began', pos)
            elif t == pygame.MOUSEMOTION:
                if self._primary == ('mouse',) and event.buttons[0]:
                    self._moved(pos)
            else:  # MOUSEBUTTONUP
                if event.button == 1 and self._primary == ('mouse',):
                    self._primary = None
                    self._primary_synthetic = False
                    self._send('ended', pos)
            return True

        if _FOCUS_LOST is not None and t == _FOCUS_LOST:
            self.cancel_all()
            return False

        return False

    # ---- optional visual feedback for touchscreens ----
    def draw_overlay(self, surface):
        if not self.show_indicator:
            return
        points = list(self._fingers.values())
        if self._primary == ('mouse',):
            points.append(self._last_pos)
        for x, y in points:
            pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), 22, 2)
            pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), 4)
