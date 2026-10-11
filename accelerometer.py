# accelerometer.py
"""
UIAccelerometer / UIAcceleration emulation.

Owns everything accelerometer-related that used to be scattered across cpu.py
and main.py: the shared UIAccelerometer instance, the delegate, the update
interval, the UIAcceleration sample object, the simulated sensor reading, and
the delivery schedule for -accelerometer:didAccelerate:.

Simulated input (poll_input, called once per frame by main.py) - the phone is "held" and eases
toward wherever the keys point it, like a real hand does, instead of snapping between fixed poses:
    nothing held     -> upright (0, -1, 0); any tilt eases back to it when you let go
    LEFT / RIGHT     -> roll the phone in the screen plane, up to 100 degrees (hold longer = further)
    UP / DOWN        -> pitch it back toward face-up / forward toward face-down
    T                -> tip back to drink (about 81 degrees of pitch)
    S or SPACE       -> shake (a burst of +-2.5 g, the way shake-to-refill / shake-to-clear apps expect)
    R                -> snap back to upright

Delivery is coalesced: at most ONE -accelerometer:didAccelerate: is ever
waiting in the action queue, and the schedule restarts from "now" after each
delivery. Previously every 20ms of wall-clock that passed while the emulator
was busy (e.g. decoding artwork at launch) queued another call, so the app got
a burst of 10+ back-to-back callbacks and the queue could grow without bound.

UIAccelerationValue is a DOUBLE, so -[UIAcceleration x|y|z] returns in r0:r1
(softfp ABI). Writing only a float into r0 left r1 holding stale garbage, which
the app then used as the high word of the double - so its low-pass filter and
atan2 saw nonsense and the computed tilt never moved. The app's own imports
(__adddf3vfp/__muldf3vfp/__divdf3vfp/_atan2 - all double arithmetic) are
consistent with that. Set EMU_ACCEL_FLOAT=1 to force the old float-in-r0 return.
"""
import logging
import math
import os
import random
import struct
import time

import pygame

ROLL_MAX = math.radians(100.0)
PITCH_MAX = math.radians(85.0)
TIP_PITCH = math.radians(81.0)
RATE_OUT = math.radians(110.0)   # rad/s while a key is held
RATE_BACK = math.radians(240.0)  # rad/s easing back to upright
SHAKE_G = 2.5
SHAKE_HZ = 8.0

SEL_DID_ACCELERATE = 'accelerometer:didAccelerate:'

UPRIGHT = (0.0, -1.0, 0.0)
TIPPED = (0.0, -0.15, -0.95)

NOISE = 0.004            # g, per-axis sensor noise
WOBBLE_AMPLITUDE = 0.12  # g, 'picked the phone up' settle at start
WOBBLE_SECONDS = 0.6
WOBBLE_HZ = 2.5
MIN_INTERVAL = 0.02
DEFAULT_INTERVAL = 0.1
MAX_INTERVAL = 1.0


class Accelerometer:
    def __init__(self, as_double=None):
        self.obj = 0            # guest pointer: [UIAccelerometer sharedAccelerometer]
        self.sample_obj = 0     # guest pointer: the UIAcceleration handed to the delegate
        self.delegate = 0       # guest pointer: object receiving -accelerometer:didAccelerate:
        self.interval = DEFAULT_INTERVAL
        self.xyz = UPRIGHT
        self._next = 0.0
        self._rng = random.Random(0xBEE5)   # deterministic sensor noise
        self._started = None                # monotonic time the app first attached a delegate
        self.roll = 0.0                     # radians, + = clockwise
        self.pitch = 0.0                    # radians, + = top of the phone tipping back
        self._last_poll = None
        self.shaking = False
        self.delivered = 0                  # -accelerometer:didAccelerate: calls queued (for the [Perf] line)
        if as_double is None:
            as_double = os.environ.get("EMU_ACCEL_FLOAT") != "1"
        self.as_double = as_double

    # ---- simulated sensor ----
    @staticmethod
    def _approach(cur, target, rate_out, rate_back, dt):
        """Move cur toward target: rate_out when heading away from upright, rate_back when returning."""
        rate = rate_back if abs(target) < abs(cur) or (target * cur < 0) else rate_out
        step = rate * dt
        if abs(target - cur) <= step:
            return target
        return cur + step if target > cur else cur - step

    def poll_input(self, keys=None):
        """Update self.xyz from the keyboard (+ sensor noise). `keys` defaults to pygame.key.get_pressed()."""
        if keys is None:
            keys = pygame.key.get_pressed()
        now = time.monotonic()
        dt = 0.016 if self._last_poll is None else min(0.1, max(0.0, now - self._last_poll))
        self._last_poll = now

        if keys[pygame.K_r]:
            self.roll = self.pitch = 0.0
        roll_t = ROLL_MAX * ((1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0))
        pitch_t = PITCH_MAX * ((1.0 if keys[pygame.K_UP] else 0.0) - (1.0 if keys[pygame.K_DOWN] else 0.0))
        if keys[pygame.K_t]:
            pitch_t = TIP_PITCH
        self.roll = self._approach(self.roll, roll_t, RATE_OUT, RATE_BACK, dt)
        self.pitch = self._approach(self.pitch, pitch_t, RATE_OUT * 1.4, RATE_BACK, dt)

        # gravity for a phone rolled by `roll` in the screen plane and pitched back by `pitch`
        cp = math.cos(self.pitch)
        base = (math.sin(self.roll) * cp, -math.cos(self.roll) * cp, -math.sin(self.pitch))
        self.shaking = bool(keys[pygame.K_s] or keys[pygame.K_SPACE])
        if self.shaking:
            swing = SHAKE_G if math.sin(2.0 * math.pi * SHAKE_HZ * now) >= 0 else -SHAKE_G
            base = (base[0] + swing, base[1] + swing * 0.4, base[2])
        self.xyz = self._with_sensor_character(base)

    def _with_sensor_character(self, base):
        """
        A real accelerometer never reports the same numbers twice: there is always a few
        thousandths of a g of noise, and picking the phone up gives a settling wobble. Apps
        rely on that implicitly (change detection, filters, "first real sample" logic). A
        perfectly constant (0,-1,0) made this app sit idle until a key was pressed - the
        beer only 'popped up' after the first arrow-key tilt. Simulate both.
        """
        x, y, z = base
        x += self._rng.uniform(-NOISE, NOISE)
        y += self._rng.uniform(-NOISE, NOISE)
        z += self._rng.uniform(-NOISE, NOISE)
        if self._started is not None:
            t = time.monotonic() - self._started
            if t < WOBBLE_SECONDS * 4:
                amp = WOBBLE_AMPLITUDE * math.exp(-t / WOBBLE_SECONDS)
                osc = math.sin(2.0 * math.pi * WOBBLE_HZ * t)
                x += amp * osc
                z += amp * 0.6 * math.cos(2.0 * math.pi * WOBBLE_HZ * t)
        return (x, y, z)

    # ---- guest-facing API (called from cpu.py's msgSend shim) ----
    def shared_instance(self, alloc):
        """-[UIAccelerometer sharedAccelerometer]. `alloc` is cpu.objc_alloc."""
        if not self.obj:
            self.obj = alloc(0)
        return self.obj

    def handle_message(self, receiver, sel_name, r2, r3):
        """
        Handles selectors sent to the UIAccelerometer instance.
        Returns True if handled (caller returns `receiver` in r0).
        """
        if not receiver or receiver != self.obj:
            return False
        if sel_name == 'setDelegate:':
            self.delegate = r2
            if self._started is None:
                self._started = time.monotonic()
            logging.info(f"[Input] accelerometer delegate set (0x{r2:08X}) - LEFT/RIGHT roll, UP/DOWN pitch, "
                         f"hold T to tip back and drink, S/SPACE shake, R recentre")
            return True
        if sel_name == 'setUpdateInterval:':
            # NSTimeInterval is a double: low word in r2, high word in r3.
            try:
                requested = struct.unpack('<d', struct.pack('<II', r2, r3))[0]
            except struct.error:
                requested = DEFAULT_INTERVAL
            if not (requested == requested):     # NaN
                requested = DEFAULT_INTERVAL
            self.interval = min(MAX_INTERVAL, max(MIN_INTERVAL, requested))
            logging.info(f"[Input] accelerometer update interval requested {requested:.4f}s -> using {self.interval:.4f}s "
                         f"({1.0 / self.interval:.0f} Hz)")
            return True
        return False

    def is_sample(self, receiver):
        return bool(receiver) and receiver == self.sample_obj

    def component_regs(self, sel_name):
        """
        Return value of -[UIAcceleration x|y|z] as (r0, r1).

        UIAccelerationValue is a DOUBLE on the real SDKs (r0 = low word, r1 = high
        word), but older code may treat it as a float (r0 only). We don't know which
        this binary does, so return a pair that is correct for both:
          r0 = the float32 bits of the value          -> exact when read as a float
          r1 = the high word of the double's bits     -> with r0 as the low word, the
               double is the true value plus a error below ~1e-6 (r0 only perturbs the
               low mantissa bits), which is irrelevant for a tilt reading.
        Previously only r0 was written, so a double-reading app saw a denormal ~0 for
        every axis and the tilt could never change.
        """
        value = float(self.xyz['xyz'.index(sel_name)])
        float_bits = struct.unpack('<I', struct.pack('<f', value))[0]
        double_hi = struct.unpack('<II', struct.pack('<d', value))[1]
        if self.as_double:      # EMU_ACCEL_DOUBLE=1: exact double (low word 0 for round values)
            return struct.unpack('<II', struct.pack('<d', value))
        return float_bits, double_hi

    # ---- delivery schedule ----
    def poll(self, now, action_queue, alloc):
        """
        Called every frame by cpu.tick(). Appends at most one
        -accelerometer:didAccelerate: call to `action_queue` when one is due
        and none is already waiting.
        """
        if not self.delegate or now < self._next:
            return
        if any(a.get('sel') == SEL_DID_ACCELERATE for a in action_queue):
            return                                   # previous delivery not consumed yet
        self._next = now + self.interval             # restart from now: no catch-up bursts
        self.delivered += 1
        if not self.sample_obj:
            self.sample_obj = alloc(0)
        action_queue.append({'target': self.delegate, 'sel': SEL_DID_ACCELERATE,
                             'arg2': self.obj, 'arg3': self.sample_obj})