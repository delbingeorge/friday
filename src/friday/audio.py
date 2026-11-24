"""Microphone input and interruptible speaker output."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

MIC_RATE = 16000
FRAME = 512  # 32 ms at 16 kHz: the frame size Silero VAD expects
SPEAKER_RATE = 24000  # Kokoro's native rate; `say` output is rendered at this rate too


class Microphone:
    """Pushes 512-sample int16 frames onto a queue from the PortAudio thread."""

    def __init__(self, device=None):
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream = sd.InputStream(
            samplerate=MIC_RATE, channels=1, dtype="int16",
            blocksize=FRAME, device=device, callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):
        try:
            self.frames.put_nowait(indata[:, 0].copy())
        except queue.Full:  # consumer stalled; drop rather than block audio thread
            pass

    def start(self):
        self._stream.start()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stream.stop()
        self._stream.close()


class Speaker:
    """Plays float32 chunks back to back; stop() cuts playback within one block."""

    def __init__(self, device=None):
        self._q: deque[tuple[np.ndarray, bool]] = deque()
        self._cur: np.ndarray | None = None
        self._cur_flag = False
        self._pos = 0
        # Last time a chunk containing the wake word was audible, so the main loop
        # can ignore the assistant "waking itself" by saying its own name.
        self.flagged_audible_at = 0.0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=SPEAKER_RATE, channels=1, dtype="float32",
            blocksize=480, device=device, callback=self._cb,  # 20 ms blocks
        )
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        n = 0
        with self._lock:
            while n < frames:
                if self._cur is None:
                    if not self._q:
                        break
                    (self._cur, self._cur_flag), self._pos = self._q.popleft(), 0
                take = min(frames - n, len(self._cur) - self._pos)
                out[n:n + take] = self._cur[self._pos:self._pos + take]
                n += take
                self._pos += take
                if self._pos >= len(self._cur):
                    self._cur = None
            if n and self._cur_flag:
                self.flagged_audible_at = time.monotonic()
        out[n:] = 0.0

    def play(self, samples: np.ndarray, flag: bool = False):
        with self._lock:
            self._q.append((np.ascontiguousarray(samples, dtype=np.float32), flag))

    def stop(self):
        with self._lock:
            self._q.clear()
            self._cur = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._cur is not None or bool(self._q)

    def close(self):
        self.stop()
        self._stream.stop()
        self._stream.close()


def tone(freq: float, ms: int, volume: float = 0.18) -> np.ndarray:
    """A short sine blip with soft edges, used as the 'I'm listening' earcon."""
    t = np.arange(int(SPEAKER_RATE * ms / 1000)) / SPEAKER_RATE
    wave = np.sin(2 * np.pi * freq * t) * volume
    ramp = min(len(wave) // 4, int(SPEAKER_RATE * 0.01))
    env = np.ones_like(wave)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (wave * env).astype(np.float32)


LISTEN_EARCON = np.concatenate([tone(880, 60), tone(1320, 70)])
