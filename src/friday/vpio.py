"""Echo-cancelled audio via the native `friday-audio` helper (macOS Voice Processing I/O).

One object serves as both the Microphone and the Speaker: Apple's echo canceller
must own the playback path to subtract Friday's voice from what the mic hears.
"""

from __future__ import annotations

import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "native" / "friday_audio.swift"
BINARY = PROJECT_ROOT / "build" / "friday-audio"


def ensure_built() -> Path:
    if not BINARY.exists() or BINARY.stat().st_mtime < SOURCE.stat().st_mtime:
        print("Compiling echo-cancellation helper…", flush=True)
        BINARY.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["swiftc", "-O", "-swift-version", "5", str(SOURCE), "-o", str(BINARY)],
            check=True, capture_output=True, text=True,
        )
    return BINARY


class VoiceProcessingIO:
    def __init__(self):
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._pending: dict[int, bool] = {}  # buffer id -> mentions wake word
        self._next_id = 0
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._flagged_done_at = 0.0
        self._ready = threading.Event()
        self._closed = False
        # Apple's echo cancellation costs ~10-20% of a core; it's only needed while Friday
        # is talking, so it's bypassed when idle (the helper then uses under 1%).
        self._bypassed: bool | None = None
        self._bypass_lock = threading.Lock()
        self._idle_timer: threading.Timer | None = None
        self.aec_enabled_at = 0.0  # echo cancellation needs ~1.6 s of Friday's voice to re-adapt
        self._proc = subprocess.Popen([str(ensure_built())], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        threading.Thread(target=self._reader, daemon=True).start()
        if not self._ready.wait(10):
            self.close()
            raise RuntimeError("friday-audio did not start (microphone permission?)")
        self._set_bypass(True)

    def _reader(self):
        out = self._proc.stdout
        while True:
            head = out.read(5)
            if len(head) < 5:
                break
            kind, n = head[:1], struct.unpack("<I", head[1:])[0]
            body = out.read(n) if n else b""
            if kind == b"M":
                try:
                    self.frames.put_nowait(np.frombuffer(body, dtype=np.int16))
                except queue.Full:
                    pass
            elif kind == b"D":
                with self._lock:
                    if self._pending.pop(struct.unpack("<I", body)[0], False):
                        self._flagged_done_at = time.monotonic()
                    drained = not self._pending
                if drained:
                    self._schedule_bypass()
            elif kind == b"R":
                self._ready.set()

    def _send(self, kind: bytes, payload: bytes = b""):
        if self._closed:
            return
        with self._write_lock:
            try:
                self._proc.stdin.write(kind + struct.pack("<I", len(payload)) + payload)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    # --- power saving ---------------------------------------------------------
    def _set_bypass(self, on: bool):
        with self._bypass_lock:
            if self._bypassed != on:
                self._send(b"B", b"\x01" if on else b"\x00")
                self._bypassed = on
                if not on:
                    self.aec_enabled_at = time.monotonic()

    def _schedule_bypass(self, delay: float = 1.5):
        """Bypass echo cancellation once playback has been quiet for `delay` seconds."""
        if self._idle_timer:
            self._idle_timer.cancel()

        def fire():
            if not self.busy:
                self._set_bypass(True)

        self._idle_timer = threading.Timer(delay, fire)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def prepare(self):
        """Switch echo cancellation on ahead of a reply so it has settled by the first word."""
        if self._idle_timer:
            self._idle_timer.cancel()
        self._set_bypass(False)
        self._schedule_bypass(delay=20)  # fallback if no reply ever plays

    # --- Microphone interface ------------------------------------------------
    def start(self):
        pass  # already running

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if self._proc.poll() is not None:
            raise RuntimeError("friday-audio exited")
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- Speaker interface ---------------------------------------------------
    def play(self, samples: np.ndarray, flag: bool = False):
        if self._idle_timer:
            self._idle_timer.cancel()
        self._set_bypass(False)  # Friday is about to talk: cancel its echo
        with self._lock:
            bid = self._next_id
            self._next_id = (self._next_id + 1) & 0xFFFFFFFF
            self._pending[bid] = flag
        pcm = np.ascontiguousarray(samples, dtype="<f4").tobytes()
        self._send(b"P", struct.pack("<I", bid) + pcm)

    def stop(self):
        with self._lock:
            self._pending.clear()
        self._send(b"S")
        self._schedule_bypass()

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    @property
    def flagged_audible_at(self) -> float:
        with self._lock:
            if any(self._pending.values()):
                return time.monotonic()
            return self._flagged_done_at

    def close(self):
        if self._closed:
            return
        self._send(b"Q")
        self._closed = True
        try:
            self._proc.wait(2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
