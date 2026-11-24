"""Local text to speech. Both engines return float32 audio at SPEAKER_RATE."""

from __future__ import annotations

import re
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import SPEAKER_RATE

_MD = re.compile(r"[*_`#>|~]+|\[([^\]]*)\]\([^)]*\)")


def clean_for_speech(text: str) -> str:
    """Strip markdown the model may still emit, keep link text."""
    text = _MD.sub(lambda m: m.group(1) or "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class KokoroTTS:
    def __init__(self, models_dir: Path, voice: str, speed: float):
        from kokoro_onnx import Kokoro

        self._k = Kokoro(str(models_dir / "kokoro-v1.0.onnx"), str(models_dir / "voices-v1.0.bin"))
        self.speed = speed
        self.set_voice(voice)
        # One long-lived worker thread: the first synth on any new thread costs ~1 s
        # of per-thread setup, which would land on every reply's first sentence.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.synth("Ready.")  # warm up the ONNX graph so the first reply isn't slow

    def synth(self, text: str) -> np.ndarray:
        return self._pool.submit(self._synth, text).result()

    def set_voice(self, spec: str):
        """'bm_george' or a blend like 'bm_george:0.7+bm_lewis:0.3'."""
        parts = []
        for item in spec.split("+"):
            name, _, w = item.strip().partition(":")
            parts.append((name, float(w) if w else 1.0))
        total = sum(w for _, w in parts)
        self._style = sum(self._k.get_voice_style(n) * (w / total) for n, w in parts)
        # b* voices are British: use British pronunciation too.
        self._lang = "en-gb" if parts[0][0].startswith("b") else "en-us"
        self.voice = spec

    def _synth(self, text: str) -> np.ndarray:
        samples, sr = self._k.create(text, voice=self._style, speed=self.speed, lang=self._lang)
        if sr != SPEAKER_RATE:
            idx = np.arange(0, len(samples), sr / SPEAKER_RATE)
            samples = np.interp(idx, np.arange(len(samples)), samples)
        return samples.astype(np.float32)


class SayTTS:
    """macOS `say`, rendered to a buffer so playback stays interruptible."""

    def __init__(self, voice: str = "", speed: float = 1.0):
        self.voice, self.rate = voice, int(185 * speed)

    def synth(self, text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            cmd = ["say", "-r", str(self.rate), "-o", f.name,
                   "--file-format=WAVE", f"--data-format=LEI16@{SPEAKER_RATE}"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run([*cmd, "--", text], check=True, capture_output=True)
            with wave.open(f.name) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return (pcm.astype(np.float32) / 32768.0)


def make_tts(cfg):
    if cfg.tts_engine == "kokoro":
        return KokoroTTS(cfg.models_path, cfg.tts_voice, cfg.tts_speed)
    return SayTTS(cfg.say_voice, cfg.tts_speed)
