"""Local speech to text on Apple Silicon (MLX). No audio leaves the laptop."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Phrases Whisper tends to invent on near-silent audio.
_HALLUCINATIONS = re.compile(
    r"^\W*(thank you|thanks for watching|you|bye|okay|\.+)\W*$", re.I
)


class _SingleThread:
    """Run every call on one persistent thread (MLX/ONNX have per-thread warm-up costs)."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        return self._pool.submit(self._transcribe, audio_int16).result()


class ParakeetSTT(_SingleThread):
    """NVIDIA Parakeet TDT 0.6B: English only, very fast, no hallucination on silence."""

    def __init__(self, repo: str):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx, self._get_logmel = mx, get_logmel
        super().__init__()
        # MLX streams are per-thread: load on the same worker thread that transcribes.
        self._model = self._pool.submit(from_pretrained, repo).result()
        self.transcribe(np.zeros(16000, dtype=np.int16))  # warm up kernels

    def _transcribe(self, audio_int16: np.ndarray) -> str:
        mx = self._mx
        x = mx.array(audio_int16.astype(np.float32) / 32768.0)
        mel = self._get_logmel(x, self._model.preprocessor_config)
        return self._model.generate(mel)[0].text.strip()


class WhisperSTT(_SingleThread):
    """Whisper via mlx-whisper: multilingual, a bit slower (always encodes a 30 s window)."""

    def __init__(self, repo: str):
        import mlx_whisper

        super().__init__()
        self._mw, self._repo = mlx_whisper, repo
        self.transcribe(np.zeros(16000, dtype=np.int16))

    def _transcribe(self, audio_int16: np.ndarray) -> str:
        x = audio_int16.astype(np.float32) / 32768.0
        r = self._mw.transcribe(
            x, path_or_hf_repo=self._repo, language="en",
            condition_on_previous_text=False, verbose=None,
        )
        text = r.get("text", "").strip()
        segs = r.get("segments") or []
        if segs and all(s.get("no_speech_prob", 0) > 0.6 for s in segs):
            return ""
        return "" if _HALLUCINATIONS.match(text) else text


def make_stt(cfg):
    if "parakeet" in cfg.stt_model:
        return ParakeetSTT(cfg.stt_model)
    return WhisperSTT(cfg.stt_model)
