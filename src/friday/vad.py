"""Silero VAD (v5) run directly through onnxruntime: ~0.1 ms per 32 ms frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

CONTEXT = 64  # samples of the previous frame the v5 model expects in front of each frame


class SileroVAD:
    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame_int16: np.ndarray) -> float:
        """Speech probability for one 512-sample int16 frame."""
        x = (frame_int16.astype(np.float32) / 32768.0)[None, :]
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -CONTEXT:]
        return float(out[0][0])
