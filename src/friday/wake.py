"""Offline custom wake words: a Vosk recognizer restricted to a tiny grammar.

Any word or phrase in the Vosk vocabulary works without training, and several
can be active at once. The grammar also lists sound-alike "distractor" words so
near misses land on those instead of on a wake word.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import vosk

vosk.SetLogLevel(-1)


class WakeWord:
    def __init__(self, model_dir: Path, phrases: list[str], distractors: list[str]):
        self.phrases = [p.lower().strip() for p in phrases]
        self._model = vosk.Model(str(model_dir))
        wake_tokens = {t for p in self.phrases for t in p.split()}
        words = [w for w in dict.fromkeys(distractors) if w.lower() not in wake_tokens]
        grammar = [*self.phrases, *words, "[unk]"]
        self._rec = vosk.KaldiRecognizer(self._model, 16000, json.dumps(grammar))
        self._pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.phrases)) + r")\b")

    def feed(self, frame_int16: np.ndarray) -> str | None:
        """Feed one frame; returns the wake word as soon as one is heard."""
        if self._rec.AcceptWaveform(frame_int16.tobytes()):
            text = json.loads(self._rec.Result()).get("text", "")
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")
        m = self._pattern.search(text) if text else None
        if m:
            self._rec.Reset()
            return m.group(1)
        return None

    def reset(self):
        self._rec.Reset()


_LEADING_FILLER = {"hey", "hi", "hello", "ok", "okay", "oh", "yo"}


def confirms_wake(transcript: str, phrases: list[str], threshold: float = 0.65) -> bool:
    """Second-stage check on an accurate transcript of the speech around a candidate.

    Accept only if the utterance *starts* with a wake word (after "hey"/"okay"), so
    "see you on Friday" or "we flew over Nepal" don't wake it. Matching is fuzzy
    because names come back as "Nora's", "Verond", "Veronica"; tokens much shorter
    than the wake word ("fry" for friday) are rejected.
    """
    toks = [re.sub(r"'s$", "", t) for t in re.findall(r"[a-z']+", transcript.lower())]
    toks = [t for t in toks if t]
    while toks and toks[0] in _LEADING_FILLER:
        toks.pop(0)
    for phrase in phrases:
        words = phrase.split()
        if len(toks) < len(words):
            continue
        heard, want = "".join(toks[: len(words)]), "".join(words)
        if heard == want or (
            len(heard) >= len(want) - 2 and SequenceMatcher(None, heard, want).ratio() >= threshold
        ):
            return True
    return False
