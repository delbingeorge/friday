"""The real-time loop: wake word -> listen -> transcribe -> Claude -> speak, with barge-in."""

from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from enum import Enum, auto

import numpy as np

from .audio import FRAME, LISTEN_EARCON, MIC_RATE, Microphone, Speaker
from .brain import Brain, SpendTracker
from .config import Config
from .stt import make_stt
from .tools import ToolRunner
from .tts import clean_for_speech, make_tts
from .vad import SileroVAD
from .wake import WakeWord, confirms_wake

FRAME_MS = FRAME * 1000 / MIC_RATE  # 32 ms
WAKE_PREROLL = 12  # ~0.4 s: catches the start of a command said in one breath with the wake word
VERIFY_FRAMES = 8  # ~0.25 s: let the candidate word finish before double-checking it


class State(Enum):
    IDLE = auto()  # waiting for the wake word
    LISTENING = auto()  # recording the user's request
    BUSY = auto()  # transcribing / thinking / speaking
    FOLLOWUP = auto()  # just answered: listening for a follow-up without the wake word


class Turn:
    """One request/response, run off the audio thread so the mic keeps being watched."""

    def __init__(self, app: "Assistant", audio: np.ndarray, ended_at: float):
        self.app = app
        self.audio = audio
        self.ended_at = ended_at  # when the user stopped talking
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        app = self.app
        try:
            text = app.stt.transcribe(self.audio)
            text = app.strip_wake(text)
            if self.cancel.is_set() or not text:
                return
            stt_ms = (time.monotonic() - self.ended_at) * 1000
            print(f"\nYou:  {text}   [stt {stt_ms:.0f} ms]")

            chunks: queue.Queue[str | None] = queue.Queue()
            first_audio = []
            speaker_thread = threading.Thread(target=self._speak, args=(chunks, first_audio), daemon=True)
            speaker_thread.start()

            stats = app.brain.respond(text, self.cancel, chunks.put)
            chunks.put(None)
            speaker_thread.join()

            lat = (first_audio[0] - self.ended_at) * 1000 if first_audio else None
            if lat is not None:
                print(f"   [first audio {lat:.0f} ms after you stopped · "
                      f"${stats['cost']:.5f} · month ${app.spend.month_total:.2f}]")
            if stats["cancelled"]:
                print("   [interrupted]")
        except Exception as e:  # noqa: BLE001 - never let a turn kill the loop
            print(f"   [error] turn failed: {e!r}")
        finally:
            self.done.set()

    def _speak(self, chunks: queue.Queue, first_audio: list):
        app = self.app
        print("Friday: ", end="", flush=True)
        while (chunk := chunks.get()) is not None:
            if self.cancel.is_set():
                continue  # drain without speaking
            print(chunk, end=" ", flush=True)
            spoken = clean_for_speech(chunk)
            if not spoken:
                continue
            audio = app.tts.synth(spoken)
            if self.cancel.is_set():
                continue
            app.speaker.play(audio, flag=app.mentions_wake(spoken))
            if not first_audio:
                first_audio.append(time.monotonic())
        print()

    def interrupt(self):
        self.cancel.set()
        self.app.speaker.stop()
        self.app.brain.abort()

    @property
    def finished(self) -> bool:
        return self.done.is_set() and not self.app.speaker.busy


class Assistant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        t = time.monotonic()
        print("Loading models…", flush=True)
        self.vad = SileroVAD(cfg.models_path / "silero_vad.onnx")
        self.wake = WakeWord(cfg.models_path / "vosk-model-small-en-us-0.15", cfg.wake_words, cfg.wake_distractors)
        self._wake_list = " / ".join(f"'{w}'" for w in cfg.wake_words)
        self.stt = make_stt(cfg)
        self.tts = make_tts(cfg)
        self.speaker, self.mic, aec = self._open_audio(cfg)
        self.barge_in = cfg.barge_in if cfg.barge_in != "auto" else ("voice" if aec else "wakeword")
        self.spend = SpendTracker(cfg.data_path / "spend.json", cfg.model)
        files = None
        if cfg.file_access:
            from .files import FileAccess

            files = FileAccess(cfg.project_roots, cfg.projects, cfg.file_roots)
        self.brain = Brain(cfg, ToolRunner(self.announce, files, cfg.editor_app), self.spend)
        how = "just talk" if self.barge_in == "voice" else f"say {self._wake_list}"
        print(f"Ready in {time.monotonic() - t:.1f}s · model {cfg.model} · "
              f"echo cancellation {'on' if aec else 'off'} · to interrupt, {how} · "
              f"spent ${self.spend.month_total:.2f} of ${cfg.monthly_budget_usd:.0f} this month")

        w = "|".join(map(re.escape, cfg.wake_words))
        self._wake_prefix = re.compile(rf"^\W*(?:(?:hey|hi|ok|okay)\W+)?(?:{w})\b\W*", re.I)
        self._wake_any = re.compile(rf"\b(?:{w})\b", re.I)

        self.state: State | None = None  # run() moves to IDLE and prints the prompt
        self.turn: Turn | None = None
        self._preroll: deque[np.ndarray] = deque(maxlen=int(800 / FRAME_MS))
        self._recent: deque[tuple[np.ndarray, bool]] = deque(maxlen=int(2500 / FRAME_MS))

    @staticmethod
    def _open_audio(cfg: Config):
        """Returns (speaker, mic, echo_cancelled)."""
        if cfg.echo_cancellation:
            try:
                from .vpio import VoiceProcessingIO

                io = VoiceProcessingIO()
                return io, io, True
            except Exception as e:  # noqa: BLE001 - fall back to plain audio
                print(f"[warning] echo cancellation unavailable ({e}); "
                      f"interrupt by saying a wake word instead.")
        return Speaker(cfg.output_device), Microphone(cfg.input_device), False

    # --- helpers --------------------------------------------------------------
    def strip_wake(self, text: str) -> str:
        text = text.strip()
        out = self._wake_prefix.sub("", text)
        if out == text:
            # The pre-roll may hold only the tail of a wake word: "tune" (Neptune),
            # "Ven," / "Ben," (Sven).
            m = re.match(r"^\W*(\w+)([,.!?]?)\s+(.+)$", text, re.S)
            if m:
                frag, punct = m.group(1).lower(), m.group(2)
                singles = [w for w in self.cfg.wake_words if " " not in w]
                if any(len(frag) >= 3 and w.endswith(frag) for w in singles) or (
                    punct and any(len(frag) <= len(w) + 2 and frag[-2:] == w[-2:] for w in singles)
                ):
                    out = m.group(3)
        return out.strip()

    def _confirm_wake(self, frames_after: int) -> bool:
        """Transcribe the speech segment containing a wake candidate and check it starts with a wake word."""
        items = list(self._recent)
        d = len(items) - 1 - frames_after
        start, silence, seen_speech = 0, 0, False
        for i in range(d, -1, -1):  # walk back to the pause before the candidate
            if items[i][1]:
                seen_speech, silence = True, 0
            elif seen_speech:
                silence += 1
                if silence >= 6:  # ~0.2 s gap
                    start = i + silence
                    break
        # Back up ~0.2 s: VAD misses soft onsets like the "s" of "Sven".
        start = max(start - 6, 0, len(items) - int(2000 / FRAME_MS))
        text = self.stt.transcribe(np.concatenate([f for f, _ in items[start:]]))
        ok = confirms_wake(text, self.cfg.wake_words)
        if not ok:
            print(f"[ignored possible wake word; heard: \"{text}\"]")
        return ok

    def mentions_wake(self, text: str) -> bool:
        return bool(self._wake_any.search(text))

    def announce(self, text: str):
        """Speak something unprompted (timers). Queues behind any current speech."""
        print(f"\n[timer] {text}")
        self.speaker.play(self.tts.synth(text), flag=self.mentions_wake(text))

    def _set(self, state: State):
        if state != self.state:
            label = {State.IDLE: f"[idle] say {self._wake_list}", State.LISTENING: "[listening]",
                     State.FOLLOWUP: "[listening for a follow-up]", State.BUSY: None}[state]
            if label:
                print(label, flush=True)
        self.state = state

    # --- main loop --------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        end_silence_frames = int(cfg.end_silence_ms / FRAME_MS)
        barge_frames = int(cfg.barge_in_ms / FRAME_MS)
        settle_frames = max(barge_frames, int(640 / FRAME_MS))
        max_frames = int(cfg.max_utterance_s * 1000 / FRAME_MS)
        min_speech_frames = int(200 / FRAME_MS)

        utterance: list[np.ndarray] = []
        pending: list | None = None  # [candidate wake word, frames since it was heard]
        wake_preroll = WAKE_PREROLL
        speech_frames = silence_frames = speech_run = 0
        listen_started = followup_until = followup_armed_at = 0.0

        def start_listening(earcon: bool, preroll_frames: int):
            nonlocal utterance, speech_frames, silence_frames, listen_started
            if self.turn and not self.turn.finished:
                self.turn.interrupt()
            if earcon:
                self.speaker.play(LISTEN_EARCON)
            utterance = list(self._preroll)[-preroll_frames:] if preroll_frames else []
            # Count speech already in the pre-roll: a short command ("Sven, stop.") can
            # finish while the wake word is being confirmed.
            flags = [sp for _, sp in list(self._recent)[-len(utterance):]] if utterance else []
            speech_frames = sum(flags)
            silence_frames = 0
            if speech_frames:
                while silence_frames < len(flags) and not flags[-1 - silence_frames]:
                    silence_frames += 1
            listen_started = time.monotonic()
            self._set(State.LISTENING)

        self.mic.start()
        self._set(State.IDLE)
        try:
            while True:
                frame = self.mic.read()
                if frame is None:
                    continue
                now = time.monotonic()
                speech = self.vad(frame) >= cfg.vad_threshold
                speech_run = speech_run + 1 if speech else 0
                self._preroll.append(frame)
                self._recent.append((frame, speech))

                # Two-stage wake word: the cheap grammar recognizer proposes a candidate,
                # then (after the word finishes) the accurate STT model confirms it.
                woke = None
                if self.state == State.LISTENING:
                    pending = None
                else:
                    cand = self.wake.feed(frame)
                    # Ignore the assistant saying its own name (allow for recognizer lag).
                    if cand and now - self.speaker.flagged_audible_at < 1.5:
                        cand = None
                    if cand and not cfg.wake_verify:
                        woke, wake_preroll = cand, WAKE_PREROLL
                    elif cand and pending is None:
                        pending = [cand, 0]
                    elif pending is not None:
                        pending[1] += 1
                        if pending[1] >= VERIFY_FRAMES:
                            word, waited = pending
                            pending = None
                            if self._confirm_wake(waited):
                                woke, wake_preroll = word, WAKE_PREROLL + waited

                if self.state == State.IDLE:
                    if woke:
                        start_listening(earcon=True, preroll_frames=wake_preroll)

                elif self.state == State.FOLLOWUP:
                    if woke:
                        start_listening(earcon=True, preroll_frames=wake_preroll)
                    elif now >= followup_armed_at and speech_run >= 3:
                        start_listening(earcon=False, preroll_frames=len(self._preroll))
                    elif now > followup_until:
                        self._set(State.IDLE)

                elif self.state == State.LISTENING:
                    utterance.append(frame)
                    if speech:
                        speech_frames += 1
                        silence_frames = 0
                    elif speech_frames:
                        silence_frames += 1

                    if speech_frames >= min_speech_frames and (
                        silence_frames >= end_silence_frames or len(utterance) >= max_frames
                    ):
                        audio = np.concatenate(utterance)
                        if hasattr(self.speaker, "prepare"):
                            self.speaker.prepare()  # echo cancellation on before the reply starts
                        self.turn = Turn(self, audio, ended_at=now)
                        self.wake.reset()
                        self._set(State.BUSY)
                    elif not speech_frames and now - listen_started > cfg.listen_timeout_s:
                        self._set(State.IDLE)
                    elif speech_frames and speech_frames < min_speech_frames and silence_frames > 2 * end_silence_frames:
                        self._set(State.IDLE)  # just a click or cough

                elif self.state == State.BUSY:
                    if woke:
                        start_listening(earcon=True, preroll_frames=wake_preroll)
                    elif self.barge_in == "voice" and speech_run >= (
                        # Just after echo cancellation switches on it leaks up to ~0.35 s of
                        # Friday's own voice while it re-adapts; demand longer speech then.
                        settle_frames if now - getattr(self.speaker, "aec_enabled_at", 0.0) < 2.5
                        else barge_frames
                    ):
                        # Also covers "what's the weather… in Paris": the cancelled turn's
                        # question stays in history, so the API merges it with the new one.
                        start_listening(earcon=False, preroll_frames=len(self._preroll))
                    elif self.turn and self.turn.finished:
                        followup_until = now + cfg.followup_seconds
                        followup_armed_at = now + 0.3  # let the room echo die down
                        self._set(State.FOLLOWUP if cfg.followup_seconds > 0 else State.IDLE)
        except KeyboardInterrupt:
            print("\nBye!")
        finally:
            if self.turn:
                self.turn.interrupt()
            self.mic.close()
            self.speaker.close()
