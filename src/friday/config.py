"""Configuration: defaults below, overridden by an optional config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- Claude -------------------------------------------------------------
    model: str = "claude-haiku-5-5"
    effort: str = "low"  # low keeps replies fast and cheap; thinking is disabled
    max_tokens: int = 1024
    web_search: bool = True
    web_search_max_uses: int = 2
    # Stable extra context for the system prompt (name, city, preferences...).
    # Keep it fixed: changing it invalidates the prompt cache.
    user_context: str = ""
    # Text files always included in Friday's context (notes about you, current priorities...).
    # Read once at startup; they're cached, so a few pages cost almost nothing per turn.
    context_files: list[str] = field(default_factory=list)
    history_turns: int = 8  # user turns kept in context
    session_idle_reset_s: float = 300.0  # forget the conversation after this much quiet
    monthly_budget_usd: float = 100.0

    # --- Files & projects (read-only tools) ----------------------------------
    file_access: bool = True
    # Folders scanned for projects (git repos / package.json / pyproject.toml...).
    project_roots: list[str] = field(default_factory=lambda: ["~/Documents/work-space"])
    # Extra or renamed projects: { "name" = "~/path" }
    projects: dict[str, str] = field(default_factory=dict)
    # Other folders Friday may search and read (outside projects).
    file_roots: list[str] = field(default_factory=lambda: ["~/Documents", "~/Desktop", "~/Downloads"])
    editor_app: str = ""  # app for open_file, e.g. "Visual Studio Code"; empty = default text editor

    # --- Wake word / listening ----------------------------------------------
    # Any of these wakes the assistant (words or short phrases in the Vosk vocabulary).
    wake_words: list[str] = field(default_factory=lambda: ["friday", "sven", "nora", "veron", "neptune"])
    # Double-check each candidate with the accurate STT model (~0.1 s) and require the
    # wake word to start the utterance ("see you on Friday" won't wake it).
    wake_verify: bool = True
    # In-vocabulary sound-alikes give the grammar recognizer somewhere else to
    # put near misses, which cuts false wakes. Entries equal to a wake word are ignored.
    wake_distractors: list[str] = field(
        default_factory=lambda: [
            # sven
            "seven", "steven", "even", "heaven", "then", "when", "ten",
            "send", "spend", "seven's", "svelte", "swan", "sweden",
            # nora
            "dora", "laura", "aura", "flora", "nova", "noah", "north", "normal",
            # veron
            "vernon", "baron", "heron", "verona", "sharon", "aaron", "vern",
            # friday
            "fried", "fry", "freddy", "friend", "monday", "tuesday", "sunday",
            # neptune
            "nepal", "napkin", "tune",
        ]
    )
    # macOS voice processing (Apple's echo cancellation) so Friday doesn't hear itself.
    echo_cancellation: bool = True
    # How to interrupt Friday mid-reply. Saying the wake word always works.
    # "voice": just start talking (needs echo cancellation or headphones).
    # "wakeword": only the wake word interrupts.
    # "auto": "voice" when echo cancellation is running, otherwise "wakeword".
    barge_in: str = "auto"
    followup_seconds: float = 6.0  # listen without wake word after a reply
    end_silence_ms: int = 650  # silence that ends an utterance
    listen_timeout_s: float = 5.0  # wait this long for speech after the wake word
    max_utterance_s: float = 20.0
    vad_threshold: float = 0.5
    barge_in_ms: int = 300  # speech needed to interrupt in "voice" mode
    input_device: str | int | None = None
    output_device: str | int | None = None

    # --- Speech to text / text to speech ------------------------------------
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v2"  # or whisper-large-v3-turbo (multilingual, ~0.9 s)
    tts_engine: str = "kokoro"  # "kokoro" (local neural) or "say" (macOS built-in)
    # Calm British male. Blends work too: "bm_george:0.7+bm_lewis:0.3".
    # Audition with: uv run friday --voices
    tts_voice: str = "bm_george"
    tts_speed: float = 1.0
    say_voice: str = ""  # e.g. "Samantha"; empty uses the system voice

    models_dir: str = "models"
    data_dir: str = "~/.friday"

    @property
    def models_path(self) -> Path:
        p = Path(self.models_dir).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.toml"
    cfg = Config()
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    if "wake_phrase" in data:  # older single-word setting
        data.setdefault("wake_words", [data.pop("wake_phrase")])
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
    for k, v in data.items():
        setattr(cfg, k, v)
    cfg.wake_words = [w.lower().strip() for w in cfg.wake_words if w.strip()]
    if not cfg.wake_words:
        raise SystemExit("wake_words must contain at least one word")
    if cfg.barge_in not in ("auto", "wakeword", "voice"):
        raise SystemExit('barge_in must be "auto", "wakeword" or "voice"')
    if cfg.tts_engine not in ("kokoro", "say"):
        raise SystemExit('tts_engine must be "kokoro" or "say"')
    return cfg
