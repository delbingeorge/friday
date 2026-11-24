# Friday

A fast, interruptible, low-cost voice assistant for macOS, powered by Claude.

Speech detection, wake word, speech-to-text and text-to-speech all run **on your Mac**.
The only paid step is the Claude reply, which costs a fraction of a cent per turn on Haiku 5.5.

- **Wake words**: Friday, Sven, Nora, Veron, Neptune (configurable, no training needed)
- **Interruptible**: just start talking over it, even on laptop speakers
- **Fast**: first audio about half a second after you stop talking, plus Claude's response time
- **Persona**: calm, precise, slightly formal; addresses you as "sir"
- **Knows your projects**: read-only access to your code and documents
- **Budget-aware**: tracks monthly spend and stops at your limit

```
mic ─► Apple echo cancellation (native Swift helper)
    ─► Silero VAD ─┬─► Vosk wake word ─► Parakeet confirms
                   └─► utterance ─► Parakeet STT (MLX)          local, ~0.1-0.3 s
                                        │
                                        ▼
                          Claude Haiku 5.5 (streaming, cached)  the only paid step
                                        │ sentence by sentence
                                        ▼
                          Kokoro TTS ─► speaker                 starts after 1st sentence
                                        ▲
                  you talk over it ─────┘ stops playback and the Claude stream
```

## Requirements

- Apple Silicon Mac, macOS 14 or later
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and dependencies)
- Xcode or the Command Line Tools (`xcode-select --install`), to compile the audio helper
- An [Anthropic API key](https://console.anthropic.com/)
- About 3 GB of disk space for models

## Setup

```bash
git clone <this repo> && cd friday
export ANTHROPIC_API_KEY=sk-ant-...       # add to ~/.zshrc to keep it
uv run friday --download-models             # VAD, wake word and voice models (~400 MB)
uv run friday                               # first start also downloads the STT model (~2.4 GB)
```

Allow microphone access for your terminal when macOS asks. The first start compiles
`native/friday_audio.swift`, which gives Friday Apple's echo cancellation.

Optional: `cp config.example.toml config.toml` and edit it. Every key is optional.

## Using it

| Say | What happens |
|---|---|
| "Friday, what's the weather in Berlin?" | Wake word and request in one breath |
| "Nora." (pause) | Wait for the chime, then ask |
| *(keep talking after a reply)* | Follow-ups work without the wake word for 6 seconds |
| *(talk over Friday)* | Playback and the Claude stream stop immediately |
| "Friday, set a timer for ten minutes" | Timer, announced out loud when done |
| "Neptune, open Safari" / "turn the volume down" | Apps, URLs, volume, Music/Spotify |
| "Veron, run my Focus shortcut" | Any macOS Shortcut by name |

A name only counts at the **start** of what you say, so "see you on Friday" won't wake it.

### Talking about your projects and files

Friday can read (never write) your code and documents and explain them by voice.

- "Nora, let's talk about octane landing." gives an overview: file tree, README, manifest,
  git branch, uncommitted changes, recent commits.
- "Where do we handle the wake word?" searches the code, reads only the relevant lines and
  refers to files by name and line number.
- "Show me." opens the file (`editor_app` picks the app).
- "Find my tax PDF." searches Documents, Desktop and Downloads.

Projects are discovered automatically under `project_roots`. Access is confined to the
configured folders, and `.env` files, keys, `.ssh`, `.aws`, credentials and database files
are refused. The parts of files Friday reads are sent to Claude to answer you.

For things Friday should always know, fill `user_context` or point `context_files` at notes
you keep up to date. They are cached, so a few pages cost almost nothing.

### Voice

The default is Kokoro's British male `bm_george`. Audition voices and blends:

```bash
uv run friday --voices
uv run friday --voices "bm_george:0.7+bm_lewis:0.3"
```

Then set `tts_voice` in `config.toml`.

## Configuration

All settings live in `config.toml` (see `config.example.toml` for every option).

| Setting | Default | Notes |
|---|---|---|
| `model` | `claude-haiku-5-5` | Sonnet 5.5 is 20x the price, Opus 5.5 40x |
| `effort` | `low` | Use `medium` for deeper code questions |
| `wake_words` | friday, sven, nora, veron, neptune | Any words in the Vosk vocabulary |
| `wake_verify` | `true` | Double-check wake words with the STT model |
| `barge_in` | `auto` | `voice` (just talk), `wakeword`, or `auto` |
| `followup_seconds` | `6.0` | Window to answer without the wake word |
| `end_silence_ms` | `650` | Lower is snappier but may cut you off |
| `web_search` | `true` | $10 per 1,000 searches |
| `monthly_budget_usd` | `100` | Friday pauses at this spend |
| `tts_voice`, `tts_speed` | `bm_george`, `1.0` | Kokoro voice, or `tts_engine = "say"` |
| `stt_model` | Parakeet 0.6B | Or a Whisper model for other languages |

Other commands: `uv run friday --list-devices`, `uv run friday -c path/to/config.toml`.

## Cost

| Per turn | Where | Cost |
|---|---|---|
| Wake word, VAD, STT, TTS | on-device | $0 |
| System prompt and tools (~2.9k tokens) | prompt cache, read at 0.1x | ~$0.00003 |
| Conversation (last 8 turns) | cached incrementally | ~$0.0001 |
| Reply (~60 tokens, thinking off) | Haiku 5.5 output | ~$0.00003 |
| Web search (only when used) | $10 per 1,000 | ~$0.011 |

Typical turns cost well under a tenth of a cent; web searches dominate when used. Spend is
saved in `~/.friday/spend.json` and printed after each turn. These figures are estimates from
token prices, not measurements.

## Battery

Idle, Friday uses about 2-3% of one core (VAD and wake word). Apple's echo cancellation, the
expensive part (15-20% of a core), is switched off while Friday is silent. It turns on when you
finish speaking and off 1.5 s after Friday stops talking. Just after it turns on it needs
about 1.6 s to adapt, so for the first 2.5 s of a reply, interrupting by voice needs a little
longer speech. Saying a wake word always interrupts at once.

## Project layout

```
src/friday/assistant.py    real-time loop: idle, listening, busy, follow-up, barge-in
src/friday/brain.py        Claude streaming, tool loop, history, persona, spend tracking
src/friday/tools.py        local tools (add your own here)
src/friday/files.py        read-only project and file access with safety rules
src/friday/wake.py         Vosk grammar wake words plus transcript confirmation
src/friday/vad.py          Silero VAD via onnxruntime
src/friday/stt.py          Parakeet / Whisper on MLX
src/friday/tts.py          Kokoro or macOS `say`
src/friday/vpio.py         echo-cancelled mic and speaker via the native helper
src/friday/audio.py        plain mic and speaker (fallback)
native/friday_audio.swift  AVAudioEngine voice processing helper
models/                  downloaded models (git-ignored)
```

## Troubleshooting

- **"I don't have an API key yet"**: `export ANTHROPIC_API_KEY=...` in the shell you run from.
- **It never wakes**: check microphone permission for your terminal, try `--list-devices`, or
  pick another wake word. The console prints `[ignored possible wake word; heard: ...]` when
  the confirmation step rejects one.
- **Wakes on things you didn't say**: keep `wake_verify = true`; avoid common words as names.
- **Can't interrupt**: look for "echo cancellation on" at startup. If it says off, the Swift
  helper failed to build; install the Command Line Tools. Without it, only wake words interrupt.
- **Other apps' audio gets quieter**: macOS voice processing ducks other audio slightly;
  Friday already requests the minimum.
- **Slow first start**: the STT model is downloading once. Later starts take about 5 s and
  skip the Hugging Face check.

## Limits

- Apple Silicon and macOS only.
- English only by default (Parakeet is English; switch `stt_model` to Whisper for others).
- Wake words must be in Vosk's small English vocabulary. "Friday" is the most likely to
  trigger by accident when you begin a sentence with it.
- Tested with synthesized speech and scripted audio; real-voice accuracy may differ.
