"""Friday: a fast, interruptible, low-cost voice assistant for macOS, powered by Claude."""

import argparse
import os
import sys


DEFAULT_AUDITION = ["bm_george", "bm_fable", "bm_lewis", "bm_daniel", "bm_george:0.6+bm_fable:0.4"]
SAMPLE_LINE = ("Good evening, sir. It appears your build has failed again. "
               "I recommend reading the error message before trying a third time.")


def audition(cfg, voices: list[str]) -> None:
    import time

    from .audio import Speaker
    from .tts import KokoroTTS

    tts = KokoroTTS(cfg.models_path, cfg.tts_voice, cfg.tts_speed)
    spk = Speaker(cfg.output_device)
    for v in voices or DEFAULT_AUDITION:
        tts.set_voice(v)
        print(f"tts_voice = \"{v}\"", flush=True)
        spk.play(tts.synth(SAMPLE_LINE))
        while spk.busy:
            time.sleep(0.05)
        time.sleep(0.6)
    spk.close()


def _hf_cached(repo: str) -> bool:
    """True if a Hugging Face repo already has a downloaded snapshot (checked without importing the hub)."""
    from pathlib import Path

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snapshots = Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots"
    return any(any(p.glob("*.safetensors")) for p in snapshots.glob("*")) if snapshots.is_dir() else False


def main() -> None:
    parser = argparse.ArgumentParser(prog="friday", description=__doc__)
    parser.add_argument("-c", "--config", help="path to config.toml (default: ./config.toml)")
    parser.add_argument("--download-models", action="store_true", help="fetch local models and exit")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--voices", nargs="*", metavar="VOICE",
                        help="audition voices (default: the British male set) and exit")
    args = parser.parse_args()

    from .config import load_config

    cfg = load_config(args.config)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    from . import models

    if args.download_models or models.missing(cfg.models_path):
        models.download(cfg.models_path)
        if args.download_models:
            return

    if args.voices is not None:
        audition(cfg, args.voices)
        return

    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if _hf_cached(cfg.stt_model):
        # Model already downloaded: skip the Hugging Face update check at every start
        # (slow on poor connections, and it prints an unauthenticated-request warning).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Note: ANTHROPIC_API_KEY is not set; relying on an `ant auth login` profile if you have one.",
              file=sys.stderr)

    from .assistant import Assistant

    Assistant(cfg).run()
