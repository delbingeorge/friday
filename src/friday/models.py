"""Download the small local models Friday needs (STT models come from Hugging Face on first run)."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
VOSK = ("vosk-model-small-en-us-0.15", "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")


def missing(models_dir: Path) -> list[str]:
    names = [n for n in FILES if not (models_dir / n).exists()]
    if not (models_dir / VOSK[0]).is_dir():
        names.append(VOSK[0])
    return names


def download(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = models_dir / name
        if dest.exists():
            continue
        print(f"Downloading {name}…", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    if not (models_dir / VOSK[0]).is_dir():
        print(f"Downloading {VOSK[0]}…", flush=True)
        with urllib.request.urlopen(VOSK[1]) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(models_dir)
    print("Models ready. Speech-to-text weights download from Hugging Face on first start.")
