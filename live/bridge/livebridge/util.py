import io
import os
import re
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # live/bridge/
TAG_RE = re.compile(r"\[(?:emo:[a-z]+|SILENT)\]", re.I)

# Zero data retention: what the bridge sends to the cloud goes only to endpoints that keep none of it (OpenRouter's
# provider.zdr; backends that cannot promise that refuse to run). LIVE_ZDR=0 lifts it.
ZDR = os.environ.get("LIVE_ZDR", "1") != "0"


class BackendUnavailable(Exception):
    """Raised by a backend before it has said anything to the station: the bridge then hands the dialog off."""


def wav16k(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return b.getvalue()


def clean_for_speech(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = re.sub(r"[*_`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def read_instructions(env: str = "LIVE_INSTRUCTIONS", name: str = "instructions") -> str:
    """The persona prompt, re-read for every dialog so that editing it needs no restart. A path in $env wins, then
    <name>.txt next to the bridge, then the shipped <name>.example.txt."""
    for p in (os.environ.get(env), HERE / f"{name}.txt", HERE / f"{name}.example.txt"):
        if p and Path(p).is_file():
            return Path(p).read_text(encoding="utf-8")
    raise BackendUnavailable(f"no {name}.txt (set ${env})")
