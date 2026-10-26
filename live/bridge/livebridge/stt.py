"""Local speech-to-text: an OpenAI-compatible /audio/transcriptions server (e.g. GigaAM) as a second opinion on what
the owner said. With LIVE_STT_URL set:
  * an utterance too short for the VAD (under MIN_SPEECH_MS voiced, a quick "удали их") is transcribed, and kept
    when the transcript has words;
  * every utterance's transcript goes to the log;
  * LIVE_STT_HINT (on by default): the transcript also goes to the model next to the audio (it waits for it, ~0.3 s).
"""
import asyncio
import logging
import os
import re
import time

import aiohttp

from .util import wav16k

log = logging.getLogger("live")

URL = os.environ.get("LIVE_STT_URL", "").rstrip("/")      # e.g. http://192.168.1.125:8008/v1
KEY = os.environ.get("LIVE_STT_KEY", "")
MODEL = os.environ.get("LIVE_STT_MODEL", "v3_e2e_rnnt")
LANGUAGE = os.environ.get("LIVE_STT_LANGUAGE", "ru")
TIMEOUT = float(os.environ.get("LIVE_STT_TIMEOUT", "3"))
HINT = os.environ.get("LIVE_STT_HINT", "1") != "0"
RESCUE_MIN_MS = int(os.environ.get("LIVE_STT_RESCUE_MIN_MS", "96"))  # shorter "speech" is not worth asking about

WORD_RE = re.compile(r"[^\W\d_]{2,}")


def enabled() -> bool:
    return bool(URL)


def has_words(text: str | None) -> bool:
    return bool(text and WORD_RE.search(text))


async def transcribe(http: aiohttp.ClientSession, pcm: bytes) -> str | None:
    """The transcript of 16 kHz s16 mono `pcm`; None when there is no STT or it failed (logged)."""
    if not URL or http is None:
        return None
    form = aiohttp.FormData()
    form.add_field("file", wav16k(pcm), filename="utterance.wav", content_type="audio/wav")
    form.add_field("model", MODEL)
    form.add_field("language", LANGUAGE)
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    t0 = time.monotonic()
    try:
        async with http.post(f"{URL}/audio/transcriptions", data=form, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            if r.status != 200:
                log.warning("stt HTTP %s: %s", r.status, (await r.text())[:200])
                return None
            data = await r.json(content_type=None)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("stt failed after %.2fs: %r", time.monotonic() - t0, e)
        return None
    text = str((data or {}).get("text") or "").strip()
    log.info("stt %.2fs: %r", time.monotonic() - t0, text[:200])
    return text
