"""Local speech-to-text: an OpenAI-compatible /audio/transcriptions server (e.g. GigaAM). With LIVE_STT_URL set:
  * LIVE_INPUT says what the model gets of a turn: "text" (default: the transcript alone; measured on real
    utterances, Gemini's first action comes after a median 1.09 s instead of 1.79 s with the sound), "both" (the
    transcript next to the sound) or "audio" (the sound alone, as without an STT);
  * the transcript is started ahead, at SPEC_MS of silence, while the turn detector still waits for the end of the
    utterance: it is ready (or nearly) when the turn ends;
  * an utterance too short for the VAD (under MIN_SPEECH_MS voiced, a quick "удали их") is kept when its transcript
    has words; every transcript goes to the log.
"""
import asyncio
import logging
import os
import re
import time

import aiohttp

from .util import wav16k

log = logging.getLogger("live")

URL = os.environ.get("LIVE_STT_URL", "").rstrip("/")      # e.g. http://192.168.1.50:8008/v1
KEY = os.environ.get("LIVE_STT_KEY", "")
MODEL = os.environ.get("LIVE_STT_MODEL", "v3_e2e_rnnt")
LANGUAGE = os.environ.get("LIVE_STT_LANGUAGE", "ru")
TIMEOUT = float(os.environ.get("LIVE_STT_TIMEOUT", "3"))
INPUT = os.environ.get("LIVE_INPUT", "")                            # text | both | audio; "" = text with an STT
SPEC_MS = int(os.environ.get("LIVE_STT_SPEC_MS", "250"))            # silence that starts the transcript ahead
RESCUE_MIN_MS = int(os.environ.get("LIVE_STT_RESCUE_MIN_MS", "96"))  # shorter "speech" is not worth asking about

WORD_RE = re.compile(r"[^\W\d_]{2,}")
# A transcript whose last word is one of these stops in the middle of a thought ("включи свет в", "купи хлеб и").
# The recogniser ends every transcript with "." or "?" (an unfinished phrase often sounds like a question: "Сколько
# будет два +?"), so the final mark says little: prepositions, conjunctions and operators lead on whatever it is,
# words that can end a question ("как?", "что?") only without "?" / "!".
LEADS_ON = set("""и а но или либо чтобы чтоб потому поэтому если в во на с со к ко по для у из от до за о об обо про при
без над под через между перед после около плюс минус умножить разделить очень не ни самый самая самое""".split())
LEADS_ON_UNLESS_ASKED = set("""что когда где куда откуда как который которая которое которые чей ну э эм ээ эээ мм это
этот эта эти так типа короче ещё еще также потом затем""".split())
OPERATOR_END = ("+", "-", "−", "*", "×", "/", "=", ",", ":", ";", "—", "–", "…")


def enabled() -> bool:
    return bool(URL)


def mode() -> str:
    return INPUT or ("text" if URL else "audio")


def needs_text() -> bool:
    return enabled() and mode() in ("text", "both")


def has_words(text: str | None) -> bool:
    return bool(text and WORD_RE.search(text))


def leads_on(text: str | None) -> bool:
    t = (text or "").strip()
    asked = t.endswith(("?", "!"))
    core = t.rstrip(".?! ")
    if not core:
        return False
    if core.endswith(OPERATOR_END) or t.endswith("..."):
        return True
    words = re.findall(r"[^\W\d_]+", core.lower())
    if not words or not core.lower().endswith(words[-1]):
        return False
    return words[-1] in LEADS_ON or (not asked and words[-1] in LEADS_ON_UNLESS_ASKED)


async def transcribe(http: aiohttp.ClientSession, pcm: bytes, quiet: bool = False) -> str | None:
    """The transcript of 16 kHz s16 mono `pcm`; None when there is no STT or it failed (logged). `quiet`: the
    words are not logged (the answer to a question for a code, surface.py)."""
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
    log.info("stt %.2fs: %s", time.monotonic() - t0, f"({len(text)} chars, not logged)" if quiet else repr(text[:200]))
    return text
