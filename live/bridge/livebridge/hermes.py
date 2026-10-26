"""The home agent ("Hermes"): any OpenAI-compatible chat endpoint that can act in the real world."""
import logging
import os
import time

import aiohttp

from .util import TAG_RE

log = logging.getLogger("live")

HERMES_URL = os.environ.get("HERMES_URL", "http://127.0.0.1:8642/v1/chat/completions")
HERMES_KEY = os.environ.get("HERMES_KEY", "")
HERMES_TIMEOUT = float(os.environ.get("HERMES_TIMEOUT", "120"))
# Prefix that tells a router in front of the agent: this is a live voice request (cloud model, fast path). "" = none.
HERMES_MARK = os.environ.get("HERMES_MARK", "@@LIVE@@")


async def ask_hermes(http: aiohttp.ClientSession, request: str) -> str:
    t0 = time.monotonic()
    content = f"{HERMES_MARK} {request}" if HERMES_MARK else request
    body = {"model": "hermes", "stream": False, "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {HERMES_KEY}"} if HERMES_KEY else {}
    async with http.post(HERMES_URL, json=body, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=HERMES_TIMEOUT)) as r:
        data = await r.json(content_type=None)
        status = r.status
    if status != 200:
        raise RuntimeError(f"hermes HTTP {status}: {str(data)[:200]}")
    text = TAG_RE.sub("", data["choices"][0]["message"].get("content") or "").strip()
    log.info("hermes %.1fs: %r -> %r", time.monotonic() - t0, request[:120], text[:160])
    return text or "Сделано, ответ пустой."
