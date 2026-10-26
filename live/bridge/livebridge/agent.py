"""The home agent: what a dialog model calls for anything that touches the real world (the pipeline's ask_hermes
tool, gpt-live's delegation). Two ways to reach it (LIVE_AGENT, or the "agent" option of a route):

  ha      (default) Home Assistant's conversation API, POST /api/conversation/process, with the conversation agent of
          the voice pipeline (HA_AGENT_ID, e.g. conversation.extended_openai_conversation in front of Hermes). HA stays
          the one door to the agent: the request lands in HA's chat log, the agent learns which station asked (the
          route options ha_device_id / ha_satellite_id: area context), and changing the agent in HA changes it for
          Live dialogs too. One Live dialog is one HA conversation (its conversation_id is kept between requests).
  hermes  the agent's own OpenAI-compatible endpoint (HERMES_URL), past Home Assistant.

There is no fallback from one to the other: a request the chosen way cannot answer is reported to the model as failed.
Either way the request carries HERMES_MARK, the prefix a model router in front of the agent recognises Live requests by.
"""
import json
import logging
import os
import time

import aiohttp

from .util import TAG_RE, BackendUnavailable

log = logging.getLogger("live")

AGENT = os.environ.get("LIVE_AGENT", "ha")
TIMEOUT = float(os.environ.get("LIVE_AGENT_TIMEOUT") or os.environ.get("HERMES_TIMEOUT") or "120")
# Prefix that tells a router in front of the agent: this is a live voice request (cloud model, fast path). "" = none.
MARK = os.environ.get("HERMES_MARK", "@@LIVE@@")

HA_URL = os.environ.get("HA_URL", "").rstrip("/")         # e.g. http://192.168.1.10:8123
HA_TOKEN = os.environ.get("HA_TOKEN", "")                  # long-lived access token of an HA user
HA_AGENT_ID = os.environ.get("HA_AGENT_ID", "")            # "" = HA's default agent (its built-in intents only)
HA_LANGUAGE = os.environ.get("HA_LANGUAGE", "ru")
HA_KEEP_CONVERSATION = os.environ.get("LIVE_HA_KEEP_CONVERSATION", "1") != "0"

HERMES_URL = os.environ.get("HERMES_URL", "http://127.0.0.1:8642/v1/chat/completions")
HERMES_KEY = os.environ.get("HERMES_KEY", "")

KINDS = ("ha", "hermes")


class Agent:
    """The agent as one dialog sees it: one per dialog, it carries that dialog's HA conversation."""

    def __init__(self, http: aiohttp.ClientSession, options: dict | None = None):
        o = options or {}
        self.http = http
        self.kind = str(o.get("agent") or AGENT)
        self.agent_id = str(o.get("ha_agent_id") or HA_AGENT_ID)
        self.device_id = o.get("ha_device_id") or None
        self.satellite_id = o.get("ha_satellite_id") or None
        self.conversation_id = None
        self.in_flight = 0

    def check(self):
        """Raise BackendUnavailable if this agent cannot work at all; backends call it in open(), before anything was
        said, so that the dialog is handed to the station's own pipeline instead."""
        if self.kind not in KINDS:
            raise BackendUnavailable(f"unknown agent {self.kind!r} (LIVE_AGENT: {' | '.join(KINDS)})")
        if self.kind == "ha" and not (HA_URL and HA_TOKEN):
            raise BackendUnavailable("the agent is reached through Home Assistant, but HA_URL / HA_TOKEN are not set")

    def describe(self) -> str:
        if self.kind == "ha":
            return f"ha {self.agent_id or 'default agent'}" + (f" as {self.satellite_id or self.device_id}"
                                                               if self.satellite_id or self.device_id else "")
        return "hermes (direct)"

    async def ask(self, request: str) -> str:
        """The agent's answer to a self-contained request; raises when there is none."""
        t0 = time.monotonic()
        text = f"{MARK} {request}" if MARK else request
        answer = await (self._ask_ha(text) if self.kind == "ha" else self._ask_hermes(text))
        answer = TAG_RE.sub("", answer).strip()
        log.info("agent %s %.1fs: %r -> %r", self.kind, time.monotonic() - t0, request[:120], answer[:160])
        return answer or "Сделано, ответ пустой."

    async def _post(self, url: str, body: dict, headers: dict):
        async with self.http.post(url, json=body, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            raw = await r.text()
            status = r.status
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw
        return status, data

    async def _ask_ha(self, text: str) -> str:
        body = {"text": text, "language": HA_LANGUAGE}
        if self.agent_id:
            body["agent_id"] = self.agent_id
        if self.device_id:
            body["device_id"] = self.device_id
        if self.satellite_id:
            body["satellite_id"] = self.satellite_id
        # Requests running side by side (tool calls of one round, or one that outlived a barge-in) each get a
        # conversation of their own: an agent that keeps per-conversation history would mix them up otherwise.
        alone = self.in_flight == 0
        if alone and HA_KEEP_CONVERSATION and self.conversation_id:
            body["conversation_id"] = self.conversation_id
        self.in_flight += 1
        try:
            status, data = await self._post(f"{HA_URL}/api/conversation/process", body,
                                            {"Authorization": f"Bearer {HA_TOKEN}"})
        finally:
            self.in_flight -= 1
        if status != 200 or not isinstance(data, dict):
            raise RuntimeError(f"home assistant HTTP {status}: {str(data)[:200]}")
        if alone and HA_KEEP_CONVERSATION and data.get("conversation_id"):
            self.conversation_id = data["conversation_id"]
        resp = data.get("response") or {}
        speech = str(((resp.get("speech") or {}).get("plain") or {}).get("speech") or "")
        if resp.get("response_type") == "error":
            code = (resp.get("data") or {}).get("code") or "error"
            raise RuntimeError(f"home assistant: {code}: {speech}"[:300])
        return speech

    async def _ask_hermes(self, text: str) -> str:
        body = {"model": "hermes", "stream": False, "messages": [{"role": "user", "content": text}]}
        headers = {"Authorization": f"Bearer {HERMES_KEY}"} if HERMES_KEY else {}
        status, data = await self._post(HERMES_URL, body, headers)
        if status != 200 or not isinstance(data, dict):
            raise RuntimeError(f"hermes HTTP {status}: {str(data)[:200]}")
        return data["choices"][0]["message"].get("content") or ""
