"""The home agent: what a dialog model calls for anything that touches the real world (the pipeline's ask_hermes
tool, gpt-live's delegation). Two ways to reach it (LIVE_AGENT, or the "agent" option of a route):

  hermes  (default) the agent's own OpenAI-compatible endpoint (HERMES_URL), past Home Assistant: the agent reaches
          the house itself (e.g. through Home Assistant's MCP server).
  ha      Home Assistant's conversation API, POST /api/conversation/process, with the conversation agent of
          the voice pipeline (HA_AGENT_ID, e.g. conversation.extended_openai_conversation in front of Hermes). HA stays
          the one door to the agent: the request lands in HA's chat log, the agent learns which station asked (the
          route options ha_device_id / ha_satellite_id: area context), and changing the agent in HA changes it for
          Live dialogs too. One Live dialog is one HA conversation (its conversation_id is kept between requests).

There is no fallback from one to the other: a request the chosen way cannot answer is reported to the model as failed.
Either way the request carries HERMES_MARK, the prefix a model router in front of the agent recognises Live requests by.

Hermes is asked with stream=true, each request in a session of its own that the caller names (X-Hermes-Session-Id,
needs HERMES_KEY; without one Hermes names it from the request's text): the agent's questions to the owner and its
late answers are routed by it (surface.py). Closing the request interrupts the agent's run there (a request nobody
waits for any more does not go on acting).
"""
import json
import logging
import os
import time

import aiohttp

from .util import TAG_RE, BackendUnavailable

log = logging.getLogger("live")

AGENT = os.environ.get("LIVE_AGENT", "hermes")
TIMEOUT = float(os.environ.get("LIVE_AGENT_TIMEOUT") or os.environ.get("HERMES_TIMEOUT") or "1800")  # background tasks included
SILENT_S = float(os.environ.get("LIVE_AGENT_SILENT_S", "900"))  # nothing at all from the agent this long (it may wait for the owner)
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

    def names_sessions(self) -> bool:
        """Can the caller name the agent session of a request (ask(session=...))?"""
        return self.kind == "hermes" and bool(HERMES_KEY)

    async def ask(self, request: str, on_session=None, session: str = "") -> str:
        """The agent's answer to a self-contained request; raises when there is none. `session`: the agent session
        to run it in (names_sessions()); `on_session(id)`: called with the agent's session id as soon as it is known
        (hermes only)."""
        t0 = time.monotonic()
        text = f"{MARK} {request}" if MARK else request
        answer = await (self._ask_ha(text) if self.kind == "ha" else self._ask_hermes(text, on_session, session))
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

    async def _ask_hermes(self, text: str, on_session=None, session: str = "") -> str:
        body = {"model": "hermes", "stream": True, "messages": [{"role": "user", "content": text}]}
        headers = {"Authorization": f"Bearer {HERMES_KEY}"} if HERMES_KEY else {}
        if session and HERMES_KEY:
            headers["X-Hermes-Session-Id"] = session
        async with self.http.post(HERMES_URL, json=body, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=TIMEOUT, sock_read=SILENT_S)) as r:
            if r.status != 200:
                raise RuntimeError(f"hermes HTTP {r.status}: {(await r.text())[:200]}")
            sid = r.headers.get("X-Hermes-Session-Id", "")
            if sid and on_session is not None:
                on_session(sid)
            if "event-stream" not in r.headers.get("Content-Type", ""):  # a server that answers in one piece
                data = json.loads(await r.text())
                return data["choices"][0]["message"].get("content") or ""
            # The stream carries every assistant message of the turn; the answer is the one after the last tool
            # (what the agent says between tool calls, "Now Tomsk.", is narration). An empty last message falls
            # back to the last one with text.
            answer, said, event = "", "", ""
            async for raw in r.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    event = ""
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if event == "hermes.tool.progress":
                    said, answer = (answer if answer.strip() else said), ""
                    continue
                if event and event != "message":
                    continue
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise RuntimeError(f"hermes: {str(ev['error'])[:200]}")
                for ch in ev.get("choices") or []:
                    answer += (ch.get("delta") or {}).get("content") or ""
            return answer if answer.strip() else said
