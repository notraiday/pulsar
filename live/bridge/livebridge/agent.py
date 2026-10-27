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

With HERMES_KEY, every request is a run of Hermes's Runs API (POST {base}/runs, events from {base}/runs/ID/events,
{base} = HERMES_URL without /chat/completions) in a session the caller names: the agent's questions to the owner and
its late answers are routed by it (surface.py), a running request can be steered (POST .../steer: the text reaches the
agent after its current tool batch) and stopped (POST .../stop; cancelling ask() stops the run: a request nobody waits
for any more does not go on acting). The same session can be continued later with all its history. Without a key
(or HERMES_RUNS=0) the request is a streamed chat completion, in a session Hermes names after the request's text;
closing it interrupts the run.
"""
import asyncio
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
HERMES_RUNS = os.environ.get("HERMES_RUNS", "1") != "0"
TERMINAL = ("run.completed", "run.failed", "run.cancelled", "run.interrupted")

_stopping: set = set()  # stop requests in flight (kept referenced until done)


def runs_base() -> str:
    return HERMES_URL[: -len("/chat/completions")] if HERMES_URL.endswith("/chat/completions") else ""

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

    def runs(self) -> bool:
        """Requests are runs: they can be steered and stopped."""
        return self.names_sessions() and HERMES_RUNS and bool(runs_base())

    async def ask(self, request: str, on_session=None, session: str = "", on_progress=None, on_run=None,
                  on_said=None) -> str:
        """The agent's answer to a self-contained request; raises when there is none. `session`: the agent session
        to run it in (names_sessions(); an existing one is continued); `on_session(id)`: called with the agent's
        session id as soon as it is known; `on_progress(step)`: each tool the agent starts, as its own one-line
        preview; `on_run(id)`: the run's id, for steer(); `on_said(text)`: what the agent writes between its steps
        (hermes runs only)."""
        t0 = time.monotonic()
        text = f"{MARK} {request}" if MARK else request
        if self.kind == "ha":
            answer = await self._ask_ha(text)
        elif self.runs() and session:
            answer = await self._ask_run(text, session, on_session, on_progress, on_run, on_said)
        else:
            answer = await self._ask_hermes(text, on_session, session, on_progress)
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

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {HERMES_KEY}"} if HERMES_KEY else {}

    async def _ask_run(self, text: str, session: str, on_session, on_progress, on_run, on_said=None) -> str:
        base = runs_base()
        async with self.http.post(f"{base}/runs", json={"input": text, "session_id": session}, headers=self._headers(),
                                  timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status not in (200, 201, 202):
                raise RuntimeError(f"hermes HTTP {r.status}: {(await r.text())[:200]}")
            run_id = str((await r.json()).get("run_id") or "")
        if not run_id:
            raise RuntimeError("hermes: no run id")
        if on_session is not None:
            on_session(session)
        if on_run is not None:
            on_run(run_id)
        try:
            return await self._run_events(base, run_id, on_progress, on_said)
        except asyncio.CancelledError:
            t = asyncio.get_running_loop().create_task(self.stop_run(run_id))  # closing the events does not stop it
            _stopping.add(t)
            t.add_done_callback(_stopping.discard)
            raise

    async def _run_events(self, base: str, run_id: str, on_progress, on_said=None) -> str:
        """The run's answer (its last message) from its event stream; a dropped stream is resumed after the last
        event seen."""
        seq, drops = -1, 0
        while True:
            headers = self._headers()
            if seq >= 0:
                headers["Last-Event-ID"] = str(seq)
            try:
                async with self.http.get(f"{base}/runs/{run_id}/events", headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=TIMEOUT, sock_read=SILENT_S)) as r:
                    if r.status != 200:
                        raise RuntimeError(f"hermes run events HTTP {r.status}: {(await r.text())[:200]}")
                    async for raw in r.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except ValueError:
                            continue
                        if not isinstance(ev, dict):
                            continue
                        if isinstance(ev.get("seq"), int):
                            seq = max(seq, ev["seq"])
                        name = ev.get("event")
                        if name == "tool.started" and on_progress is not None:
                            on_progress(str(ev.get("preview") or ev.get("tool") or ""))
                        elif name == "message.interim" and on_said is not None:
                            on_said(TAG_RE.sub("", str(ev.get("text") or "")))
                        elif name == "run.completed":
                            return str(ev.get("output") or "")
                        elif name == "run.failed":
                            raise RuntimeError(f"hermes: {str(ev.get('error') or ev.get('message') or 'the run failed')[:200]}")
                        elif name in TERMINAL:  # stopped from elsewhere: what it had
                            return str(ev.get("output") or ev.get("partial") or "") or "Агент остановлен."
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError, aiohttp.ClientOSError) as e:
                log.info("hermes run %s: event stream dropped (%r), resuming after event %d", run_id, e, seq)
            drops += 1
            if drops > 5:
                raise RuntimeError("hermes: the run's event stream keeps dropping")
            await asyncio.sleep(1.0)

    async def steer(self, run_id: str, text: str) -> bool:
        """Give a running request more to go on: the agent gets `text` after its current tool batch."""
        try:
            async with self.http.post(f"{runs_base()}/runs/{run_id}/steer", json={"input": text},
                                      headers=self._headers(), timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200 and bool((await r.json()).get("accepted"))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            log.warning("hermes run %s: steer failed: %r", run_id, e)
            return False

    async def stop_run(self, run_id: str):
        try:
            async with self.http.post(f"{runs_base()}/runs/{run_id}/stop", json={}, headers=self._headers(),
                                      timeout=aiohttp.ClientTimeout(total=10)) as r:
                log.info("hermes run %s stopped (HTTP %d)", run_id, r.status)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("hermes run %s: stop failed: %r", run_id, e)

    async def _ask_hermes(self, text: str, on_session=None, session: str = "", on_progress=None) -> str:
        body = {"model": "hermes", "stream": True, "messages": [{"role": "user", "content": text}]}
        headers = self._headers()
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
                    if on_progress is not None:
                        try:
                            ev = json.loads(data)
                        except ValueError:
                            ev = {}
                        if isinstance(ev, dict) and ev.get("status", "running") == "running":
                            on_progress(str(ev.get("label") or ev.get("tool") or ""))
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
