"""Pipeline backend: OpenRouter (the chat model hears the speech, its reply is cut into sentences and spoken by a
TTS model) with the home agent behind the ask_hermes tool (through Home Assistant by default, see agent.py).

The VAD in Session does the turn-taking: it finds the end of an utterance, which goes to the chat model as audio;
the reply text is cut into sentences and each sentence is synthesised and streamed to the station while the model
keeps writing. Speech over the answer cancels all of it (barge-in); tool calls keep running through a barge-in and
their results are spoken later. The dialog ends when the model calls end_dialog, after IDLE_S of silence once the
answer has played, or when the station ends it.

Options (routes.json): llm, tts_voice, instructions (a file); the agent's: agent, ha_agent_id, ha_device_id,
ha_satellite_id (agent.py).
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

import aiohttp

from ..agent import Agent
from ..session import Session
from ..util import BackendUnavailable, HERE, clean_for_speech, read_instructions, wav16k
from ..wire import Station, frame

log = logging.getLogger("live")

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OR_PROXY = os.environ.get("LIVE_OR_PROXY") or None  # OpenRouter refuses some countries' IPs
REASONING = os.environ.get("LIVE_REASONING", "minimal")  # chat model reasoning effort, "" = default
LLM = os.environ.get("LIVE_LLM", "google/gemini-3.6-flash")  # 3.8-flash: ~3 s to first token, 3.6: ~1.5 s, same tool use
TTS = os.environ.get("LIVE_TTS", "elevenlabs/eleven-turbo-v2.5")  # ~0.45 s to first audio; google/gemini-3.8-flash-tts takes 1.6-4.5 s
TTS_VOICE = os.environ.get("LIVE_TTS_VOICE", "matilda")  # gemini: Kore
TTS_SPEED = float(os.environ.get("LIVE_TTS_SPEED", "1.1"))  # ElevenLabs accepts 0.7-1.2 through OpenRouter; use 1.0 with gemini
TTS_STYLE = os.environ.get("LIVE_TTS_STYLE", (
    "Говори по-русски живо и естественно, как человек с настроением: быстрый разговорный темп, лёгкая "
    "дерзость и ирония, без дикторской интонации. Читай только сам текст, ничего не добавляй."))
DIRECT_MAX = int(os.environ.get("LIVE_DIRECT_MAX", "260"))  # Hermes answers up to this many chars are spoken as they are (0 = always reword)
LLM_TIMEOUT = float(os.environ.get("LIVE_LLM_TIMEOUT", "60"))
# OpenRouter load-balances by price; for Gemini that includes the "flex" tiers (median 7-11 s to first token, p99 85 s
# measured 2026-10-26). Pin the priority endpoints (median ~1.6 s) and never fall back to flex. LIVE_LLM_PROVIDER: JSON, "" = off.
_PROV = os.environ.get("LIVE_LLM_PROVIDER")
LLM_PROVIDER = (json.loads(_PROV) if _PROV else None) if _PROV is not None else (
    {"order": ["google-ai-studio/priority", "google-vertex/global/priority"], "allow_fallbacks": False}
    if LLM.startswith("google/") else None)
LLM_HEDGE_S = float(os.environ.get("LIVE_LLM_HEDGE_S", "3.0"))  # nothing useful from the model after this long: race a 2nd identical request (0 = off)

SENT_RE = re.compile(r"(.+?[.!?…]+[\"»)]?)(?:\s+|$)", re.S)

TOOLS = [
    {"type": "function", "function": {
        "name": "ask_hermes",
        "description": (
            "Ask the home agent (Hermes) to do or look up something. Use it for EVERYTHING that touches the "
            "real world or personal data: smart home control and device states (lights, switches, climate, "
            "vacuum, media), timers and reminders, calendar, money and spending, mail, weather, web "
            "search, and the owner's remembered facts. Never claim an action happened unless this tool's answer "
            "says so."),
        "parameters": {"type": "object", "properties": {"request": {
            "type": "string",
            "description": "The full request in Russian, self-contained (resolve 'it', 'there', dates)."}},
            "required": ["request"]}}},
    {"type": "function", "function": {
        "name": "end_dialog",
        "description": "End the conversation: call it when the user says goodbye, thanks you and is done, or "
                       "tells you to stop listening. Say a very short goodbye first.",
        "parameters": {"type": "object", "properties": {}}}},
]


def tts_body(text: str, voice: str = "") -> dict:
    body = {"model": TTS, "input": text, "voice": voice or TTS_VOICE, "response_format": "pcm"}
    if TTS_SPEED != 1.0:
        body["speed"] = TTS_SPEED
    if TTS.startswith("google/"):
        body["instructions"] = TTS_STYLE  # only Gemini takes a style prompt
    return body


def or_headers() -> dict:
    return {"Authorization": f"Bearer {OR_KEY}", "HTTP-Referer": "https://home.local", "X-Title": "alena-live"}


class Dialog(Session):
    def __init__(self, station: Station, http: aiohttp.ClientSession, options: dict | None = None):
        super().__init__(station)
        o = options or {}
        self.http = http
        self.agent = Agent(http, o)
        self.llm = o.get("llm") or LLM
        self.voice = o.get("tts_voice") or TTS_VOICE
        self.instructions_file = o.get("instructions")
        self.instructions = ""
        self.history = []
        self.resp = None
        self.tts_task = None
        self.llm_provider = ""
        self._warm = None
        self.tools = {}        # call id -> task: tool calls outlive a barge-in, the work is not thrown away
        self.orphans = set()   # call ids whose turn was cut short before their result reached the history
        self.placeheld = set() # ... that were given a "still running" placeholder; the result comes as a note
        self.notes = []        # late results waiting for the next model round
        self.carry = False     # the next answer has to cover results the owner has not heard yet

    async def open(self):
        if not OR_KEY:
            raise BackendUnavailable("OPENROUTER_API_KEY is not set")
        self.agent.check()
        if self.instructions_file:
            p = Path(self.instructions_file)
            self.instructions = (p if p.is_absolute() else HERE / p).read_text(encoding="utf-8")
        else:
            self.instructions = read_instructions()
        log.info("dialog from %s: %s (llm %s, tts %s/%s, agent %s)", self.st.peer, self.st.hello, self.llm, TTS,
                 self.voice, self.agent.describe())
        self._warm = asyncio.create_task(self.prewarm())

    async def prewarm(self):
        """Open the connection to OpenRouter (through the proxy) while the owner is still speaking."""
        try:
            async with self.http.get(f"{OR_BASE}/key", headers=or_headers(), proxy=OR_PROXY,
                                     timeout=aiohttp.ClientTimeout(total=8)) as r:
                await r.read()
        except Exception:
            pass

    def working(self) -> bool:
        return any(not t.done() for t in self.tools.values())

    def followup_due(self) -> bool:
        return self.has_unspoken_results()

    def on_utterance(self, pcm: bytes):
        self.start_response(pcm)

    # ---- reply ----
    # ---- tool calls that survive an interruption ----
    def answered_ids(self) -> set:
        return {m.get("tool_call_id") for m in self.history if m.get("role") == "tool"}

    def insert_tool(self, cid: str, content: str):
        """Put a tool result right after the assistant message that made the call (before later user turns)."""
        for i, m in enumerate(self.history):
            if any(tc["id"] == cid for tc in m.get("tool_calls") or []):
                j = i + 1
                while j < len(self.history) and self.history[j].get("role") == "tool":
                    j += 1
                self.history.insert(j, {"role": "tool", "tool_call_id": cid, "content": content})
                return

    @staticmethod
    def tool_text(t: asyncio.Task) -> str:
        if t.cancelled():
            return "Отменено."
        e = t.exception()
        return f"Агент не ответил: {e}"[:300] if e else t.result()

    def orphan_pending(self):
        """The turn is cut short (barge-in). Its tools keep running; remember which results nobody recorded."""
        answered = self.answered_ids()
        self.orphans |= {cid for cid in self.tools if cid not in answered}

    def has_unspoken_results(self) -> bool:
        return bool(self.notes) or any(cid not in self.placeheld and self.tools.get(cid) and self.tools[cid].done()
                                       for cid in self.orphans)

    def prepare_history(self):
        """Before every model round: results of interrupted requests go where they belong, still-running ones get a
        placeholder, and late results arrive as a note. The model then answers for everything at once."""
        answered = self.answered_ids()
        for cid in sorted(self.orphans):
            t = self.tools.get(cid)
            if cid in answered or t is None:
                self.orphans.discard(cid)
            elif t.done():
                self.insert_tool(cid, self.tool_text(t))
                self.orphans.discard(cid)
                self.tools.pop(cid, None)
                self.carry = True
            elif cid not in self.placeheld:
                self.insert_tool(cid, "Агент ещё выполняет эту просьбу, результат придёт отдельно. Не повторяй её.")
                self.placeheld.add(cid)
        if self.notes:
            self.history.append({"role": "user", "content": "\n".join(self.notes)})
            self.notes.clear()
            self.carry = True

    def tool_done(self, c, t: asyncio.Task):
        cid = c["id"]
        if cid in self.orphans and cid in self.placeheld:
            try:
                req = json.loads(c["args"] or "{}").get("request", "")
            except ValueError:
                req = ""
            self.notes.append(f"(Агент закончил прежнюю просьбу «{req}»: {self.tool_text(t)} Расскажи хозяину.)")
            self.placeheld.discard(cid)
            self.orphans.discard(cid)
            self.tools.pop(cid, None)

    def start_followup(self):
        log.info("speaking the result of an interrupted request")
        self.turn_open = True
        self.set_state("thinking")
        self.resp = asyncio.create_task(self.respond())

    def cancel_response(self, flush: bool):
        if self.resp and not self.resp.done():
            self.orphan_pending()
        for t in (self.resp, self.tts_task):
            if t and not t.done():
                t.cancel()
        if flush:
            self.send(frame(b"F"))
        self.speaking = False
        self.turn_open = False

    def start_response(self, pcm: bytes):
        self.cancel_response(flush=False)
        for i in [i for i, m in enumerate(self.history) if isinstance(m.get("content"), list)][:-2]:
            self.history[i] = {"role": "user", "content": "(реплика хозяина)"}  # only the last turns keep their sound
        self.history.append({"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(wav16k(pcm)).decode(), "format": "wav"}}]})
        del self.history[:-14]
        while self.history and self.history[0].get("role") != "user":
            self.history.pop(0)  # never start with a tool result whose call was trimmed away
        self.turn_open = True
        self.set_state("thinking")
        self.resp = asyncio.create_task(self.respond())

    async def respond(self):
        q: asyncio.Queue = asyncio.Queue()
        self.tts_task = asyncio.create_task(self.tts_worker(q))
        try:
            for _round in range(4):
                calls = await self.llm_round(q)
                if not calls:
                    break
                self.shrink_audio(calls)
                tasks = []
                for c in calls:
                    t = asyncio.create_task(self.run_call(c))
                    self.tools[c["id"]] = t
                    t.add_done_callback(lambda t, c=c: self.tool_done(c, t))
                    tasks.append(t)
                # shield: a barge-in cancels this turn (speech, model stream) but not the tools themselves
                results = await asyncio.gather(*[asyncio.shield(t) for t in tasks])
                for c, r in zip(calls, results):
                    self.history.append({"role": "tool", "tool_call_id": c["id"], "content": r})
                    self.tools.pop(c["id"], None)
                direct = self.direct_answer(calls, results)
                if direct:
                    log.info("direct answer from the agent (%d chars), no second model round", len(direct))
                    self.history.append({"role": "assistant", "content": direct})
                    await self.flush_sentences(direct, q, final=True)
                    break
            await q.put(None)
            await self.tts_task
            self.carry = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("response failed: %r", e)
            self.tts_task.cancel()
        finally:
            self.turn_open = False
            self.touch()
            if not self.speaking and not self.ending:
                self.set_state("listening")

    def direct_answer(self, calls, results) -> str:
        """A short Hermes answer is already in the right words (its voice profile carries the persona):
        speak it as it is and skip the second model round (~2 s)."""
        if (not DIRECT_MAX or self.carry or self.ending or len(calls) != 1 or calls[0]["name"] != "ask_hermes"
                or self.has_unspoken_results()):  # an earlier interrupted request is ready: let the model say both
            return ""
        text = clean_for_speech(results[0])
        if not text or len(text) > DIRECT_MAX or text.startswith(("Агент не ответил", "Сделано, ответ пустой")):
            return ""
        return text

    def shrink_audio(self, calls):
        """The model has heard the question and put it into the tool request: keep that instead of the sound."""
        req = next((json.loads(c["args"] or "{}").get("request", "") for c in calls if c["name"] == "ask_hermes"), "")
        for i in range(len(self.history) - 1, -1, -1):
            m = self.history[i]
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                self.history[i] = {"role": "user", "content": "(голосом) " + req if req else "(реплика хозяина)"}
                break

    async def llm_events(self, body: dict, headers: dict):
        """Stream the model's SSE events. Rare provider stalls (12 s to a first token were seen) are hedged: if nothing
        useful (text / tool call / error) arrives within LLM_HEDGE_S, a second identical request is raced and the
        first one to produce something wins; the loser is cancelled."""
        q: asyncio.Queue = asyncio.Queue()

        async def one(idx: int, prov):
            req = dict(body, provider=prov) if prov else body
            try:
                async with self.http.post(f"{OR_BASE}/chat/completions", json=req, headers=headers, proxy=OR_PROXY,
                                          timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT, sock_read=30)) as r:
                    if r.status != 200:
                        await q.put((idx, "err", f"openrouter chat HTTP {r.status}: {(await r.text())[:300]}"))
                        return
                    async for raw in r.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            await q.put((idx, "ev", json.loads(data)))
                        except ValueError:
                            continue
                await q.put((idx, "end", None))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await q.put((idx, "err", repr(e)))

        def useful(ev: dict) -> bool:
            return bool(ev.get("error")) or any(
                (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("tool_calls")
                for ch in ev.get("choices") or [])

        tasks = {0: asyncio.create_task(one(0, LLM_PROVIDER))}
        t0, winner, hedged = time.monotonic(), None, False
        try:
            while True:
                wait = None
                if winner is None and not hedged and LLM_HEDGE_S > 0:
                    wait = max(0.01, LLM_HEDGE_S - (time.monotonic() - t0))
                try:
                    idx, kind, val = await asyncio.wait_for(q.get(), wait)
                except asyncio.TimeoutError:
                    hedged = True
                    log.info("llm silent for %.1fs, racing a second request", LLM_HEDGE_S)
                    other = dict(LLM_PROVIDER, order=LLM_PROVIDER["order"][::-1]) if LLM_PROVIDER else None
                    tasks[1] = asyncio.create_task(one(1, other))  # the other provider first
                    continue
                if winner is not None and idx != winner:
                    continue
                if kind == "err":
                    tasks.pop(idx, None)
                    if winner is None:
                        if tasks:
                            continue  # the other request may still deliver
                        if not hedged and LLM_HEDGE_S > 0:
                            hedged = True
                            log.warning("llm request failed (%s), retrying once without provider pinning", str(val)[:120])
                            tasks[1] = asyncio.create_task(one(1, None))
                            continue
                    raise RuntimeError(val)
                if kind == "end":
                    return
                if winner is None:
                    if not useful(val):
                        continue
                    winner = idx
                    self.llm_provider = str(val.get("provider") or "")
                    for i, t in tasks.items():
                        if i != idx:
                            t.cancel()
                    if hedged:
                        log.info("llm: %s request won", "second" if idx else "first")
                yield val
        finally:
            for t in tasks.values():
                t.cancel()

    async def llm_round(self, q: asyncio.Queue):
        self.prepare_history()
        body = {"model": self.llm, "stream": True, "tools": TOOLS,
                "messages": [{"role": "system", "content": self.instructions}] + self.history}  # provider pinning: llm_events
        if REASONING and "lite" not in self.llm:
            body["reasoning"] = {"effort": REASONING}
        headers = or_headers()
        t0 = time.monotonic()
        text, pending, calls, first, tool_seen = "", "", {}, True, False
        async for ev in self.llm_events(body, headers):
            if ev.get("error"):
                raise RuntimeError(f"openrouter: {str(ev['error'])[:300]}")
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    if first:
                        first = False
                        log.info("first text after %.2fs", time.monotonic() - t0)
                    text += d["content"]
                    pending += d["content"]
                    pending = await self.flush_sentences(pending, q, final=False)
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] = fn.get("name") or c["name"]
                    c["args"] += fn.get("arguments") or ""
                    if c["name"] and not tool_seen:
                        tool_seen = True
                        log.info("tool call %s after %.2fs", c["name"], time.monotonic() - t0)
        await self.flush_sentences(pending, q, final=True)
        calls_l = [dict(c, id=c["id"] or f"call_{i}") for i, c in sorted(calls.items())]
        msg = {"role": "assistant", "content": clean_for_speech(text) or None}
        if calls_l:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for c in calls_l]
        if msg["content"] or calls_l:
            self.history.append(msg)
        log.info("llm %.1fs [%s]: %r tools=%s", time.monotonic() - t0, self.llm_provider,
                 msg["content"] and msg["content"][:200], [c["name"] for c in calls_l])
        return calls_l

    async def flush_sentences(self, pending: str, q: asyncio.Queue, final: bool) -> str:
        while True:
            m = SENT_RE.match(pending)
            if m and (final or m.end() < len(pending) or pending.rstrip().endswith(("!", "?", "…"))
                      or len(pending) > 140):
                s = clean_for_speech(m.group(1))
                pending = pending[m.end():]
                if re.search(r"\w", s):
                    await q.put(s)
                continue
            break
        if final and re.search(r"\w", clean_for_speech(pending)):
            await q.put(clean_for_speech(pending))
            pending = ""
        return pending

    async def tts_worker(self, q: asyncio.Queue):
        headers = or_headers()
        while True:
            s = await q.get()
            if s is None:
                return
            t0 = time.monotonic()
            body = tts_body(s, self.voice)
            first, carry = True, b""
            async with self.http.post(f"{OR_BASE}/audio/speech", json=body, headers=headers, proxy=OR_PROXY,
                                      timeout=aiohttp.ClientTimeout(total=60, sock_read=30)) as r:
                if r.status != 200:
                    log.warning("tts HTTP %s: %s", r.status, (await r.text())[:200])
                    continue
                async for chunk in r.content.iter_chunked(4800):
                    pcm = carry + chunk
                    carry = pcm[len(pcm) // 2 * 2:]
                    pcm = pcm[:len(pcm) // 2 * 2]
                    if not pcm:
                        continue
                    if first:
                        first = False
                        log.info("tts first audio %.2fs: %r", time.monotonic() - t0, s[:80])
                    self.speaking = True
                    self.touch()
                    self.set_state("speaking")
                    self.send(frame(b"A", pcm))
                    await self.st.drain()

    async def run_call(self, c) -> str:
        try:
            args = json.loads(c["args"] or "{}")
        except ValueError:
            args = {}
        log.info("tool %s %s", c["name"], args)
        if c["name"] == "end_dialog":
            self.ending = "goodbye"
            return "ok"
        if c["name"] == "ask_hermes":
            if not self.speaking:
                self.set_state("thinking")
            try:
                return await self.agent.ask(str(args.get("request", "")))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("agent failed: %r", e)
                return f"Агент не ответил: {e}"[:300]
        return f"unknown tool {c['name']}"


async def run(station: Station, http: aiohttp.ClientSession, options: dict):
    await Dialog(station, http, options).run()
