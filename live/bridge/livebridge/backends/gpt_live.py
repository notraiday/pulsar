"""GPT-Live backend (OpenAI gpt-live-1, full-duplex speech-to-speech) -- EXPERIMENTAL.

Written from OpenAI's public GPT-Live documentation (WebSocket protocol, client delegation) and tested only against
a fake server that follows those documents; it has never talked to the real service. Expect to adjust event names,
headers and the barge-in heuristic below once it does.

The model listens to the mic stream and speaks on its own (turn-taking, interruption, voice). What stays in the
bridge:
  * audio: station mic s16le 16 kHz -> 24 kHz -> session.input_audio.append; session.output_audio.delta -> station;
  * barge-in: the model streams audio faster than real time and sends no "interrupted" signal, so the station's
    queue is cut by the same Silero VAD the pipeline backend uses (frame F), and the audio the model had already
    produced is dropped until it pauses (DROP_GAP_S) -- the next response;
  * the home agent: delegation is "client": on session.delegation.created the bridge asks Hermes with what the owner
    just said (from the input transcripts) and hands the answer back as session.commentary.append.
The dialog ends when the station or the model closes it, or after the idle time (there is no end_dialog tool).

Options (routes.json): voice, model, instructions (a file).
Environment: OPENAI_API_KEY, LIVE_GPT_URL, LIVE_GPT_MODEL, LIVE_GPT_VOICE, LIVE_GPT_PROXY (else LIVE_OR_PROXY),
LIVE_GPT_HEADERS (JSON, extra connection headers), LIVE_GPT_DROP_GAP_S.
"""
import asyncio
import audioop
import base64
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

from ..hermes import ask_hermes
from ..session import Session
from ..util import BackendUnavailable, HERE, read_instructions
from ..wire import Station, frame

log = logging.getLogger("live")

URL = os.environ.get("LIVE_GPT_URL", "wss://api.openai.com/v1/live/sessions")
MODEL = os.environ.get("LIVE_GPT_MODEL", "gpt-live-1")
VOICE = os.environ.get("LIVE_GPT_VOICE", "marin")
PROXY = os.environ.get("LIVE_GPT_PROXY") or os.environ.get("LIVE_OR_PROXY") or None
DROP_GAP_S = float(os.environ.get("LIVE_GPT_DROP_GAP_S", "0.35"))
START_TIMEOUT_S = 10.0
APPEND_CHARS = 1200   # the service takes at most 500 tokens per append
TURN_GAP_S = 1.0      # a pause this long in the input transcript starts a new utterance
KEEP_TURNS = 6


def headers() -> dict:
    h = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    extra = os.environ.get("LIVE_GPT_HEADERS")
    if extra:
        h.update(json.loads(extra))
    return h


class GptLive(Session):
    def __init__(self, station: Station, http: aiohttp.ClientSession, options: dict | None = None):
        super().__init__(station)
        o = options or {}
        self.http = http
        self.model = o.get("model") or MODEL
        self.voice = o.get("voice") or VOICE
        self.instructions_file = o.get("instructions")
        self.ws = None
        self.rate_state = None     # audioop.ratecv state: 16 kHz mic -> 24 kHz
        self.drop = False          # after a barge-in: audio the model had already produced is not played
        self.last_delta = 0.0
        self.turns: list[str] = []  # what the owner said, from the input transcripts
        self.last_heard = 0.0
        self.said = ""             # the model's words of the current response (log only)
        self.tools: dict[str, asyncio.Task] = {}  # delegation id -> the agent request
        self.n = 0

    # ---- session ----
    async def open(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise BackendUnavailable("OPENAI_API_KEY is not set")
        if self.instructions_file:
            p = Path(self.instructions_file)
            text = (p if p.is_absolute() else HERE / p).read_text(encoding="utf-8")
        else:
            text = read_instructions("LIVE_GPT_INSTRUCTIONS", "instructions.gpt-live")
        try:
            async with asyncio.timeout(START_TIMEOUT_S):
                self.ws = await self.http.ws_connect(URL, headers=headers(), proxy=PROXY, heartbeat=20)
                await self.ws.send_json({"type": "session.start", "event_id": "start", "session": {
                    "model": self.model, "instructions": text,
                    "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": self.voice}},
                    "delegation": {"type": "client"}}})
                while True:
                    msg = await self.ws.receive()
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        raise RuntimeError(f"connection closed before session.started ({msg.type.name})")
                    ev = json.loads(msg.data)
                    if ev.get("type") == "session.started":
                        break
                    if ev.get("type") == "error":
                        raise RuntimeError(f"session rejected: {str(ev)[:300]}")
        except BackendUnavailable:
            raise
        except Exception as e:
            await self.close()
            raise BackendUnavailable(f"gpt-live: {e!r}") from e
        log.info("dialog from %s: %s (gpt-live %s, voice %s)", self.st.peer, self.st.hello, self.model, self.voice)

    async def close(self):
        for t in self.tools.values():
            t.cancel()
        if self.ws is not None and not self.ws.closed:
            try:
                await asyncio.wait_for(self.ws.send_json({"type": "session.close"}), 1)
            except Exception:
                pass
            await self.ws.close()

    def tasks(self):
        return [self.downlink()]

    # ---- station -> model ----
    async def on_mic(self, pcm: bytes):
        self.feed(pcm)  # the VAD only decides barge-in here; the model hears the audio itself
        pcm24, self.rate_state = audioop.ratecv(pcm, 2, 1, 16000, 24000, self.rate_state)
        await self.ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(pcm24).decode()})

    def cancel_response(self, flush: bool):
        super().cancel_response(flush)
        if flush:
            self.drop = True
            self.last_delta = time.monotonic()

    # ---- model -> station ----
    async def downlink(self):
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                ev = json.loads(msg.data)
            except ValueError:
                continue
            if self.on_event(ev):
                break
        self.ending = self.ending or "model closed the session"

    def on_event(self, ev: dict) -> bool:
        """True: the session is over."""
        t = ev.get("type", "")
        if t == "session.output_audio.delta":
            self.play(base64.b64decode(ev.get("delta") or ""))
        elif t == "session.input_transcript.delta":
            self.hear(str(ev.get("delta") or ""))
        elif t == "session.output_transcript.delta":
            self.said += str(ev.get("delta") or "")
        elif t == "session.delegation.created":
            self.delegate((ev.get("delegation") or {}).get("id") or "")
        elif t == "error":
            log.warning("gpt-live error: %s", str(ev)[:300])
        elif t == "session.closed":
            return True
        return False

    def play(self, pcm: bytes):
        if not pcm:
            return
        now = time.monotonic()
        if self.drop:
            if now - self.last_delta < DROP_GAP_S:   # still the response the owner interrupted
                self.last_delta = now
                return
            self.drop = False                         # the model paused: this is the next one
        self.last_delta = now
        if self.said:
            log.info("model said: %r", self.said[:200])
            self.said = ""
        self.speaking = True
        self.touch()
        self.set_state("speaking")
        self.send(frame(b"A", pcm[:len(pcm) // 2 * 2]))

    def hear(self, text: str):
        now = time.monotonic()
        if not self.turns or now - self.last_heard > TURN_GAP_S:
            self.turns.append("")
            del self.turns[:-KEEP_TURNS]
        self.turns[-1] += text
        self.last_heard = now
        self.touch()

    # ---- the home agent ----
    def working(self) -> bool:
        return any(not t.done() for t in self.tools.values())

    def delegate(self, did: str):
        request = " ".join(t.strip() for t in self.turns[-2:] if t.strip())
        log.info("delegation %s: %r", did, request[:200])
        if not self.speaking:
            self.set_state("thinking")
        self.tools[did] = asyncio.create_task(self.run_delegation(did, request))

    async def run_delegation(self, did: str, request: str):
        try:
            text = await ask_hermes(self.http, request or "(просьба не расслышана)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("hermes failed: %r", e)
            text = f"Агент не ответил: {e}"
        self.n += 1
        self.touch()
        if not self.ws.closed:
            await self.ws.send_json({"type": "session.commentary.append", "event_id": f"result_{self.n}",
                                     "delegation_id": did, "content": text[:APPEND_CHARS]})


async def run(station: Station, http: aiohttp.ClientSession, options: dict):
    await GptLive(station, http, options).run()
