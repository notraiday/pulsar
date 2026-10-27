"""The speakers as the agent's way to the owner, for what it needs while it works: a one-time code a site sent,
a yes / no before it pays, a short answer. The owner's reply goes back to the agent and to nothing else: no chat
model sees it, it is not put into any dialog's history and a code is not logged.

The agent side (a Hermes plugin) talks to it over HTTP (LIVE_SURFACE_LISTEN, bearer LIVE_SURFACE_TOKEN):

  POST /ask     {"session", "kind": "code"|"confirm"|"text", "text", "timeout"} -> {"id", "wake"}
                The question is spoken in the live dialog that started that agent session. When there is none
                ("wake": true), the caller wakes the speakers itself (Home Assistant's start_conversation with
                exactly this text): the dialogs that open with it take the question; the one that hears the
                owner loudest gets the answer (reminders.py).
                A request the dialog dropped (cancelled) gets no questions: status "no_answer" at once.
  GET  /ask/ID?wait=S                -> {"status": "answered"|"unclear"|"timeout"|"no_answer"|"pending", "answer"}
                "code": the digits; "confirm": "yes" | "no"; "text": the transcript.
  POST /result  {"session", "text"}  -> {"announce": bool}
                The agent is about to answer: true when a Live dialog started that session and nobody is listening
                for it any more (the dialog has ended), so the caller announces it. Sessions the bridge never saw
                (Home Assistant's own pipeline) are never announced.

Agent sessions are bound to dialogs by their X-Hermes-Session-Id: the dialog names each request's session (agent.py).
"""
import asyncio
import hmac
import itertools
import logging
import os
import time
import weakref

from aiohttp import web

from . import spoken
from .util import clean_for_speech

log = logging.getLogger("live")

LISTEN = os.environ.get("LIVE_SURFACE_LISTEN", "")      # e.g. 0.0.0.0:10702; "" = off
TOKEN = os.environ.get("LIVE_SURFACE_TOKEN", "")
ASK_TIMEOUT_S = float(os.environ.get("LIVE_ASK_TIMEOUT_S", "120"))
KINDS = ("code", "confirm", "text")

_ids = itertools.count(1)


class Ask:
    def __init__(self, session: str, kind: str, text: str, timeout: float):
        self.id = f"a{next(_ids)}"
        self.session, self.kind, self.text = session, kind, clean_for_speech(text)
        self.deadline = time.monotonic() + timeout
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.wake = False          # waits for a dialog the house opens with this text
        self.tries = 0             # "unclear" answers re-asked so far
        self.asked = False         # spoken (or being spoken) to the owner
        self.holders = weakref.WeakSet()  # dialogs that may take the answer

    @property
    def secret(self) -> bool:
        return self.kind == "code"

    def resolve(self, status: str, answer: str = ""):
        if not self.future.done():
            self.future.set_result({"status": status, "answer": answer})

    def parse(self, transcript: str) -> tuple[str, str]:
        """(status, answer) for what the owner said."""
        if self.kind == "code":
            d = spoken.digits(transcript)
            return ("answered", d) if len(d) >= 3 else ("unclear", "")
        if self.kind == "confirm":
            v = spoken.yes_no(transcript)
            return ("answered", v) if v != "unclear" else ("unclear", "")
        t = (transcript or "").strip()
        return ("answered", t) if t else ("unclear", "")


class Registry:
    def __init__(self):
        self.dialogs: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()  # session -> dialog
        self.detached: set[str] = set()
        self.known: set[str] = set()      # sessions a dialog started: only theirs are ever announced
        self.announced: set[str] = set()
        self.abandoned: set[str] = set()  # requests nobody waits for: their questions are not put to anyone
        self.asks: dict[str, Ask] = {}

    # ---- agent sessions ----
    def bind(self, session: str, dialog):
        if session:
            self.dialogs[session] = dialog
            self.known.add(session)
            self.announced.discard(session)
            self.abandoned.discard(session)

    def abandon(self, session: str, dialog):
        """The dialog stopped waiting for this session's request (and closed it, which interrupts the agent)."""
        if not session or self.dialogs.get(session) is not dialog:
            return
        del self.dialogs[session]
        self.abandoned.add(session)
        for a in self.asks.values():
            if a.session == session:
                a.resolve("no_answer")
        log.info("session %s abandoned: its request was dropped", session)

    def detach(self, session: str):
        if session:
            self.detached.add(session)

    def listening(self, session: str) -> bool:
        """Will the dialog that asked speak this session's answer itself?"""
        d = self.dialogs.get(session)
        return d is not None and not d.ending and not d.st.closing

    # ---- questions ----
    def ask(self, session: str, kind: str, text: str, timeout: float) -> Ask:
        a = Ask(session, kind, text, timeout)
        self.asks[a.id] = a
        d = self.dialogs.get(session)
        if session in self.abandoned:
            log.info("ask %s (%s) for %s: its request was dropped, not asked", a.id, kind, session)
            a.resolve("no_answer")
        elif d is not None and not d.ending and not d.st.closing and d.take_ask(a):
            log.info("ask %s (%s) in the live dialog of %s", a.id, kind, session)
        else:
            a.wake = True
            log.info("ask %s (%s) for %s: no live dialog, waiting for the speakers to wake", a.id, kind, session)
        asyncio.get_running_loop().call_later(timeout, a.resolve, "timeout")
        a.future.add_done_callback(lambda _f, i=a.id: self._done(i))
        return a

    def _done(self, ask_id: str):
        a = self.asks.get(ask_id)
        if a is not None:
            log.info("ask %s: %s", ask_id, a.future.result()["status"])
        # kept a little for a late GET; dropped by prune()

    def for_wake(self, text: str):
        """The open question a dialog the house just opened with `text` is for."""
        key = " ".join(clean_for_speech(text).lower().split())
        for a in self.asks.values():
            if a.wake and not a.future.done() and " ".join(a.text.lower().split()) == key:
                return a
        return None

    def prune(self):
        now = time.monotonic()
        for i in [i for i, a in self.asks.items() if a.future.done() and now - a.deadline > 300]:
            del self.asks[i]


registry = Registry()


# ---- HTTP side ----

def _authorized(request: web.Request) -> bool:
    if not TOKEN:
        return False
    got = request.headers.get("Authorization", "")
    return hmac.compare_digest(got, f"Bearer {TOKEN}")


@web.middleware
async def _auth(request, handler):
    if not _authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return await handler(request)


async def _post_ask(request: web.Request):
    body = await request.json()
    kind = str(body.get("kind") or "text")
    text = str(body.get("text") or "").strip()
    if kind not in KINDS or not text:
        return web.json_response({"error": f"kind must be one of {KINDS} and text non-empty"}, status=400)
    timeout = min(float(body.get("timeout") or ASK_TIMEOUT_S), 600.0)
    registry.prune()
    a = registry.ask(str(body.get("session") or ""), kind, text, timeout)
    return web.json_response({"id": a.id, "wake": a.wake, "text": a.text})


async def _get_ask(request: web.Request):
    a = registry.asks.get(request.match_info["id"])
    if a is None:
        return web.json_response({"error": "no such ask"}, status=404)
    wait = min(float(request.query.get("wait") or 0), 600.0)
    if wait > 0 and not a.future.done():
        try:
            await asyncio.wait_for(asyncio.shield(a.future), wait)
        except asyncio.TimeoutError:
            pass
    return web.json_response(a.future.result() if a.future.done() else {"status": "pending", "answer": ""})


async def _post_result(request: web.Request):
    body = await request.json()
    session = str(body.get("session") or "")
    # Only sessions a Live dialog started: the agent also answers Home Assistant's own pipeline, which speaks
    # its answers itself.
    announce = (session in registry.known and not registry.listening(session) and session not in registry.announced
                and session not in registry.abandoned)
    if announce:
        registry.announced.add(session)
    log.info("result of %s: %s", session or "?", "announce it" if announce else "the dialog speaks it")
    return web.json_response({"announce": announce})


def app() -> web.Application:
    a = web.Application(middlewares=[_auth])
    a.router.add_post("/ask", _post_ask)
    a.router.add_get("/ask/{id}", _get_ask)
    a.router.add_post("/result", _post_result)
    return a


async def serve(listen: str = "") -> web.AppRunner | None:
    listen = listen or LISTEN
    if not listen:
        return None
    if not TOKEN:
        log.warning("LIVE_SURFACE_LISTEN is set but LIVE_SURFACE_TOKEN is not: the owner surface stays off")
        return None
    host, port = listen.rsplit(":", 1)
    runner = web.AppRunner(app())
    await runner.setup()
    await web.TCPSite(runner, host, int(port)).start()
    log.info("owner surface on %s", listen)
    return runner
