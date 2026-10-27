"""The speakers as the agent's way to the owner, for what it needs while it works: a one-time code a site sent,
a yes / no before it pays, a short answer. The owner's reply goes back to the agent and to nothing else: no chat
model sees it, it is not put into any dialog's history and a code is not logged.

The agent side (a Hermes plugin) talks to it over HTTP (LIVE_SURFACE_LISTEN, bearer LIVE_SURFACE_TOKEN):

  POST /ask     {"session", "kind": "code"|"confirm"|"text", "text", "timeout"} -> {"id", "wake"}
                The question is spoken in the live dialog that started that agent session; when that one is gone
                (the request went on in the background), in the Live dialog the owner is in now. When there is
                none ("wake": true), the caller wakes the speakers itself (Home Assistant's start_conversation with
                exactly this text): the dialogs that open with it take the question; the one that hears the
                owner loudest gets the answer (reminders.py). A station in a Live dialog refuses
                start_conversation, so a dialog in progress always takes the question itself.
                A request the dialog dropped (cancelled) gets no questions: status "no_answer" at once.
  GET  /ask/ID?wait=S                -> {"status": "answered"|"unclear"|"timeout"|"no_answer"|"pending", "answer"}
                "code": the digits; "confirm": "yes" | "no"; "text": the transcript.
  POST /result  {"session", "text"}  -> {"announce": bool}
                The agent is about to answer. The dialog that asked speaks it while it lasts; once it has ended, a
                Live dialog in progress gets it (spoken there); with none, true: the caller announces it.
                Sessions the bridge never saw (Home Assistant's own pipeline) are never announced.
  GET  /session/ID                   -> {"background": bool}
                The request went on in the background: nobody waits for it, the agent may think longer.

Agent sessions are bound to dialogs by their X-Hermes-Session-Id: the dialog names each request's session (agent.py).
Requests running in the background are listed for every dialog (what was asked, the agent's last step), so the
owner can ask how one is going or stop it.
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
# Said before a background request's question in a dialog that did not start it.
FOREIGN_LEAD = "По фоновой задаче: "

_ids = itertools.count(1)


class Ask:
    def __init__(self, session: str, kind: str, text: str, timeout: float):
        self.id = f"a{next(_ids)}"
        self.session, self.kind, self.text = session, kind, clean_for_speech(text)
        self.deadline = time.monotonic() + timeout
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.wake = False          # waits for a dialog the house opens with this text
        self.lead = ""             # said before the text
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


class Background:
    """An agent request that went on after its dialog stopped waiting for it."""

    def __init__(self, session: str, request: str, task: asyncio.Task):
        self.session, self.request, self.task = session, request, task
        self.started = time.monotonic()
        self.step, self.step_at = "", 0.0

    def describe(self, now: float) -> str:
        out = f"«{self.request[:200]}», идёт {_ago(now - self.started)}"
        if self.step:
            out += f"; последний шаг агента {_ago(now - self.step_at)} назад: {self.step}"
        return out


def _ago(s: float) -> str:
    return f"{int(s // 60)} мин" if s >= 60 else f"{int(s)} с"


def _live(d) -> bool:
    return d is not None and not d.ending and not d.st.closing


class Registry:
    def __init__(self):
        self.dialogs: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()  # session -> dialog
        self.open: "weakref.WeakSet" = weakref.WeakSet()  # Live dialogs in progress
        self.detached: set[str] = set()
        self.background: dict[str, Background] = {}       # detached requests still running
        self.requests: dict[str, str] = {}                 # session -> what it was asked
        self.known: set[str] = set()      # sessions a dialog started: only theirs are ever announced
        self.announced: set[str] = set()
        self.abandoned: set[str] = set()  # requests nobody waits for: their questions are not put to anyone
        self.asks: dict[str, Ask] = {}

    # ---- dialogs and agent sessions ----
    def opened(self, dialog):
        self.open.add(dialog)

    def closed(self, dialog):
        self.open.discard(dialog)

    def current(self, but=None):
        """The Live dialog the owner is in now (the one active last), if any."""
        live = [d for d in self.open if d is not but and _live(d)]
        return max(live, key=lambda d: d.last_activity, default=None)

    def bind(self, session: str, dialog, request: str = ""):
        if session:
            self.dialogs[session] = dialog
            self.known.add(session)
            self.announced.discard(session)
            self.abandoned.discard(session)
            if request:
                self.requests[session] = request

    def abandon(self, session: str, dialog=None):
        """Nobody waits for this session's request any more (closing it interrupts the agent). With `dialog`: only
        while the session is still that dialog's."""
        if not session or (dialog is not None and self.dialogs.get(session) is not dialog):
            return
        self.dialogs.pop(session, None)
        self.abandoned.add(session)
        for a in self.asks.values():
            if a.session == session:
                a.resolve("no_answer")
        log.info("session %s abandoned: its request was dropped", session)

    def detach(self, session: str, task: asyncio.Task | None = None, request: str = ""):
        if not session:
            return
        self.detached.add(session)
        if request:  # the model's words for it (self-contained), not the transcript an early request started with
            self.requests[session] = request
        if task is not None and not task.done():
            self.background[session] = Background(session, self.requests.get(session, ""), task)
            task.add_done_callback(lambda _t, s=session: self.background.pop(s, None))

    def progress(self, session: str, step: str):
        b = self.background.get(session)
        if b is not None and step:
            b.step, b.step_at = step[:120], time.monotonic()

    def running(self) -> list[Background]:
        return sorted((b for b in self.background.values() if not b.task.done()), key=lambda b: b.started)

    def stop(self, session: str) -> bool:
        """The owner stopped a background request: closing it interrupts the agent's run."""
        b = self.background.pop(session, None)
        if b is None or b.task.done():
            return False
        self.abandon(session)
        b.task.cancel()
        log.info("background request of %s stopped by the owner", session)
        return True

    def listening(self, session: str) -> bool:
        """Will the dialog that asked speak this session's answer itself?"""
        return _live(self.dialogs.get(session))

    # ---- questions ----
    def ask(self, session: str, kind: str, text: str, timeout: float) -> Ask:
        a = Ask(session, kind, text, timeout)
        self.asks[a.id] = a
        own = self.dialogs.get(session)
        if session in self.abandoned:
            log.info("ask %s (%s) for %s: its request was dropped, not asked", a.id, kind, session)
            a.resolve("no_answer")
        elif _live(own) and own.take_ask(a):
            log.info("ask %s (%s) in the live dialog of %s", a.id, kind, session)
        elif (d := self.current()) is not None and d.take_ask(a, lead=FOREIGN_LEAD):
            log.info("ask %s (%s) for %s in the Live dialog in progress (%s)", a.id, kind, session, d.st.peer)
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

    # ---- answers ----
    def result(self, session: str, text: str) -> str:
        """Who tells the owner this session's answer: "dialog" (the one that asked), "current" (handed to the Live
        dialog in progress), "announce" (nobody: the caller announces it) or "none" (not ours to tell)."""
        if session not in self.known or session in self.abandoned or session in self.announced:
            return "none"
        if self.listening(session):
            return "dialog"
        self.announced.add(session)
        d = self.current()
        if d is not None:
            d.background_result(self.requests.get(session, ""), text)
            return "current"
        return "announce"

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


_RESULT_LOG = {"dialog": "the dialog speaks it", "current": "told in the Live dialog in progress",
               "announce": "announce it", "none": "not ours to tell"}


async def _post_result(request: web.Request):
    body = await request.json()
    session = str(body.get("session") or "")
    # Only sessions a Live dialog started: the agent also answers Home Assistant's own pipeline, which speaks
    # its answers itself.
    who = registry.result(session, str(body.get("text") or "").strip())
    log.info("result of %s: %s", session or "?", _RESULT_LOG[who])
    return web.json_response({"announce": who == "announce"})


async def _get_session(request: web.Request):
    return web.json_response({"background": request.match_info["id"] in registry.detached})


def app() -> web.Application:
    a = web.Application(middlewares=[_auth])
    a.router.add_post("/ask", _post_ask)
    a.router.add_get("/ask/{id}", _get_ask)
    a.router.add_post("/result", _post_result)
    a.router.add_get("/session/{id}", _get_session)
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
