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
  GET  /session/ID                   -> {"background": bool, "think": bool}
                The request went on in the background (nobody waits for it), or is a website task: the agent
                may think longer.

Agent sessions are bound to dialogs by their X-Hermes-Session-Id: the dialog names each request's session (agent.py).
Requests running in the background and website tasks are jobs, listed for every dialog for a while (what was asked,
the agent's last step or its answer): the owner can ask how one is going, add to it or change it (the dialog steers
the running request, or continues the job's session), or stop it.
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


class Job:
    """An agent request the owner may come back to: one that went on in the background, or a website task. Numbered
    for the dialog model (ask_hermes task=N continues or steers it, stop_background stops it)."""

    def __init__(self, no: int, session: str, request: str):
        self.no, self.session, self.request = no, session, request
        self.task: asyncio.Task | None = None
        self.started = time.monotonic()
        self.step, self.step_at = "", 0.0
        self.done_at: float | None = None
        self.result = ""
        self.told = False          # its answer reached the owner (a dialog's note, or an announcement)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def attach(self, task: asyncio.Task):
        """A run of this job's session (the first one, or one continuing it)."""
        if task is self.task:
            return
        self.task, self.started, self.done_at, self.told = task, time.monotonic(), None, False
        task.add_done_callback(self._finished)

    def _finished(self, t: asyncio.Task):
        if t is not self.task:
            return
        self.done_at = time.monotonic()
        if t.cancelled():
            self.result = "остановлена."
        elif t.exception() is not None:
            self.result = f"не удалась: {t.exception()}"[:200]
        else:
            self.result = str(t.result())[:300]

    def describe(self, now: float) -> str:
        out = f"№{self.no}. «{self.request[:200]}» — "
        if self.running:
            out += f"идёт {_ago(now - self.started)}"
            if self.step:
                out += f"; последний шаг агента {_ago(now - self.step_at)} назад: {self.step}"
        else:
            out += f"закончена {_ago(now - (self.done_at or now))} назад: {self.result}"
        return out


def _ago(s: float) -> str:
    return f"{int(s // 60)} мин" if s >= 60 else f"{int(s)} с"


def _live(d) -> bool:
    return d is not None and not d.ending and not d.st.closing


class Registry:
    RECENT_S = 1800.0   # a finished job stays on the list this long (the owner may continue it)
    LISTED = 6

    def __init__(self):
        self.dialogs: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()  # session -> dialog
        self.open: "weakref.WeakSet" = weakref.WeakSet()  # Live dialogs in progress
        self.detached: set[str] = set()   # sessions whose current request went on in the background
        self.think: set[str] = set()      # sessions whose model calls think from the start (website tasks)
        self.runs: dict[str, str] = {}    # session -> its current run (agent.Agent.steer)
        self.jobs: dict[str, Job] = {}    # session -> job
        self._jobs_made = 0
        self.known: set[str] = set()      # sessions a dialog started: only theirs are ever announced
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

    def bind(self, session: str, dialog, new_run: bool = False):
        """`dialog` is the one this session's questions and answer go to. `new_run`: a request of it starts now (the
        dialog waits for it; not in the background)."""
        if session:
            self.dialogs[session] = dialog
            self.known.add(session)
            self.abandoned.discard(session)
            if new_run:
                self.detached.discard(session)

    def abandon(self, session: str, dialog=None):
        """Nobody waits for this session's request any more (closing it stops the agent's run). With `dialog`: only
        while the session is still that dialog's."""
        if not session or (dialog is not None and self.dialogs.get(session) is not dialog):
            return
        self.dialogs.pop(session, None)
        self.abandoned.add(session)
        for a in self.asks.values():
            if a.session == session:
                a.resolve("no_answer")
        log.info("session %s abandoned: its request was dropped", session)

    def track(self, session: str, request: str, task: asyncio.Task) -> Job | None:
        """Keep this request on the owner's list of jobs (a continued session keeps its number and first words)."""
        if not session:
            return None
        job = self.jobs.get(session)
        if job is None:
            self._jobs_made += 1
            job = self.jobs[session] = Job(self._jobs_made, session, request)
        job.attach(task)
        return job

    def detach(self, session: str, task: asyncio.Task | None = None, request: str = ""):
        if not session:
            return
        self.detached.add(session)
        if task is not None and not task.done():
            self.track(session, request, task)

    def progress(self, session: str, step: str):
        job = self.jobs.get(session)
        if job is not None and step:
            job.step, job.step_at = step[:120], time.monotonic()

    def listed(self) -> list[Job]:
        """The jobs a dialog model is told about: running ones, and ones finished in the last RECENT_S."""
        now = time.monotonic()
        for s in [s for s, j in self.jobs.items() if not j.running and j.done_at and now - j.done_at > self.RECENT_S]:
            del self.jobs[s]
        jobs = [j for j in self.jobs.values() if j.running or j.done_at is not None]
        return sorted(jobs, key=lambda j: j.no)[-self.LISTED:]

    def running(self) -> list[Job]:
        return [j for j in self.listed() if j.running]

    def job(self, no) -> Job | None:
        try:
            no = int(no)
        except (TypeError, ValueError):
            return None
        return next((j for j in self.listed() if j.no == no), None)

    def stop(self, session: str) -> bool:
        """The owner stopped a job: cancelling its request stops the agent's run."""
        job = self.jobs.get(session)
        if job is None or not job.running:
            return False
        self.abandon(session)
        job.task.cancel()
        log.info("job №%d (%s) stopped by the owner", job.no, session)
        return True

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
    def result(self, session: str, text: str, can_announce: bool = True) -> str:
        """Who tells the owner this session's answer: "dialog" (the dialog waiting for the request: its reply),
        "note" (the session's dialog, or the Live dialog in progress, gets it as a note), "announce" (no dialog: the
        caller announces it) or "none" (not ours to tell, or told already). Called by the agent side before its
        answer goes out (/result), and by the bridge when the request returns (`can_announce` False): whichever
        comes first tells it."""
        if session not in self.known or session in self.abandoned:
            return "none"
        if session not in self.detached:
            return "dialog" if _live(self.dialogs.get(session)) else "none"
        job = self.jobs.get(session)
        if job is not None and job.told:
            return "none"
        own = self.dialogs.get(session)
        d = own if _live(own) else self.current()
        if d is None and not can_announce:
            return "none"
        if job is not None:
            job.told = True
        if d is not None:
            d.background_result(job.request if job else "", text)
            return "note"
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


_RESULT_LOG = {"dialog": "the dialog speaks it", "note": "told in a Live dialog", "announce": "announce it",
               "none": "not ours to tell (or told already)"}


async def _post_result(request: web.Request):
    body = await request.json()
    session = str(body.get("session") or "")
    # Only sessions a Live dialog started: the agent also answers Home Assistant's own pipeline, which speaks
    # its answers itself.
    who = registry.result(session, str(body.get("text") or "").strip())
    log.info("result of %s: %s", session or "?", _RESULT_LOG[who])
    return web.json_response({"announce": who == "announce"})


async def _get_session(request: web.Request):
    sid = request.match_info["id"]
    return web.json_response({"background": sid in registry.detached, "think": sid in registry.think})


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
