"""Pipeline backend: OpenRouter (the chat model hears the speech, its reply is cut into sentences and spoken by a
TTS model) with the home agent behind the ask_hermes tool (through Home Assistant by default, see agent.py).

The VAD in Session does the turn-taking: it finds the end of an utterance, which goes to the chat model as audio;
the reply text is cut into sentences and each sentence is synthesised and streamed to the station while the model
keeps writing. Speech over the answer cancels all of it (barge-in); tool calls keep running through a barge-in and
their results are spoken later. The dialog ends when the model calls end_dialog (at once: no further model round,
only what it said alongside is played out), after IDLE_S of silence once the
answer has played, or when the station ends it.

A dialog the house starts (the hello carries "reminder": a station turned HA's start_conversation into it) opens with
that text spoken on every station that got it, and only the one that hears the owner loudest goes on (reminders.py).

The first turn of a dialog the owner opened has nothing before it to resolve: its transcript is already a
self-contained request. The agent is asked with it at once, beside the model's round; when the model asks the agent
too (nearly every first turn is a command), it gets that request's answer, ~0.5 s sooner (LIVE_AGENT_RACE=0: off).

An agent request still running after LIVE_AGENT_DETACH_S goes on in the background: the owner hears that it is being
done and the dialog may end. Its answer is spoken in the dialog if that is still open; otherwise the agent side
announces it (surface.py /result). Questions the agent asks the owner meanwhile (codes, yes / no) are spoken here when
this dialog started its session; the reply goes back to the agent only (no model round, no history).

Zero data retention (util.ZDR, on unless LIVE_ZDR=0): every OpenRouter request (model and TTS) carries
provider.zdr, and the default endpoints of each model are ZDR ones (Gemini: Vertex only, AI Studio keeps data).

Options (routes.json): llm, tts_voice, instructions (a file); the agent's: agent, ha_agent_id, ha_device_id,
ha_satellite_id (agent.py).
"""
import asyncio
import base64
import json
import logging
import os
import re
import secrets
import time
import weakref
from pathlib import Path

import aiohttp

from .. import reminders, stt, surface
from ..agent import Agent
from ..session import Session
from ..util import TAG_RE, ZDR, BackendUnavailable, HERE, clean_for_speech, read_instructions, wav16k
from ..wire import Station, frame

log = logging.getLogger("live")

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OR_PROXY = os.environ.get("LIVE_OR_PROXY") or None  # OpenRouter refuses some countries' IPs
REASONING = os.environ.get("LIVE_REASONING", "")  # "" = per model (reasoning_for), "off", or an effort: minimal, low, ...
# The chat model decides (ask the agent / answer / end) on a transcript. Measured 2026-10-27 on 42 real turns, ZDR
# endpoints, median / p90 to the decision: deepseek-v4.1-flash (Together, Baseten) 0.53 / 0.76 s, gemini-3.5-flash-lite
# (Vertex) 0.71 / 1.01 s, gemini-3.6-flash (Vertex) 0.93 / 1.50 s; the same decisions (Gemini 3.6 a little worse).
LLM = os.environ.get("LIVE_LLM", "deepseek/deepseek-v4.1-flash")
TTS = os.environ.get("LIVE_TTS", "elevenlabs/eleven-turbo-v2.5")  # ~0.3 s to first audio (eleven-flash-v2.5: the same through OpenRouter); google/gemini-3.8-flash-tts takes 1.6-4.5 s
TTS_VOICE = os.environ.get("LIVE_TTS_VOICE", "matilda")  # gemini: Kore
TTS_SPEED = float(os.environ.get("LIVE_TTS_SPEED", "1.1"))  # ElevenLabs accepts 0.7-1.2 through OpenRouter; use 1.0 with gemini
TTS_STYLE = os.environ.get("LIVE_TTS_STYLE", (
    "Говори по-русски живо и естественно, как человек с настроением: быстрый разговорный темп, лёгкая "
    "дерзость и ирония, без дикторской интонации. Читай только сам текст, ничего не добавляй."))
DIRECT_MAX = int(os.environ.get("LIVE_DIRECT_MAX", "260"))  # Hermes answers up to this many chars are spoken as they are (0 = always reword)
LLM_TIMEOUT = float(os.environ.get("LIVE_LLM_TIMEOUT", "60"))
LLM_HEDGE_S = float(os.environ.get("LIVE_LLM_HEDGE_S", "1.5"))  # nothing useful from the model after this long: race a 2nd identical request (0 = off)
TTS_HEDGE_S = float(os.environ.get("LIVE_TTS_HEDGE_S", "0.8"))  # no audio after this long: race a 2nd TTS request (0 = off)
AGENT_RACE = os.environ.get("LIVE_AGENT_RACE", "1") != "0"  # first turn: the agent is asked with the transcript at once
DETACH_S = float(os.environ.get("LIVE_AGENT_DETACH_S", "20"))  # an agent request still running this long goes on in the background (0 = never)
DETACHED_NOTE = ("Это дело идёт дальше в фоне: если понадобится код или подтверждение, ты спросишь хозяина сама, итог "
                 "скажешь, когда закончишь.")
DETACHED_SPEECH = "Это надолго — занимаюсь, позову, если что-то понадобится."
STEERED_NOTE = ("Принято в идущее дело: ты учтёшь это после текущего шага, итог скажешь, когда закончишь. Если хозяину "
                "ещё ничего не ответила — коротко подтверди от себя («Хорошо, добавлю»).")
ACKS = {"code": "Принято, ввожу.", "yes": "Хорошо.", "no": "Поняла, не делаю.", "text": "Поняла."}
_background: set = set()  # detached agent requests: they outlive their dialog
_session_of: "weakref.WeakKeyDictionary[asyncio.Task, str]" = weakref.WeakKeyDictionary()  # agent request -> agent session
ZDR_PROVIDER = {"zdr": True, "data_collection": "deny"}
COVERED = object()  # run_call: the first ask of the round got the whole utterance


def provider_for(model: str) -> dict | None:
    """OpenRouter routing of the chat model. OpenRouter load-balances by price, which for Gemini includes the "flex"
    tiers (median 7-11 s to a first token, p99 85 s, measured 2026-10-26): the default pins the fast endpoints and never
    falls back past them. LIVE_LLM_PROVIDER: JSON instead of the default, "" = none. With ZDR, zdr is always added."""
    env = os.environ.get("LIVE_LLM_PROVIDER")
    if env is not None:
        prov = json.loads(env) if env else {}
    elif model.startswith("google/"):  # Google's ZDR endpoints are the Vertex ones only
        prov = {"order": ["google-vertex/global", "google-vertex/global/priority", "google-vertex/us"],
                "allow_fallbacks": False}
    elif model.startswith("deepseek/"):  # the fastest ZDR hosts of deepseek-v4.1-flash, ~0.23 s to a first token
        prov = {"order": ["together", "baseten", "inference-net"], "allow_fallbacks": False}
    else:
        prov = {}
    if ZDR:
        prov = dict(prov, **ZDR_PROVIDER)
    return prov or None


def reasoning_for(model: str) -> dict | None:
    if REASONING == "off":
        return {"enabled": False}
    if REASONING:
        return None if "lite" in model else {"effort": REASONING}  # flash-lite takes no reasoning parameters
    if model.startswith("google/"):
        return None if "lite" in model else {"effort": "minimal"}
    if model.startswith("deepseek/"):
        return {"enabled": False}
    return None


# Added to the persona for a dialog that opened with a reminder (it is not in the history: that starts with the owner).
REMINDER_NOTE = (
    "\n\nЭтот разговор начала ты сама: по расписанию ты только что сказала хозяину вслух: «{text}». Он может "
    "ответить, переспросить, спросить подробности, попросить напомнить позже (тогда поставь новое напоминание "
    "через ask_hermes) или больше не напоминать об этом (тогда попроси ask_hermes удалить это напоминание, назвав "
    "его текст). На «о чём это?» отвечай сама: текст напоминания ты знаешь. Если он просто подтвердил («ок», "
    "«спасибо», «понял») — сразу вызови end_dialog.")
# The local transcript next to the audio of a turn (LIVE_INPUT=both, stt.py).
STT_HINT = "(Расшифровка этой реплики локальным распознавателем, может ошибаться — сверяй со звуком: «{text}».)"
# Added to the persona when the owner's turns come as transcripts alone (LIVE_INPUT=text).
TEXT_NOTE = (
    "\n\nРеплики хозяина приходят расшифровкой его речи (локальное распознавание), звука ты не слышишь. В расшифровке "
    "бывают ошибки — понимай по смыслу и по разговору. Выведи ровно [SILENT] только если расшифровка совсем "
    "бессмысленна (обрывок слова, случайный набор слов) или это явно чужой разговор или телевизор; любая обычная "
    "реплика, просьба или подкол — это тебе, отвечай.")

# Added to the persona while there are jobs the owner may come back to (surface.Job): this dialog's or earlier ones'.
JOBS_NOTE = (
    "\n\nТвои дела, идущие и недавние (ты делаешь их сама, через ask_hermes):\n{jobs}\n"
    "Реплика хозяина про одно из них — уточнить, добавить, изменить, продолжить, спросить, что именно сделано, — "
    "ask_hermes с task=номер, а в request его слова как есть: идущее дело ты учтёшь на ходу, законченное "
    "продолжишь со всем, что о нём знаешь. Новое, не связанное — ask_hermes без task (там об этих делах ничего не "
    "известно). «Как там?» про идущее дело — отвечай сама по этому списку. Отменить идущее — stop_background с "
    "номером. Код или подтверждение ты спросишь у хозяина сама, когда понадобится.{current}")
CURRENT_JOB = "\nСейчас разговор о деле №{no}."
# Added to the persona of a dialog the house opened to tell a job's result (surface /result: "announce").
ANNOUNCED_NOTE = (
    "\n\nЭтот разговор начала ты сама: ты закончила дело №{no} («{request}») и только что сказала хозяину вслух: "
    "«{text}». Подробный итог, который ты знаешь: {result}\nСпросит подробности — отвечай по нему. Захочет что-то ещё "
    "по этому делу (добавить, изменить, продолжить) — ask_hermes с task={no} и его словами как есть. Просто "
    "подтвердил («ок», «спасибо», «понял») — сразу вызови end_dialog.")
# How the house's announcement of a finished job is written (summarize()).
SUMMARY_NOTE = (
    "\n\nСейчас ты сама начинаешь разговор: ты закончила дело хозяина и коротко говоришь ему итог. Не больше двух "
    "коротких фраз, до двадцати пяти слов, по-русски, от первого лица: что получилось и одно самое важное (цифра, "
    "срок, время). Ничего не перечисляй и не зачитывай данные: если итог — список или много цифр, скажи, что "
    "готово и сколько, и предложи рассказать подробнее. Без слов «агент», «задача номер», без английского. Нужен "
    "ответ хозяина — короткий вопрос в конце. Ответ — только то, что ты скажешь вслух.")
SUMMARY_MAX = 260  # chars; a longer one is cut after its second sentence

SENT_RE = re.compile(r"(.+?[.!?…]+[\"»)]?)(?:\s+|$)", re.S)

TOOLS = [
    {"type": "function", "function": {
        "name": "ask_hermes",
        "description": (
            "Ask the home agent (Hermes) to do or look up something. Use it for EVERYTHING that touches the "
            "real world or personal data: smart home control and device states (lights, switches, climate, "
            "vacuum, media), timers and reminders, calendar, money and spending, mail, weather, web "
            "search, websites (it has a browser: orders, bookings, accounts) and the owner's remembered facts. "
            "Never claim an action happened unless this tool's answer says so. To the owner you and this tool are "
            "one: speak of its work as your own ('Сейчас найду', 'Добавлю', 'Заказала'), never 'передала', "
            "'агент', 'Гермес'. Alongside the call say at most a short filler ('Щас гляну.', 'Займусь.'): do not "
            "ask the owner its questions yourself and do not guess its answer; when it needs a choice, a "
            "confirmation or a code, the owner is asked by the speakers on its own."),
        "parameters": {"type": "object", "properties": {
            "request": {"type": "string",
                        "description": "The full request in Russian, self-contained (resolve 'it', 'there', dates)."},
            "browser": {"type": "boolean",
                        "description": "true when it needs websites: ordering, booking, an account on a site, looking "
                                       "something up on a particular site (the agent then thinks it through)."},
            "task": {"type": "integer",
                     "description": "Only when this is about one of your ongoing or recent jobs listed in the "
                                    "instructions: its number (request: the owner's words as they are). Omit for "
                                    "anything new."}},
            "required": ["request"]}}},
    {"type": "function", "function": {
        "name": "end_dialog",
        "description": "End the conversation right now. Call it, without saying anything, as soon as the owner is "
                       "done: thanks you and wants nothing more ('ок, спасибо', 'спасибо, всё', 'понял, спасибо'), "
                       "says goodbye, 'хватит', 'отбой', or tells you to stop listening. The station stops listening "
                       "at once; there is no next turn.",
        "parameters": {"type": "object", "properties": {}}}},
]
# Offered only while the agent has running jobs.
STOP_TOOL = {"type": "function", "function": {
    "name": "stop_background",
    "description": "Stop one of your running jobs (numbered in the instructions), when the owner asks to cancel or "
                   "stop it. The work is interrupted; what was already done stays done.",
    "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}}


def _args(call) -> dict:
    try:
        a = json.loads(call.get("args") or "{}")
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def tts_body(text: str, voice: str = "") -> dict:
    body = {"model": TTS, "input": text, "voice": voice or TTS_VOICE, "response_format": "pcm"}
    if TTS_SPEED != 1.0:
        body["speed"] = TTS_SPEED
    if TTS.startswith("google/"):
        body["instructions"] = TTS_STYLE  # only Gemini takes a style prompt
    if ZDR:
        body["provider"] = dict(ZDR_PROVIDER)
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
        self.provider = provider_for(self.llm)
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
        self.asked = {}        # call id -> the agent request, for notes about results the owner has not heard
        self.questions = []    # surface.Ask the agent asked the owner, the first one is being asked
        self.in_tools = False  # the reply waits for its tool calls (the agent may be asking the owner meanwhile)
        self.round_said = False  # the last model round said something (not only tool calls)
        self.job_no = 0        # the job this dialog is about (its announcement opened it, or the owner went back to it)
        self.woken = False     # the house opened this dialog to ask the owner something (surface.py)

    async def open(self):
        if not OR_KEY:
            raise BackendUnavailable("OPENROUTER_API_KEY is not set")
        self.agent.check()
        if self.instructions_file:
            p = Path(self.instructions_file)
            self.instructions = (p if p.is_absolute() else HERE / p).read_text(encoding="utf-8")
        else:
            self.instructions = read_instructions()
        log.info("dialog from %s: %s (llm %s%s, tts %s/%s, agent %s)", self.st.peer, self.st.hello, self.llm,
                 " via " + ",".join(self.provider.get("order") or ["any"]) + (" zdr" if self.provider.get("zdr") else "")
                 if self.provider else "", TTS, self.voice, self.agent.describe())
        self._warm = asyncio.create_task(self.prewarm())
        surface.registry.opened(self)
        if stt.needs_text() and stt.mode() == "text":
            self.instructions += TEXT_NOTE
        reminder = clean_for_speech(str(self.st.hello.get("reminder") or ""))
        if reminder:
            a = surface.registry.for_wake(reminder)
            if a is not None:  # the agent's question: the group says it, the answer goes back to the agent
                self.woken = True
                a.asked = True
                a.holders.add(self)
                self.questions.append(a)
                log.info("dialog opened for ask %s (%s)", a.id, a.kind)
            elif (job := surface.registry.for_announcement(reminder)) is not None:
                self.job_no = job.no  # the house tells a job's result: the owner may go on with it
                surface.registry.bind(job.session, self)
                self.instructions += ANNOUNCED_NOTE.format(no=job.no, request=job.request[:300], text=reminder,
                                                           result=job.result[:1500] or reminder)
                log.info("dialog opened for job №%d", job.no)
            else:
                self.instructions += REMINDER_NOTE.format(text=reminder)
            reminders.join(self, reminder, self.tts_chunks)

    async def prewarm(self):
        """Open two connections to OpenRouter while the owner is still speaking: through the proxy a new one costs
        ~0.3 s more than a kept one (0.34 s against 0.07 s), and the model's stream and the TTS of its first sentence
        run side by side."""
        async def one():
            try:
                async with self.http.get(f"{OR_BASE}/key", headers=or_headers(), proxy=OR_PROXY,
                                         timeout=aiohttp.ClientTimeout(total=8)) as r:
                    await r.read()
            except Exception:
                pass
        await asyncio.gather(one(), one())

    def working(self) -> bool:
        return any(not t.done() for t in self.tools.values())

    def followup_due(self) -> bool:
        return self.has_unspoken_results() and not self.questions  # not over the agent's question

    def on_utterance(self, pcm: bytes, text: str | None = None, pending=None):
        if self.group is not None and not self.group.won(self):
            if pending is not None:
                pending.cancel()
            return  # another station of the reminder answers
        if self.awaiting_answer():
            self.spawn(self.answer(self.questions[0], pcm, text, pending))
            return
        self.start_response(pcm, text, pending)

    async def close(self):
        """The dialog is over: agent requests it was waiting for stop (closing the request interrupts the agent);
        detached ones go on. A question nobody else can take any more gets no answer."""
        surface.registry.closed(self)
        for t in self.tools.values():
            if not t.done():
                t.cancel()
        for a in self.questions:
            a.holders.discard(self)
            if not a.holders:
                a.resolve("no_answer")

    # ---- the agent's questions to the owner (surface.py) ----
    def awaiting_answer(self) -> bool:
        return bool(self.questions) and self.questions[0].asked and not self.questions[0].future.done()

    def secret_turn(self) -> bool:
        return bool(self.questions) and self.questions[0].asked and self.questions[0].secret

    def take_ask(self, a, lead: str = "") -> bool:
        """Called by the registry: ask `a` in this dialog (after the questions already waiting). `lead`: said
        before it (a background request's question in a dialog that did not start it)."""
        if self.ending or self.st.closing:
            return False
        a.lead = lead
        a.holders.add(self)
        self.questions.append(a)
        if len(self.questions) == 1:
            self.spawn(self.pose(a))
        return True

    async def pose(self, a, prefix: str = ""):
        for _ in range(60):  # let what is playing, or a reply being written, finish (at most ~15 s)
            if not self.speaking and not (self.turn_open and not self.in_tools):
                break
            await asyncio.sleep(0.25)
        a.asked = True
        self.touch()
        await self.say(prefix + a.lead + a.text)
        if not self.turn_open:
            self.set_state("listening")

    async def say(self, text: str):
        async for pcm in self.tts_chunks(text):
            self.speaking = True
            self.touch()
            self.set_state("speaking")
            self.send(frame(b"A", pcm))
            await self.st.drain()

    async def answer(self, a, pcm: bytes, text: str | None, pending):
        """The owner's reply to question `a`: parsed and handed to the agent; never a model turn."""
        if text is None:
            text = await pending if pending is not None else await stt.transcribe(self.http, pcm, quiet=a.secret)
        if a.future.done():  # timed out meanwhile
            self.next_question()
            return
        status, value = a.parse(text or "")
        if status == "unclear" and a.tries < 1:
            a.tries += 1
            log.info("ask %s: answer unclear, asking again", a.id)
            await self.pose(a, "Не расслышала. ")
            return
        a.resolve(status, value)
        log.info("ask %s answered: %s", a.id, status if a.secret else f"{status} {value[:80]!r}")
        self.next_question()
        ack = ACKS.get(value if a.kind == "confirm" else a.kind, "") if status == "answered" else "Не поняла, пропускаю."
        if ack:
            await self.say(ack)
        if self.questions:
            self.spawn(self.pose(self.questions[0]))
        elif self.woken:
            self.ending = "answered"
        elif not self.turn_open and not self.speaking:
            self.set_state("listening")

    def next_question(self):
        if self.questions:
            self.questions.pop(0).holders.discard(self)

    def asking_for(self, session: str) -> bool:
        return bool(session) and any(a.session == session and not a.future.done() for a in self.questions)

    # ---- requests running in the background (surface.py) ----
    def background_result(self, request: str, text: str):
        """A job that went on in the background has finished while the owner is in this dialog."""
        self.notes.append(f"(Ты закончила дело «{request}». Итог: {text or 'готово.'} Скажи хозяину главное коротко, "
                          "от себя; подробности — если спросит.)")

    def jobs_note(self, jobs) -> str:
        if not jobs:
            return ""
        now = time.monotonic()
        current = CURRENT_JOB.format(no=self.job_no) if any(j.no == self.job_no for j in jobs) else ""
        return JOBS_NOTE.format(jobs="\n".join(j.describe(now) for j in jobs), current=current)

    def stop_background(self, args: dict) -> str:
        job = surface.registry.job(args.get("number"))
        if job is None or not job.running:
            return "Нет идущей задачи с таким номером." if surface.registry.running() else "Идущих задач нет."
        surface.registry.stop(job.session)
        return f"Остановлено: «{job.request[:120]}». Агент прерван; что он успел сделать до этого, отменено не было."

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
                req = self.asked.get(cid, "")
                self.notes.append(f"(Ответ на прежнюю просьбу «{req}» хозяин ещё не слышал: его перебили. "
                                  "Скажи его тоже, вместе с ответом на новую реплику.)")
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
        if flush and self.group is not None:
            self.group.cut_off(self)  # talked over the reminder: no more of it here
        if self.resp and not self.resp.done():
            self.orphan_pending()
        for t in (self.resp, self.tts_task):
            if t and not t.done():
                t.cancel()
        if flush:
            self.send(frame(b"F"))
        self.speaking = False
        self.turn_open = False

    def start_response(self, pcm: bytes, text: str | None = None, pending=None):
        """`text`: a transcript already made (a short utterance the STT kept: it always goes to the model);
        `pending`: the transcript being made ahead (stt.py)."""
        self.cancel_response(flush=False)
        self.turn_open = True
        self.set_state("thinking")
        self.resp = asyncio.create_task(self.respond((pcm, text, pending)))

    def add_user_turn(self, pcm: bytes, text: str | None, forced: bool):
        """LIVE_INPUT=text: the transcript alone (the sound when there is none: the STT failed); both: the transcript
        next to the sound; audio: the sound (with a transcript only for a short utterance the STT kept)."""
        for i in [i for i, m in enumerate(self.history) if isinstance(m.get("content"), list)][:-2]:
            self.history[i] = {"role": "user", "content": "(реплика хозяина)"}  # only the last turns keep their sound
        mode = stt.mode()
        if mode == "text" and text:
            content = text
        else:
            content = [{"type": "input_audio",
                        "input_audio": {"data": base64.b64encode(wav16k(pcm)).decode(), "format": "wav"}}]
            if text and (mode == "both" or forced):
                content.insert(0, {"type": "text", "text": STT_HINT.format(text=text)})
        self.history.append({"role": "user", "content": content})
        del self.history[:-14]
        while self.history and self.history[0].get("role") != "user":
            self.history.pop(0)  # never start with a tool result whose call was trimmed away

    async def respond(self, utt=None):
        """`utt`: (pcm, transcript or None, task making it or None) of the utterance this answers; None for a
        follow-up of late results."""
        if utt is not None:
            pcm, text, pending = utt
            forced = text is not None
            if text is None and stt.enabled():
                if stt.needs_text():
                    text = await pending if pending is not None else await stt.transcribe(self.http, pcm)
                elif pending is None:
                    self.spawn(stt.transcribe(self.http, pcm))  # only for the log
            if stt.mode() == "text" and text is not None and not stt.has_words(text):
                log.info("no words in the transcript, not answered")
                self.turn_open = False
                self.touch()
                if not self.speaking and not self.ending:
                    self.set_state("listening")
                return
            self.add_user_turn(pcm, text, forced)
            early = self.race(text)
        else:
            early = None
        q: asyncio.Queue = asyncio.Queue()
        self.tts_task = asyncio.create_task(self.tts_worker(q))
        try:
            for _round in range(4):
                calls = await self.llm_round(q)
                pre = {}  # call id -> the early request that answers it
                if early is not None:
                    asks = [c for c in calls if c["name"] == "ask_hermes" and not _args(c).get("task")]
                    if asks:
                        # the agent got the whole utterance: further new asks of this round are covered by it
                        pre = {asks[0]["id"]: early, **{c["id"]: COVERED for c in asks[1:]}}
                    else:
                        log.info("the model did not ask the agent anything new: the early request is dropped")
                        early.cancel()
                    early = None
                if not calls:
                    break
                self.shrink_audio(calls)
                # end_dialog takes effect at once: the dialog closes as soon as whatever was said in this round has
                # been played. Only a request made alongside still gets its answer (spoken as it is, or put into
                # words by one more round when it is long).
                ends = [c for c in calls if c["name"] == "end_dialog"]
                if ends:
                    log.info("end_dialog: the owner is done")
                    self.ending = "goodbye"
                    for c in ends:
                        self.history.append({"role": "tool", "tool_call_id": c["id"], "content": "ok"})
                    calls = [c for c in calls if c["name"] != "end_dialog"]
                    if not calls:
                        break
                tasks = []
                for c in calls:
                    try:
                        self.asked[c["id"]] = json.loads(c["args"] or "{}").get("request", "")
                    except ValueError:
                        pass
                    t = asyncio.create_task(self.run_call(c, pre.get(c["id"])))
                    self.tools[c["id"]] = t
                    t.add_done_callback(lambda t, c=c: self.tool_done(c, t))
                    tasks.append(t)
                # shield: a barge-in cancels this turn (speech, model stream) but not the tools themselves
                self.in_tools = True
                try:
                    results = await asyncio.gather(*[asyncio.shield(t) for t in tasks])
                finally:
                    self.in_tools = False
                for c, r in zip(calls, results):
                    self.history.append({"role": "tool", "tool_call_id": c["id"], "content": r})
                    self.tools.pop(c["id"], None)
                if self.round_said and all(r == STEERED_NOTE for r in results):
                    break  # what was said alongside the call is the reply ("Хорошо, добавлю")
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
            if early is not None:
                early.cancel()  # cut short before the model decided
            self.turn_open = False
            self.touch()
            if not self.speaking and not self.ending:
                self.set_state("listening")

    def race(self, text: str | None):
        """The first turn of a dialog the owner opened: ask the agent with the transcript now, beside the model round
        (see the module doc). None when this turn is not one."""
        if not (AGENT_RACE and stt.has_words(text) and self.group is None and not self.st.hello.get("reminder")
                and sum(1 for m in self.history if m.get("role") == "user") == 1):
            return None
        if surface.registry.listed():  # it may be about one of the agent's jobs: the model decides first
            return None
        log.info("first turn: the agent is asked with the transcript beside the model round")
        return self.spawn(self.ask_agent(text))

    def direct_answer(self, calls, results) -> str:
        """A short Hermes answer is already in the right words (its voice profile carries the persona):
        speak it as it is and skip the second model round (~2 s)."""
        if (not DIRECT_MAX or self.carry or len(calls) != 1 or calls[0]["name"] != "ask_hermes"
                or self.has_unspoken_results()):  # an earlier interrupted request is ready: let the model say both
            return ""
        if results[0] == DETACHED_NOTE:
            return DETACHED_SPEECH
        text = clean_for_speech(results[0])
        if not text or len(text) > DIRECT_MAX or text.startswith(("Агент не ответил", "Сделано, ответ пустой")):
            return ""
        return text

    def shrink_audio(self, calls):
        """The model has heard the question and put it into the tool request: keep that instead of the sound."""
        req = next((json.loads(c["args"] or "{}").get("request", "") for c in calls if c["name"] == "ask_hermes"), "")
        for i in range(len(self.history) - 1, -1, -1):
            m = self.history[i]
            if m.get("role") == "user":  # only this turn's own sound (a text turn has none)
                if isinstance(m.get("content"), list):
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

        prov = self.provider
        tasks = {0: asyncio.create_task(one(0, prov))}
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
                    other = dict(prov, order=prov["order"][::-1]) if prov and prov.get("order") else prov
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
                            log.warning("llm request failed (%s), retrying once without provider pinning%s",
                                        str(val)[:120], " (ZDR endpoints only)" if ZDR else "")
                            tasks[1] = asyncio.create_task(one(1, dict(ZDR_PROVIDER) if ZDR else None))
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
        jobs = surface.registry.listed()
        body = {"model": self.llm, "stream": True, "tools": TOOLS + ([STOP_TOOL] if any(j.running for j in jobs) else []),
                "messages": [{"role": "system", "content": self.instructions + self.jobs_note(jobs)}]
                + self.history}  # provider pinning: llm_events
        reasoning = reasoning_for(self.llm)
        if reasoning:
            body["reasoning"] = reasoning
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
        self.round_said = bool(msg["content"])
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

    async def tts_chunks(self, s: str):
        """24 kHz s16 PCM of `s` as the TTS streams it (even-sized chunks); nothing when the TTS refuses. OpenRouter
        hands ElevenLabs audio over whole (0.27 s for a few words, ~0.45 s for 130 chars), but now and then takes
        1-1.5 s: with no audio after TTS_HEDGE_S a second identical request is raced, the first to deliver wins."""
        t0 = time.monotonic()
        q: asyncio.Queue = asyncio.Queue()

        async def attempt(i: int):
            try:
                async with self.http.post(f"{OR_BASE}/audio/speech", json=tts_body(s, self.voice), headers=or_headers(),
                                          proxy=OR_PROXY, timeout=aiohttp.ClientTimeout(total=60, sock_read=30)) as r:
                    if r.status != 200:
                        await q.put((i, "err", f"tts HTTP {r.status}: {(await r.text())[:200]}"))
                        return
                    async for chunk in r.content.iter_chunked(4800):
                        await q.put((i, "pcm", chunk))
                await q.put((i, "end", None))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await q.put((i, "err", f"tts failed: {e!r}"))

        tasks = {0: asyncio.create_task(attempt(0))}
        winner, hedged, first, carry = None, False, True, b""
        try:
            while True:
                wait = None
                if winner is None and not hedged and TTS_HEDGE_S > 0:
                    wait = max(0.01, TTS_HEDGE_S - (time.monotonic() - t0))
                try:
                    i, kind, val = await asyncio.wait_for(q.get(), wait)
                except asyncio.TimeoutError:
                    hedged = True
                    log.info("tts silent for %.1fs, racing a second request: %r", TTS_HEDGE_S, s[:60])
                    tasks[1] = asyncio.create_task(attempt(1))
                    continue
                if winner is not None and i != winner:
                    continue
                if kind != "pcm":
                    tasks.pop(i, None)
                    if winner is None and tasks:
                        continue  # the other request may still deliver
                    if kind == "err":
                        log.warning("%s", val)
                    return
                if winner is None:
                    winner = i
                    for j, t in tasks.items():
                        if j != i:
                            t.cancel()
                pcm = carry + val
                carry = pcm[len(pcm) // 2 * 2:]
                pcm = pcm[:len(pcm) // 2 * 2]
                if not pcm:
                    continue
                if first:
                    first = False
                    log.info("tts first audio %.2fs%s: %r", time.monotonic() - t0, " (2nd request)" if i else "", s[:80])
                yield pcm
        finally:
            for t in tasks.values():
                t.cancel()

    async def tts_worker(self, q: asyncio.Queue):
        while True:
            s = await q.get()
            if s is None:
                return
            async for pcm in self.tts_chunks(s):
                self.speaking = True
                self.touch()
                self.set_state("speaking")
                self.send(frame(b"A", pcm))
                await self.st.drain()

    async def run_call(self, c, early=None) -> str:
        """`early`: the request the race started for this call (race()), or COVERED: that request covers it too."""
        try:
            args = json.loads(c["args"] or "{}")
        except ValueError:
            args = {}
        log.info("tool %s %s", c["name"], args)
        if c["name"] == "ask_hermes":
            if not self.speaking:
                self.set_state("thinking")
            if early is COVERED:
                return "Сделано вместе с первой просьбой: агент получил всю реплику хозяина целиком."
            request = str(args.get("request", ""))
            browser = bool(args.get("browser"))
            if args.get("task"):
                return await self.continue_job(args.get("task"), request, browser)
            if early is not None:
                log.info("the model asked the agent: it gets the request already running")
                task = early
                if browser:
                    self.website_task(_session_of.get(task, ""), request, task)
            else:
                task = asyncio.create_task(self.ask_agent(request, think=browser))
            return await self.wait_or_detach(task, request)
        if c["name"] == "stop_background":
            return self.stop_background(args)
        return f"unknown tool {c['name']}"

    async def wait_or_detach(self, task: asyncio.Task, request: str) -> str:
        """The agent's answer, or DETACHED_NOTE when it takes longer than DETACH_S: the request then goes on in the
        background (it is no longer this dialog's to cancel) and its answer comes as a note, or as an
        announcement once the dialog is gone."""
        try:
            if DETACH_S <= 0:
                return await asyncio.shield(task)
            done, _ = await asyncio.wait({task}, timeout=DETACH_S)
            while task not in done and self.asking_for(_session_of.get(task, "")):
                done, _ = await asyncio.wait({task}, timeout=0.5)  # the owner is answering its question here
            if task in done:
                return task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise
        self.bg.discard(task)
        _background.add(task)
        sid = _session_of.get(task, "")
        surface.registry.detach(sid, task, request)
        log.info("agent request still running after %.0fs, it goes on in the background (session %s)", DETACH_S,
                 sid or "?")

        def finished(t: asyncio.Task):
            _background.discard(t)
            if not t.cancelled():  # unless the agent side has told it already (surface /result): a note
                surface.registry.result(sid, self.tool_text(t), can_announce=False)
        task.add_done_callback(finished)
        return DETACHED_NOTE

    def website_task(self, sid: str, request: str, task: asyncio.Task):
        """A request that needs websites: its model calls think from the start (the agent side asks GET /session),
        and it is one of the owner's jobs (they may come back to it)."""
        if sid:
            surface.registry.think.add(sid)
            surface.registry.track(sid, request, task)

    async def continue_job(self, no, request: str, browser: bool) -> str:
        """The owner's words about one of the agent's jobs: a running one gets them on the way (steer), a finished
        one goes on in its own session (the agent knows everything it did)."""
        job = surface.registry.job(no)
        if job is None:
            return "Нет дела с таким номером: если это новая просьба, вызови ask_hermes без task."
        self.job_no = job.no
        surface.registry.bind(job.session, self)  # its questions and its answer come to this dialog now
        if job.running:
            run_id = surface.registry.runs.get(job.session, "")
            if run_id and await self.agent.steer(run_id, request):
                log.info("job №%d steered: %r", job.no, request[:120])
                return STEERED_NOTE
            await asyncio.wait({job.task}, timeout=10)  # finishing just now: continue it once it has
            if job.running:
                return ("Сейчас это не принять в идущее дело: скажи хозяину от себя, что учтёшь, как только закончишь "
                        "текущий шаг, и предложи повторить чуть позже.")
        log.info("job №%d continued in its session %s: %r", job.no, job.session, request[:120])
        task = asyncio.create_task(self.ask_agent(request, session=job.session,
                                                  think=browser or job.session in surface.registry.think))
        surface.registry.track(job.session, request, task)
        return await self.wait_or_detach(task, request)

    async def ask_agent(self, request: str, session: str = "", think: bool = False) -> str:
        """One agent session per request, named here when the agent allows it (`session`: continue that one): it is
        bound to this dialog before the agent can ask anything. A request cancelled (dropped, cut short, the dialog
        gone without it) abandons its session: questions from that run are not put to anyone."""
        task = asyncio.current_task()

        def bound(sid: str):
            _session_of[task] = sid
            surface.registry.bind(sid, self, new_run=True)

        def step(label: str):
            surface.registry.progress(_session_of.get(task, ""), label)

        def run(run_id: str):
            surface.registry.runs[_session_of.get(task, "")] = run_id

        def said(text: str):
            surface.registry.said(_session_of.get(task, ""), text)
        sid = session or (f"live-{secrets.token_hex(8)}" if self.agent.names_sessions() else "")
        if sid:
            bound(sid)
            if think:
                self.website_task(sid, request, task)
        try:
            return await self.agent.ask(request, on_session=bound, session=sid, on_progress=step, on_run=run,
                                        on_said=said)
        except asyncio.CancelledError:
            surface.registry.abandon(_session_of.get(task, ""), self)
            raise
        except Exception as e:
            log.warning("agent failed: %r", e)
            return f"Агент не ответил: {e}"[:300]


async def summarize(request: str, result: str) -> str:
    """What the speakers say when a job finished with no dialog to tell it in: a short summary in the dialog
    model's voice (its persona), not the agent's working notes. Falls back to the result itself."""
    fallback = clean_for_speech(TAG_RE.sub("", result))[:400] or "Готово."
    if not OR_KEY:
        return fallback
    body = {"model": LLM, "stream": False, "max_tokens": 160, "messages": [
        {"role": "system", "content": read_instructions() + SUMMARY_NOTE},
        {"role": "user", "content": f"Просьба хозяина: «{request[:400]}»\nЧем кончилось (твои рабочие заметки): "
                                    f"{result[:3000]}"}]}
    prov = provider_for(LLM)
    if prov:
        body["provider"] = prov
    reasoning = reasoning_for(LLM)
    if reasoning:
        body["reasoning"] = reasoning
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(f"{OR_BASE}/chat/completions", json=body, headers=or_headers(), proxy=OR_PROXY,
                                 timeout=aiohttp.ClientTimeout(total=8)) as r:
                data = await r.json(content_type=None)
        text = clean_for_speech(TAG_RE.sub("", str(data["choices"][0]["message"].get("content") or "")))
    except Exception as e:
        log.warning("summary failed (%r): announcing the result as it is", e)
        return fallback
    if len(text) > SUMMARY_MAX:
        sentences = [m.group(1).strip() for m in SENT_RE.finditer(text)]
        text = " ".join(sentences[:2]) if sentences else text[:SUMMARY_MAX]
    log.info("summary %.1fs: %r", time.monotonic() - t0, text)
    return text or fallback


async def run(station: Station, http: aiohttp.ClientSession, options: dict):
    await Dialog(station, http, options).run()
