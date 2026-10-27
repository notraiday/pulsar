"""What every dialog backend that listens through the station's mic needs: the station side of the dialog and a
voice-activity turn detector (Silero VAD on the echo-cancelled 16 kHz mic stream).

A backend subclasses Session and fills in the hooks (open, on_utterance, cancel_response, ...). The base does the
reading of station frames, the VAD (utterance starts / ends, barge-in over a playing or pending answer, noise
rejection), the idle watchdog and the end of the dialog.
"""
import asyncio
import audioop
import logging
import math
import os
import time
from pathlib import Path

from pysilero_vad import SileroVoiceActivityDetector

from . import stt
from .util import wav16k
from .wire import Station, frame

log = logging.getLogger("live")

IDLE_S = float(os.environ.get("LIVE_IDLE_S", "8"))
FIRST_IDLE_S = float(os.environ.get("LIVE_FIRST_IDLE_S", "10"))  # nothing said after the wake word
# Silero VAD (neural, tells speech from keyboard clicks / room noise) on 32 ms frames
VAD_ON = float(os.environ.get("LIVE_VAD_ON", "0.5"))     # speech probability that makes a frame "voiced" outside an utterance
VAD_OFF = float(os.environ.get("LIVE_VAD_OFF", "0.35"))  # ... and inside one (hysteresis, bridges short pauses)
VOICED_RATIO = float(os.environ.get("LIVE_VOICED_RATIO", "0.75"))  # share of voiced frames in the start/barge window
END_SILENCE_MS = int(os.environ.get("LIVE_END_SILENCE_MS", "600"))
# ... when the transcript made ahead stops on a word that leads on ("и", "что", "в"): the owner is mid-thought
END_SILENCE_LONG_MS = int(os.environ.get("LIVE_END_SILENCE_LONG_MS", "1400"))
START_MS = int(os.environ.get("LIVE_START_MS", "192"))
BARGE_MS = int(os.environ.get("LIVE_BARGE_MS", "480"))   # speech needed to cut an answer that is playing or being made
# A barge-in while the answer plays must be this many dB above the cancelled/ambient floor (the level the station's
# echo canceller leaves between sounds, measured from the non-voiced frames). Room noise and AEC leak on transients
# sit ~5-15 dB over that floor; the owner addressing the speaker is ~20 dB over it. Lower it if real interruptions are
# missed, raise it if noise/echo still cuts the answer.
BARGE_FLOOR_MARGIN_DB = float(os.environ.get("LIVE_BARGE_FLOOR_MARGIN_DB", "14"))
EFLOOR_ALPHA = 0.1  # EWMA weight for the floor estimate (~320 ms)
MIN_SPEECH_MS = int(os.environ.get("LIVE_MIN_SPEECH_MS", "320"))
MAX_UTT_S = float(os.environ.get("LIVE_MAX_UTT_S", "25"))
FRAME_MS = 32
LEAD_FRAMES = 4  # audio kept before the detection window when an utterance starts
DUMP_DIR = os.environ.get("LIVE_DUMP_DIR") or None  # debug: wav of every utterance / barge-in, last DUMP_KEEP files
DUMP_KEEP = int(os.environ.get("LIVE_DUMP_KEEP", "80"))


class Session:
    FRAME = 1024  # bytes: 32 ms of s16le at 16 kHz, what Silero wants

    def __init__(self, station: Station):
        self.st = station
        self.state = ""
        self.speaking = False      # audio sent and not yet reported played
        self.turn_open = False     # a reply is being produced
        self.last_activity = time.monotonic()
        self.heard_user = False
        self.ending = None         # reason, once the dialog should end after playback
        self.vad = SileroVoiceActivityDetector()
        self.buf = b""
        self.flags = []   # per frame: voiced?
        self.probs = []   # per frame: Silero speech probability
        self.dbs = []     # per frame: mic level, dBFS
        self.ring = []    # last ~3 s of mic frames, for dumps
        self.pre = []
        self.utt = None
        self.voiced_in_utt = 0
        self.silence = 0
        self.echo = [0, 0.0, 0.0, 0]    # frames, sum dBFS, max VAD p, frames over VAD_ON: mic while the station plays
        self.echo_floor = None          # EWMA dBFS of the non-voiced mic while playing: the level the AEC leaves between sounds
        self.echo_floor_n = 0           # frames that went into it (gate only once there is a stable estimate)
        self.quiet = [0, 0.0]           # frames, sum dBFS: mic while nothing plays and nobody talks
        self.group = None               # reminders.Group of a dialog the house started (reminder on several stations)
        self.vad_off = False            # another station of the group answers: hear nothing more
        self.http = None                # backends that have one set it (stt.py uses it)
        self.spec = None                # transcript of the utterance so far, started at its first pause (stt.py)
        self.rest = None                # task deciding, from that transcript, when the pause ends the utterance
        self.overlap = False            # the owner talks while the answer is being made: it waits (on_utterance decides)
        self.utt_started = 0.0          # when the current / last utterance started (monotonic)
        self.bg: set = set()            # background tasks of the dialog (cancelled when it ends)
        self.play_end = 0.0             # when the audio sent so far will have played (monotonic; 0 = nothing queued)

    # ---- hooks for backends ----
    async def open(self):
        """Connect to whatever makes the answers. Raise BackendUnavailable (before sending anything) to hand off."""

    async def close(self):
        """Release what open() acquired."""

    def tasks(self) -> list:
        """Extra coroutines that run for the whole dialog; the dialog ends when any of them returns."""
        return []

    async def on_mic(self, pcm: bytes):
        self.feed(pcm)

    def on_utterance(self, pcm: bytes, text: str | None = None, pending=None):
        """The turn detector heard the end of an utterance (`text`: its local transcript when one was already needed,
        `pending`: a task that is making it)."""

    def working(self) -> bool:
        """Background work (tool calls) that keeps the dialog open although nothing plays."""
        return False

    def awaiting_answer(self) -> bool:
        """The agent asked the owner something (surface.py): the next utterance is the answer, not a barge-in."""
        return False

    def secret_turn(self) -> bool:
        """What the owner says now is secret (a code): its words are not logged and its sound is not dumped."""
        return False

    def followup_due(self) -> bool:
        return False

    def start_followup(self):
        pass

    def on_noise(self):
        """What the turn detector took for the start of an utterance was noise (nothing goes to the backend)."""

    def cancel_response(self, flush: bool):
        """The owner spoke over the answer (flush: the station is to drop what it has queued)."""
        if flush:
            self.send(frame(b"F"))
            self.play_end = 0.0
        self.speaking = False
        self.turn_open = False

    def lose(self, reason: str):
        """Another station of this reminder's group answers the owner: this dialog ends now."""
        self.vad_off = True
        self.utt, self.pre, self.flags = None, [], []
        self.overlap = False
        self.ending = reason
        self.cancel_response(flush=True)

    # ---- station I/O ----
    def send(self, data: bytes):
        self.st.send(data)

    def queued(self, pcm_bytes: int) -> tuple[float, float]:
        """`pcm_bytes` of 24 kHz s16 answer audio go to the station now: (start, end) of their playback, estimated.
        The station plays what it gets in order, at once when its queue is empty."""
        start = max(time.monotonic(), self.play_end)
        self.play_end = start + pcm_bytes / 48000
        return start, self.play_end

    def set_state(self, s: str):
        if s != self.state:
            self.state = s
            self.send(frame(b"S", s.encode()))

    def touch(self):
        self.last_activity = time.monotonic()

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self.bg.add(t)
        t.add_done_callback(self.bg.discard)
        return t

    async def run(self):
        await self.open()
        self.set_state("listening")
        tasks = [asyncio.create_task(c) for c in (self.uplink(), self.watchdog(), *self.tasks())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            t.cancel()
            if t.done() and not t.cancelled() and t.exception():
                log.warning("dialog task failed: %r", t.exception())
                self.ending = self.ending or "error"
        self.cancel_response(flush=False)
        for t in list(self.bg):
            t.cancel()
        if self.group is not None:
            self.group.leave(self)
        await self.close()
        await self.st.finish(b"X", self.ending or "closed")
        log.info("dialog from %s ended: %s", self.st.peer, self.ending)

    async def uplink(self):
        while True:
            f = await self.st.read()
            if f is None:
                self.ending = self.ending or "station disconnected"
                return
            t, p = f
            if t == b"A":
                await self.on_mic(p)
            elif t == b"D":
                self.speaking = False
                self.touch()
                if self.group is not None:
                    self.group.drained()
                if not self.turn_open:
                    if self.ending:
                        return
                    self.set_state("listening")
            elif t == b"E":
                self.ending = "station: " + p.decode(errors="replace")
                return

    async def watchdog(self):
        while True:
            await asyncio.sleep(0.25)
            busy = self.speaking or self.turn_open or self.utt is not None or self.working()
            if self.ending and not busy:
                return
            if not self.turn_open and self.utt is None and not self.ending and not self.st.closing \
                    and self.followup_due():
                self.start_followup()
                continue
            if busy:
                continue
            if time.monotonic() - self.last_activity > (IDLE_S if self.heard_user else FIRST_IDLE_S):
                self.ending = "idle"
                return

    # ---- voice activity ----
    def feed(self, pcm: bytes):
        self.buf += pcm
        while len(self.buf) >= self.FRAME:
            fr, self.buf = self.buf[:self.FRAME], self.buf[self.FRAME:]
            self.on_frame(fr, self.vad.process_chunk(fr))

    def on_frame(self, fr: bytes, p: float):
        voiced = p >= (VAD_OFF if self.utt is not None else VAD_ON)
        db = 20 * math.log10(max(audioop.rms(fr, 2), 1) / 32768)
        for seq, v, n in ((self.flags, voiced, 94), (self.probs, p, 94), (self.dbs, db, 94), (self.ring, fr, 94)):
            seq.append(v)
            del seq[:-n]
        if self.utt is None:
            if self.speaking:
                e = self.echo
                e[0] += 1
                e[1] += db
                e[2] = max(e[2], p)
                e[3] += p >= VAD_ON
                if not voiced:  # the floor the AEC leaves between sounds; bursts are excluded so they do not raise it
                    self.echo_floor = db if self.echo_floor is None else \
                        EFLOOR_ALPHA * db + (1 - EFLOOR_ALPHA) * self.echo_floor
                    self.echo_floor_n += 1
            else:
                if self.echo[0]:
                    self.report_echo()
                if not self.turn_open:
                    self.quiet[0] += 1
                    self.quiet[1] += db
            self.pre.append(fr)
            del self.pre[:-(BARGE_MS // FRAME_MS + 8)]
            if self.vad_off or (self.group is not None and self.group.holding()):
                self.flags.clear()  # several stations are saying the reminder: this mic hears the others too
                return
            busy = self.speaking or self.turn_open
            answering = self.awaiting_answer()
            # over our own voice it takes BARGE_MS to tell the owner from the echo; while the answer is only being
            # made nothing plays, and the owner is heard at once (the answer waits, see `overlap`)
            need = (BARGE_MS if self.speaking else START_MS) // FRAME_MS
            recent = self.flags[-need:]
            if len(recent) == need and sum(recent) >= math.ceil(need * VOICED_RATIO):
                if self.speaking and not answering:
                    cand = sum(self.dbs[-need:]) / need
                    if self.echo_floor is not None and self.echo_floor_n >= need \
                            and cand < self.echo_floor + BARGE_FLOOR_MARGIN_DB:
                        return  # within the cancelled floor + margin: room noise or AEC leak, not the owner
                    self.barge_in(need)
                elif busy and not answering:
                    log.info("the owner talks while the answer is being made: it waits")
                    self.overlap = True
                elif answering and self.speaking:
                    log.info("the owner answers over the question")
                    self.echo = [0, 0.0, 0.0, 0]
                    self.send(frame(b"F"))           # stop the question; the agent's work goes on
                    self.play_end = 0.0
                    self.speaking = False
                self.utt = self.pre[-(need + LEAD_FRAMES):]
                self.utt_started = time.monotonic() - need * FRAME_MS / 1000
                self.voiced_in_utt = sum(recent)
                if self.group is not None:
                    self.group.claim(self, [d for d, v in zip(self.dbs[-need:], recent) if v])
                self.silence = 0
                self.heard_user = True
                self.touch()
                self.set_state("listening")
            return
        self.utt.append(fr)
        if voiced:
            self.voiced_in_utt += 1
            self.silence = 0
            self.touch()
            if self.overlap and self.voiced_in_utt * FRAME_MS >= BARGE_MS:
                self.overlap = False  # long enough to be the owner going on: the answer begun is cut now
                if self.turn_open:
                    self.barge_in(BARGE_MS // FRAME_MS)
            if self.group is not None:
                self.group.heard(self, db)
            if self.spec is not None:
                self.spec.cancel()  # the owner goes on: that transcript is already short of the end
                self.spec = None
                self.forget_rest()
        else:
            self.silence += 1
            if (self.silence == max(1, stt.SPEC_MS // FRAME_MS) and self.spec is None and self.http is not None
                    and stt.needs_text()):
                self.spec = self.spawn(stt.transcribe(self.http, b"".join(self.utt), quiet=self.secret_turn()))
        if len(self.utt) * FRAME_MS / 1000 > MAX_UTT_S:
            self.end_utterance()
        elif self.silence * FRAME_MS >= END_SILENCE_MS and self.rest is None:
            if self.spec is None:
                self.end_utterance()
            else:  # the transcript made ahead decides how long the pause has to be
                self.rest = self.spawn(self.wait_rest(self.utt, self.spec))

    def barge_in(self, need: int):
        log.info("barge-in by voice (%s, p %.2f, %.0f dBFS)", "speaking" if self.speaking else "thinking",
                 sum(self.probs[-need:]) / need, sum(self.dbs[-need:]) / need)
        self.dump("barge-" + ("speaking" if self.speaking else "thinking"), b"".join(self.ring), self.probs, self.dbs)
        self.echo = [0, 0.0, 0.0, 0]    # the last frames were the owner, not echo
        self.cancel_response(flush=True)

    def end_utterance(self):
        utt, voiced_n = self.utt, self.voiced_in_utt
        self.utt, self.pre, self.flags = None, [], []
        spec, self.spec = self.spec, None
        self.forget_rest()
        pcm = b"".join(utt)
        if voiced_n * FRAME_MS < MIN_SPEECH_MS:
            if stt.enabled() and self.http is not None and voiced_n * FRAME_MS >= stt.RESCUE_MIN_MS:
                self.spawn(self.rescue(pcm, voiced_n * FRAME_MS, spec))  # a short command, or noise?
                return
            if spec is not None:
                spec.cancel()
            log.info("ignored a short noise (%d ms voiced)", voiced_n * FRAME_MS)
            self.dump("noise", pcm)
            self.release()
            self.on_noise()
            return
        log.info("utterance %.1fs (%d ms voiced)", len(pcm) / 32000, voiced_n * FRAME_MS)
        self.dump("utt", pcm)
        self.on_utterance(pcm, None, spec)

    def release(self):
        """What made the answer wait was not the owner talking: it goes on."""
        if self.overlap:
            self.overlap = False
            if self.turn_open and not self.speaking:
                self.set_state("thinking")

    def forget_rest(self):
        if self.rest is not None and self.rest is not asyncio.current_task():
            self.rest.cancel()
        self.rest = None

    async def wait_rest(self, utt: list, spec: asyncio.Task):
        """END_SILENCE_MS of silence: the utterance ends, unless its transcript made ahead stops on a word that leads
        on ("включи свет в", "купи хлеб и"): then the owner is mid-thought and gets up to END_SILENCE_LONG_MS. A turn
        whose transcript is made ahead waits for it anyway (stt.needs_text), so waiting for it here costs nothing."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0, END_SILENCE_LONG_MS - END_SILENCE_MS) / 1000
        try:
            await asyncio.wait_for(asyncio.shield(spec), deadline - loop.time())
        except (asyncio.TimeoutError, Exception):
            pass
        if self.utt is not utt or self.spec is not spec:
            return  # the owner went on (or the utterance ended otherwise)
        if spec.done() and not spec.cancelled() and spec.exception() is None and stt.leads_on(spec.result()):
            log.info("the transcript so far leads on: waiting up to %d ms for the rest", END_SILENCE_LONG_MS)
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            if self.utt is not utt or self.spec is not spec:
                return
        self.end_utterance()

    async def rescue(self, pcm: bytes, voiced_ms: int, spec=None):
        """Too short for the VAD: the local STT decides whether it was words ("удали их") or noise."""
        text = await spec if spec is not None else await stt.transcribe(self.http, pcm, quiet=self.secret_turn())
        if stt.has_words(text) and not self.vad_off and not self.ending:
            log.info("short utterance kept (%d ms voiced): %s", voiced_ms, "(not logged)" if self.secret_turn() else repr(text))
            self.dump("utt", pcm)
            self.on_utterance(pcm, text)
        else:
            log.info("ignored a short noise (%d ms voiced)%s", voiced_ms,
                     f", heard {text!r}" if text and not self.secret_turn() else "")
            self.dump("noise", pcm)
            self.release()
            if not self.vad_off and not self.ending:
                self.on_noise()

    def report_echo(self):
        """What the station's echo canceller leaves of our own voice in the mic stream (it feeds us the cleaned signal)."""
        n, sdb, pmax, over = self.echo
        self.echo = [0, 0.0, 0.0, 0]
        if n < 15:
            return
        q = self.quiet
        floor = f"{q[1] / q[0]:.0f}" if q[0] > 20 else "n/a"
        log.info("echo check: %.1fs of our own audio, mic mean %.0f dBFS (room floor %s), max VAD p %.2f, %d frames over the threshold",
                 n * FRAME_MS / 1000, sdb / n, floor, pmax, over)

    def dump(self, kind: str, pcm: bytes, probs=None, dbs=None):
        """Debug aid (LIVE_DUMP_DIR): keep the sound that made a decision, so it can be listened to and re-run."""
        if not DUMP_DIR or self.secret_turn():
            return
        try:
            d = Path(DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%H%M%S')}-{kind}"
            (d / f"{name}.wav").write_bytes(wav16k(pcm))
            if probs is not None:
                (d / f"{name}.txt").write_text("\n".join(f"{p:.2f} {b:.0f}" for p, b in zip(probs, dbs)) + "\n")
            old = sorted(d.glob("*.wav"), key=lambda f: f.stat().st_mtime)[:-DUMP_KEEP]
            for f in old:
                f.unlink(missing_ok=True)
                f.with_suffix(".txt").unlink(missing_ok=True)
        except OSError as e:
            log.warning("dump failed: %r", e)
