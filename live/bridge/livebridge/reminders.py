"""Dialogs that the house starts: a reminder (Home Assistant's assist_satellite.start_conversation, which a station
with a bridge turns into a Live dialog whose hello carries the text) is spoken on every station that got it, and the
one that hears the owner loudest keeps the dialog.

Stations that open a dialog with the same reminder text within JOIN_S form a group. The group synthesises the text
once and streams the same audio to all of them (one joining late gets it from where the others are, so they stay in
step). While more than one station is speaking it, nobody listens: each mic hears the other stations' voice, which
its echo canceller cannot remove. Then every station listens; the first utterance opens a DECIDE_S window, the station
whose mic heard it loudest (mean dBFS of its voiced frames until the decision) answers, the others end their
dialog.
"""
import asyncio
import logging
import os
import time

from .wire import frame

log = logging.getLogger("live")

JOIN_S = float(os.environ.get("LIVE_REMINDER_JOIN_S", "3"))        # stations of one reminder open within this
GATHER_S = float(os.environ.get("LIVE_REMINDER_GATHER_S", "0.35"))  # wait for the others before speaking
DECIDE_S = float(os.environ.get("LIVE_REMINDER_DECIDE_S", "0.35"))  # who heard the owner loudest, after the first
GUARD_S = 0.3                                                       # after the last station went quiet
PCM_RATE = 48000                                                    # bytes per second of 24 kHz s16 mono

_groups: dict[str, "Group"] = {}


def _key(text: str) -> str:
    return " ".join(text.lower().split())


class Group:
    def __init__(self, text: str):
        self.text = text
        self.created = time.monotonic()
        self.members: list = []
        self.cut: set = set()       # members that talked over it alone (barge-in): no more of its audio
        self.pcm = bytearray()
        self.t_first = None         # when its first audio went out
        self.sending = True
        self.quiet_since = 0.0      # when a member last reported its playback drained
        self.claims: dict = {}      # member -> [sum dBFS, frames] of the voiced frames of the utterance it heard
        self.winner = None
        self._decide = None
        self.task = None

    # ---- membership ----
    def add(self, s):
        self.members.append(s)
        s.group = self
        if self.t_first is not None and self.pcm:
            # Joined while the others speak: start where they are.
            offset = int((time.monotonic() - self.t_first) * PCM_RATE) // 2 * 2
            if offset < len(self.pcm):
                self._send(s, bytes(self.pcm[offset:]))

    def leave(self, s):
        if s in self.members:
            self.members.remove(s)
        self.claims.pop(s, None)
        if not self.members:
            if self.task and not self.task.done():
                self.task.cancel()
            if _groups.get(_key(self.text)) is self:
                del _groups[_key(self.text)]

    # ---- speaking ----
    def _send(self, s, pcm: bytes):
        if s in self.cut or s.st.closing:
            return
        s.speaking = True
        s.touch()
        s.set_state("speaking")
        for i in range(0, len(pcm), 4800):
            s.send(frame(b"A", pcm[i:i + 4800]))

    async def speak(self, chunks):
        """`chunks`: async iterator of 24 kHz PCM for the text (the TTS of the member that opened the group)."""
        try:
            await asyncio.sleep(GATHER_S)
            log.info("reminder to %d station(s): %r", len(self.members), self.text[:120])
            async for pcm in chunks:
                if self.t_first is None:
                    self.t_first = time.monotonic()
                self.pcm += pcm
                for s in list(self.members):
                    self._send(s, pcm)
                    await s.st.drain()
            if not self.pcm:
                log.warning("reminder: nothing to say (TTS gave no audio)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("reminder: TTS failed: %r", e)
        finally:
            self.sending = False

    def drained(self):
        self.quiet_since = time.monotonic()

    def holding(self) -> bool:
        """Several stations are (or were just) saying it: no one listens yet."""
        if len(self.members) < 2:
            return False
        return (self.sending or any(s.speaking for s in self.members)
                or time.monotonic() - self.quiet_since < GUARD_S)

    def cut_off(self, s):
        self.cut.add(s)

    # ---- who answers ----
    def claim(self, s, voiced_dbs: list):
        """`s` heard the owner start talking; `voiced_dbs`: levels of the voiced frames so far."""
        if self.winner is not None:
            return
        c = self.claims.setdefault(s, [0.0, 0])
        c[0] += sum(voiced_dbs)
        c[1] += len(voiced_dbs)
        if self._decide is None:
            self._decide = asyncio.get_running_loop().call_later(DECIDE_S, self.decide)

    def heard(self, s, db: float):
        """One more voiced frame of the utterance `s` claimed, while the group has not decided."""
        c = self.claims.get(s)
        if c is not None and self.winner is None:
            c[0] += db
            c[1] += 1

    def level(self, s) -> float:
        total, n = self.claims[s]
        return total / n if n else -200.0

    def decide(self):
        if self.winner is not None or not self.claims:
            return
        if self._decide is not None:
            self._decide.cancel()
        self.winner = max(self.claims, key=self.level)
        log.info("reminder: %s answers (%s)", self.winner.st.name or self.winner.st.peer,
                 ", ".join(f"{m.st.name or m.st.peer} {self.level(m):.1f} dBFS" for m in self.claims))
        for s in list(self.members):
            if s is not self.winner:
                s.lose("another station answers")

    def won(self, s) -> bool:
        """This member may answer what it heard (deciding now if the window is still open)."""
        if self.winner is None and s in self.claims:
            self.decide()
        return self.winner is None or self.winner is s


def join(s, text: str, chunks_for) -> Group:
    """Put dialog `s` into the group of this reminder (a new one if there is none from the last JOIN_S).
    `chunks_for(text)`: the TTS of `s`, used if `s` opens the group."""
    key = _key(text)
    g = _groups.get(key)
    if g is None or time.monotonic() - g.created > JOIN_S:
        g = Group(text)
        _groups[key] = g
        g.task = asyncio.create_task(g.speak(chunks_for(text)))
    g.add(s)
    return g
