"""Station <-> bridge framing and the connection object backends work with.

Both ways: [u8 type][u32 LE length][payload]; the payload of a frame is at most MAX_FRAME bytes.
  station -> bridge: H hello (JSON: wake_word, station, protocol)  A mic s16le 16 kHz mono (echo-cancelled)
                     D playback drained (everything sent so far has been played)  E end (reason)
  bridge -> station: A answer s16le 24 kHz mono  F flush (drop what is queued)  S state (listening|thinking|speaking)
                     X end of the dialog (reason)  R hand-off: the bridge does not take this dialog, the station runs
                     its own (Home Assistant) pipeline and replays the last ~1.5 s of the microphone into it
"""
import asyncio
import json
import struct

MAX_FRAME = 8 << 20


def frame(t: bytes, payload: bytes = b"") -> bytes:
    return t + struct.pack("<I", len(payload)) + payload


async def read_frame(reader):
    """(type, payload) or None when the peer is gone or sends nonsense."""
    try:
        hdr = await reader.readexactly(5)
        n = struct.unpack("<I", hdr[1:])[0]
        if n > MAX_FRAME:
            return None
        return hdr[:1], (await reader.readexactly(n) if n else b"")
    except (asyncio.IncompleteReadError, ConnectionError):
        return None


def parse_hello(payload: bytes) -> dict:
    try:
        v = json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


class Station:
    """One connected station, after its hello."""

    def __init__(self, reader, writer, hello: dict):
        self.r, self.w, self.hello = reader, writer, hello
        self.peer = writer.get_extra_info("peername")
        self.sent = False  # something went to the station: from then on a failure ends the dialog instead of handing it off

    @property
    def name(self) -> str:
        return str(self.hello.get("station") or "")

    @property
    def closing(self) -> bool:
        return self.w.is_closing()

    def send(self, data: bytes):
        if not self.closing:
            self.sent = True
            self.w.write(data)

    def send_frame(self, t: bytes, payload: bytes = b""):
        self.send(frame(t, payload))

    async def read(self):
        return await read_frame(self.r)

    async def drain(self):
        try:
            await self.w.drain()
        except Exception:
            pass

    async def finish(self, t: bytes = b"X", reason: str = ""):
        """Last frame (X end / R hand-off), flush and close."""
        self.send_frame(t, reason.encode())
        await self.drain()
        self.w.close()
