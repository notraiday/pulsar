import asyncio
import audioop
import sys
from pathlib import Path

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livebridge import session, wire  # noqa: E402


class FakeVad:
    """Loud frame = speech: the tests drive the turn detector without real speech audio."""

    def process_chunk(self, fr: bytes) -> float:
        return 0.9 if audioop.rms(fr, 2) > 1000 else 0.0


@pytest.fixture
def fake_vad(monkeypatch):
    monkeypatch.setattr(session, "SileroVoiceActivityDetector", FakeVad)


LOUD = (b"\xe8\x03\x18\xfc") * 256     # 1024 bytes, rms ~1000..: alternating +1000 / -1000
LOUDER = (b"\xd0\x07\x30\xf8") * 256   # +2000 / -2000
QUIET = b"\0" * 1024


async def serve_app(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def station(port: int, hello: dict | None = None):
    """A fake station connection: (reader, writer) after sending its hello."""
    import json
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(wire.frame(b"H", json.dumps(hello if hello is not None else {"station": "test", "protocol": 1}).encode()))
    return r, w


async def frames_until(r, stop: set[bytes], timeout: float = 10.0) -> list[tuple[bytes, bytes]]:
    """Frames read until one of the types in `stop` arrives (included) or the peer closes."""
    out = []
    async with asyncio.timeout(timeout):
        while True:
            f = await wire.read_frame(r)
            if f is None:
                return out
            out.append(f)
            if f[0] in stop:
                return out
