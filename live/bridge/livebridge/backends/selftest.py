"""No-model check of the station side ("test"): a tone, then how loud the mic hears it (echo cancellation)."""
import asyncio
import logging
import math
import struct
import time

from ..wire import frame

log = logging.getLogger("live")


async def run(station, http, options):
    rate = 24000
    tone = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(rate * 2))
    await asyncio.sleep(1.0)
    station.send(frame(b"S", b"speaking"))
    for i in range(0, len(tone), 4800):
        station.send(frame(b"A", tone[i:i + 4800]))
    t0 = time.monotonic()
    drained_at, levels = None, []
    while time.monotonic() - t0 < 7:
        try:
            f = await asyncio.wait_for(station.read(), 5)
        except asyncio.TimeoutError:
            break
        if f is None:
            break
        t, p = f
        if t == b"D" and drained_at is None:
            drained_at = time.monotonic() - t0
            station.send(frame(b"S", b"listening"))
        elif t == b"A" and p:
            v = struct.unpack("<%dh" % (len(p) // 2), p)
            rms = math.sqrt(sum(x * x for x in v) / len(v))
            levels.append((time.monotonic() - t0, 20 * math.log10(max(rms, 1) / 32768)))

    def avg(a, b):
        xs = [lv for t, lv in levels if a <= t < b]
        return sum(xs) / len(xs) if xs else float("nan")
    d = drained_at or 2.0
    log.info("selftest: drained at %.2fs; mic dBFS while playing %.1f, after %.1f (%d chunks)",
             d, avg(0.2, d - 0.1), avg(d + 0.5, 7), len(levels))
    await station.finish(b"X", "selftest done")
