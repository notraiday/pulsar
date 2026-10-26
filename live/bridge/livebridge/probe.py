"""Time the home agent the way a Live dialog reaches it, from the bridge's own environment:

    python -m livebridge.probe [--agent ha|hermes|both] [-n 3] [--satellite assist_satellite.x] "Который час?"

The agent really runs the request: ask only questions that change nothing (time, weather, a sensor), never "turn
on ..." or "create ...". Each request of one run is sent alone, in the same conversation, like a dialog would.
"""
import argparse
import asyncio
import logging
import statistics
import time

import aiohttp

from .agent import KINDS, Agent


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--agent", default="both", choices=(*KINDS, "both"))
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--satellite", help="ha_satellite_id to ask as (area context)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    async with aiohttp.ClientSession() as http:
        for kind in KINDS if a.agent == "both" else (a.agent,):
            agent = Agent(http, {"agent": kind, "ha_satellite_id": a.satellite})
            try:
                agent.check()
            except Exception as e:
                print(f"{kind}: not usable: {e}")
                continue
            times = []
            for _ in range(a.n):
                t0 = time.monotonic()
                try:
                    answer = await agent.ask(a.question)
                except Exception as e:
                    print(f"{kind}: failed after {time.monotonic() - t0:.2f}s: {e}")
                    continue
                times.append(time.monotonic() - t0)
                print(f"{kind}: {times[-1]:.2f}s  {answer[:120]!r}")
            if times:
                print(f"{kind}: median {statistics.median(times):.2f}s over {len(times)} ({agent.describe()})")


if __name__ == "__main__":
    asyncio.run(main())
