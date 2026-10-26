#!/usr/bin/env python3
"""Live bridge: pulsar stations <-> realtime model backends.

One TCP connection = one dialog. The station says hello (its name, the wake word); the bridge picks a backend for it
from routes.json (see livebridge/routes.py) and the backend owns the dialog from there. A backend that cannot start
hands the dialog back to the station (frame R): it then runs its own Home Assistant pipeline. Protocol:
livebridge/wire.py and docs/live.md.
"""
import asyncio
import logging
import os

import aiohttp

from livebridge import backends
from livebridge.routes import Routes
from livebridge.util import HERE
from livebridge.wire import Station, parse_hello, read_frame

log = logging.getLogger("live")

LISTEN = os.environ.get("LIVE_LISTEN", "0.0.0.0:10700")
ROUTES = os.environ.get("LIVE_ROUTES") or str(HERE / "routes.json")
HELLO_TIMEOUT = 10.0


async def dialog(reader, writer, http: aiohttp.ClientSession, routes: Routes):
    f = await asyncio.wait_for(read_frame(reader), HELLO_TIMEOUT)
    if not f or f[0] != b"H":
        writer.close()
        return
    station = Station(reader, writer, parse_hello(f[1]))
    name, options = routes.pick(station.name)
    log.info("dialog from %s (%r): backend %s", station.peer, station.name, name)
    try:
        await backends.load(name).run(station, http, options)
    except Exception as e:
        # Nothing said to the station yet: it can still run its own pipeline. Otherwise the dialog is over.
        if not station.sent and routes.fallback != "none" and not station.closing:
            log.warning("backend %s could not start (%s), falling back to %s", name, e, routes.fallback)
            try:
                await backends.load(routes.fallback).run(station, http, {})
                return
            except Exception:
                log.exception("fallback %s failed too", routes.fallback)
        else:
            log.exception("dialog crashed")
        await station.finish(b"X", "bridge error")
    finally:
        if not writer.is_closing():
            writer.close()


async def serve(listen: str, routes: Routes, http: aiohttp.ClientSession):
    host, port = listen.rsplit(":", 1)

    async def handle(reader, writer):
        try:
            await dialog(reader, writer, http, routes)
        except Exception:
            log.exception("dialog failed")
            writer.close()

    return await asyncio.start_server(handle, host, int(port))


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(keepalive_timeout=120))
    routes = Routes(ROUTES)
    server = await serve(LISTEN, routes, http)
    log.info("live bridge on %s, routes %s (default backend %s)", LISTEN, ROUTES, routes.default)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
