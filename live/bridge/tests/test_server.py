"""The bridge's routing and hand-off, over real TCP with a fake station."""
import asyncio
import json

import aiohttp
import pytest

import bridge
from conftest import frames_until, station
from livebridge import agent
from livebridge.routes import Routes
from livebridge.backends import pipeline, gpt_live


def run_bridge(tmp_path, routes_doc, scenario, monkeypatch=None):
    p = tmp_path / "routes.json"
    p.write_text(json.dumps(routes_doc))

    async def go():
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(p), http)
            port = server.sockets[0].getsockname()[1]
            try:
                return await scenario(port)
            finally:
                server.close()
                await server.wait_closed()
    return asyncio.run(go())


def test_handoff_backend_answers_with_r(tmp_path):
    async def scenario(port):
        r, w = await station(port)
        return await frames_until(r, {b"R", b"X"})
    assert run_bridge(tmp_path, {"default": "handoff"}, scenario) == [(b"R", b"")]


def test_routing_is_per_station_name(tmp_path):
    async def scenario(port):
        out = {}
        for name in ("a", "b"):
            r, w = await station(port, {"station": name})
            out[name] = await frames_until(r, {b"R", b"X"})
        return out
    got = run_bridge(tmp_path, {"default": "handoff", "stations": {"b": "selftest-missing"}, "fallback": "none"},
                     scenario)
    assert got["a"] == [(b"R", b"")]
    assert got["b"] == [(b"X", b"bridge error")]


def test_unknown_backend_falls_back_to_handoff(tmp_path):
    async def scenario(port):
        r, w = await station(port)
        return await frames_until(r, {b"R", b"X"})
    assert run_bridge(tmp_path, {"default": "no-such-backend"}, scenario) == [(b"R", b"")]


def test_a_backend_that_cannot_start_hands_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OR_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def scenario(port):
        out = []
        for _ in range(2):
            r, w = await station(port)
            out.append(await frames_until(r, {b"R", b"X"}))
        return out
    assert run_bridge(tmp_path, {"default": "pipeline", "stations": {"*": "gpt-live"}}, scenario) == [
        [(b"R", b"")]] * 2
    assert run_bridge(tmp_path, {"default": "pipeline"}, scenario) == [[(b"R", b"")]] * 2


def test_a_dialog_without_a_way_to_the_agent_is_handed_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(agent, "AGENT", "ha")
    monkeypatch.setattr(agent, "HA_TOKEN", "")

    async def scenario(port):
        r, w = await station(port)
        return await frames_until(r, {b"R", b"X"})
    assert run_bridge(tmp_path, {"default": "pipeline"}, scenario) == [(b"R", b"")]


def test_connections_without_a_hello_are_closed(tmp_path):
    async def scenario(port):
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(b"A\x02\0\0\0xx")
        return await asyncio.wait_for(r.read(), 5)
    assert run_bridge(tmp_path, {"default": "handoff"}, scenario) == b""
