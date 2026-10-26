"""The pipeline backend without any key: a fake OpenRouter + a fake home agent on loopback."""
import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web

import bridge
from conftest import LOUDER, QUIET, fake_ha, frames_until, serve_app, station
from livebridge import agent, session, wire
from livebridge.backends import pipeline
from livebridge.routes import Routes

# ---- llm_events: hedging against a stalled provider ----


def chunk(**delta):
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()


EVS = [chunk(role="assistant"), chunk(content="Привет"), chunk(content=" мир."), b"data: [DONE]\n\n"]


class Resp:
    def __init__(self, status, lines, first_delay):
        self.status, self.lines, self.d = status, lines, first_delay

    async def text(self):
        return "boom"

    @property
    def content(self):
        async def gen():
            await asyncio.sleep(self.d)
            for line in self.lines:
                yield line
        return gen()


class CM:
    def __init__(self, r):
        self.r = r

    async def __aenter__(self):
        return self.r

    async def __aexit__(self, *a):
        return False


class W:
    def is_closing(self):
        return False

    def write(self, d):
        pass

    def get_extra_info(self, k):
        return None


class Http:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def post(self, *a, **k):
        spec = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return CM(Resp(*spec))


def hedge(monkeypatch, script):
    monkeypatch.setattr(pipeline, "LLM_HEDGE_S", 0.2)
    http = Http(script)
    d = pipeline.Dialog(wire.Station(None, W(), {}), http)

    async def go():
        text, err = "", None
        try:
            async for ev in d.llm_events({}, {}):
                for ch in ev.get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
        except RuntimeError as e:
            err = str(e)
        return text, err
    text, err = asyncio.run(go())
    return http.calls, text, err


def test_normal_request_is_not_hedged(monkeypatch, fake_vad):
    assert hedge(monkeypatch, [(200, EVS, 0.0)]) == (1, "Привет мир.", None)


def test_a_stalled_provider_loses_to_the_second_request(monkeypatch, fake_vad):
    t0 = time.monotonic()
    assert hedge(monkeypatch, [(200, EVS, 20.0), (200, EVS, 0.0)]) == (2, "Привет мир.", None)
    assert time.monotonic() - t0 < 5


def test_http_error_is_retried_once(monkeypatch, fake_vad):
    assert hedge(monkeypatch, [(500, [], 0.0), (200, EVS, 0.0)]) == (2, "Привет мир.", None)


def test_two_failures_surface(monkeypatch, fake_vad):
    calls, text, err = hedge(monkeypatch, [(500, [], 0.0), (500, [], 0.0)])
    assert calls == 2 and text == "" and err


# ---- a whole dialog ----


def sse(*events):
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"


def fake_openrouter(seen):
    async def chat(req):
        body = await req.json()
        seen["chat"].append(body)
        if any(m.get("role") == "tool" for m in body["messages"]):
            ev = [{"choices": [{"delta": {"content": "Готово."}}]}]
        else:
            ev = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "ask_hermes", "arguments": json.dumps({"request": "какая погода"}, ensure_ascii=False)}}]}}]}]
        return web.Response(text=sse(*ev), content_type="text/event-stream")

    async def speech(req):
        body = await req.json()
        seen["tts"].append(body["input"])
        return web.Response(body=b"\x10\x00" * 4800)

    async def key(req):
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", key)
    return app


def fake_hermes(seen):
    async def chat(req):
        seen["hermes"].append((await req.json())["messages"][-1]["content"])
        return web.json_response({"choices": [{"message": {"content": "Плюс десять, облачно."}}]})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app


@pytest.mark.parametrize("way", ["ha", "hermes"])
def test_a_spoken_question_goes_through_the_agent_and_is_answered_aloud(tmp_path, monkeypatch, fake_vad, way):
    seen = {"chat": [], "tts": [], "hermes": [], "ha": []}
    monkeypatch.setattr(session, "IDLE_S", 0.4)
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(agent, "AGENT", way)

    async def go():
        or_runner, or_port = await serve_app(fake_openrouter(seen))
        hm_runner, hm_port = await serve_app(fake_ha(seen["ha"]) if way == "ha" else fake_hermes(seen))
        monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{or_port}")
        monkeypatch.setattr(agent, "HERMES_URL", f"http://127.0.0.1:{hm_port}/v1/chat/completions")
        monkeypatch.setattr(agent, "HA_URL", f"http://127.0.0.1:{hm_port}")
        monkeypatch.setattr(agent, "HA_TOKEN", "ha-token")
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline", "stations": {"kitchen": {
            "backend": "pipeline", "options": {"ha_satellite_id": "assist_satellite.kitchen"}}}}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            port = server.sockets[0].getsockname()[1]
            r, w = await station(port, {"station": "kitchen", "protocol": 1})
            for _ in range(20):                       # 640 ms of "speech", then a pause
                w.write(wire.frame(b"A", LOUDER))
            for _ in range(25):
                w.write(wire.frame(b"A", QUIET))
            got = await frames_until(r, {b"A"})
            w.write(wire.frame(b"D"))                 # the station has played it
            got += await frames_until(r, {b"X"})
            server.close()
            await or_runner.cleanup()
            await hm_runner.cleanup()
            return got
    got = asyncio.run(go())
    types = [t for t, _ in got]
    states = [p.decode() for t, p in got if t == b"S"]
    assert states[:3] == ["listening", "thinking", "speaking"]
    assert b"A" in types and got[-1] == (b"X", b"idle")
    # the agent got the request with the router mark (through HA: as the station's satellite); its short answer was
    # spoken as it was, in one model round
    if way == "ha":
        assert [(b["text"], b.get("satellite_id")) for _, b in seen["ha"]] == [
            ("@@LIVE@@ какая погода", "assist_satellite.kitchen")]
        assert seen["hermes"] == []
    else:
        assert seen["hermes"] == ["@@LIVE@@ какая погода"] and seen["ha"] == []
    assert seen["tts"] == ["Плюс десять, облачно."]
    assert len(seen["chat"]) == 1
    # the question went to the model as audio
    assert seen["chat"][0]["messages"][1]["content"][0]["type"] == "input_audio"


def test_end_dialog_closes_the_dialog_at_once(tmp_path, monkeypatch, fake_vad):
    """"Ок, спасибо": the model calls end_dialog alone; no second model round, nothing spoken, X right away."""
    seen = {"chat": [], "tts": []}
    monkeypatch.setattr(session, "IDLE_S", 30)        # the end must not come from the idle timer
    monkeypatch.setattr(session, "FIRST_IDLE_S", 30)
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(agent, "AGENT", "hermes")

    async def chat(req):
        seen["chat"].append(await req.json())
        ev = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "e1", "function": {
            "name": "end_dialog", "arguments": "{}"}}]}}]}]
        return web.Response(text=sse(*ev), content_type="text/event-stream")

    async def speech(req):
        seen["tts"].append((await req.json())["input"])
        return web.Response(body=b"\x10\x00" * 4800)

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", lambda req: web.json_response({}))

    async def go():
        runner, port = await serve_app(app)
        monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{port}")
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])
            for _ in range(20):
                w.write(wire.frame(b"A", LOUDER))
            for _ in range(25):
                w.write(wire.frame(b"A", QUIET))
            t0 = time.monotonic()
            got = await frames_until(r, {b"X"}, timeout=5)
            took = time.monotonic() - t0
            server.close()
            await runner.cleanup()
            return got, took
    got, took = asyncio.run(go())
    assert got[-1] == (b"X", b"goodbye")
    assert b"A" not in [t for t, _ in got] and seen["tts"] == []
    assert len(seen["chat"]) == 1
    assert took < 3
