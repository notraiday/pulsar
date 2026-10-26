"""Reminders (a dialog the house starts) on one and on two stations: one synthesis played on all, nobody listens while
several of them speak, the station that hears the owner loudest answers."""
import asyncio
import json

import aiohttp
from aiohttp import web

import bridge
from conftest import LOUDER, QUIET, frames_until, serve_app, station
from livebridge import agent, reminders, session, wire
from livebridge.backends import pipeline
from livebridge.routes import Routes

MID = (b"\xdc\x05\x24\xfa") * 256   # +-1500: voiced for the fake VAD, quieter than LOUDER
SPEECH = b"\x30\x00" * 4800          # what the fake TTS says (0.2 s)


def fake_openrouter(seen):
    async def chat(req):
        seen["chat"].append(await req.json())
        ev = {"choices": [{"delta": {"content": "Хорошо, напомню позже."}}]}
        return web.Response(text="data: " + json.dumps(ev) + "\n\ndata: [DONE]\n\n", content_type="text/event-stream")

    async def speech(req):
        seen["tts"].append((await req.json())["input"])
        return web.Response(body=SPEECH)

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", lambda req: web.json_response({}))
    return app


async def talk(w, frames):
    """Mic frames in real time (32 ms each), as a station sends them."""
    for f in frames:
        w.write(wire.frame(b"A", f))
        await asyncio.sleep(0.032)


def setup(monkeypatch):
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(agent, "AGENT", "hermes")
    monkeypatch.setattr(session, "IDLE_S", 0.5)
    monkeypatch.setattr(session, "FIRST_IDLE_S", 3)
    monkeypatch.setattr(reminders, "GATHER_S", 0.05)
    monkeypatch.setattr(reminders, "DECIDE_S", 0.15)


async def with_bridge(tmp_path, monkeypatch, seen, scenario):
    runner, port = await serve_app(fake_openrouter(seen))
    monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{port}")
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"default": "pipeline"}))
    try:
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            try:
                return await scenario(server.sockets[0].getsockname()[1])
            finally:
                server.close()
    finally:
        await runner.cleanup()


def test_two_stations_say_it_together_and_the_louder_one_answers(tmp_path, monkeypatch, fake_vad):
    setup(monkeypatch)
    seen = {"chat": [], "tts": []}
    text = "[emo:good] Пора выключить чайник!"

    async def scenario(port):
        (ra, wa), (rb, wb) = await asyncio.gather(
            station(port, {"station": "midi", "protocol": 1, "reminder": text}),
            station(port, {"station": "mini", "protocol": 1, "reminder": text}))
        got_a, got_b = await asyncio.gather(frames_until(ra, {b"A"}), frames_until(rb, {b"A"}))
        # Each mic hears the other station saying it: that is no barge-in and no question.
        await asyncio.gather(talk(wa, [LOUDER] * 20), talk(wb, [LOUDER] * 20))
        for w in (wa, wb):
            w.write(wire.frame(b"D"))
        await asyncio.sleep(0.4)
        # The owner answers; the mini starts hearing it first, but the midi hears it louder.
        end_b = asyncio.create_task(frames_until(rb, {b"X"}, timeout=8))
        await asyncio.gather(talk(wb, [MID] * 20 + [QUIET] * 25),
                             talk(wa, [QUIET, QUIET] + [LOUDER] * 18 + [QUIET] * 25))
        end_b = await end_b
        rest_a = await frames_until(ra, {b"A"}, timeout=5)
        wa.write(wire.frame(b"D"))
        rest_a += await frames_until(ra, {b"X"}, timeout=5)
        return got_a, got_b, end_b, rest_a

    got_a, got_b, end_b, rest_a = asyncio.run(with_bridge(tmp_path, monkeypatch, seen, scenario))
    # one synthesis (tags stripped), the same audio on both
    assert seen["tts"][0] == "Пора выключить чайник!"
    assert [p for t, p in got_a if t == b"A"] == [p for t, p in got_b if t == b"A"] != []
    # the quieter station leaves, the louder one answers in the reminder's context
    assert end_b[-1] == (b"X", b"another station answers")
    assert len(seen["chat"]) == 1
    assert "Пора выключить чайник!" in seen["chat"][0]["messages"][0]["content"]
    assert seen["tts"][1:] == ["Хорошо, напомню позже."]
    assert b"F" not in [t for t, _ in got_a + rest_a]
    assert rest_a[-1] == (b"X", b"idle")


def test_one_station_can_talk_over_its_reminder(tmp_path, monkeypatch, fake_vad):
    setup(monkeypatch)
    seen = {"chat": [], "tts": []}

    async def scenario(port):
        r, w = await station(port, {"station": "midi", "protocol": 1, "reminder": "Пора выключить чайник!"})
        got = await frames_until(r, {b"A"})
        await talk(w, [LOUDER] * 20 + [QUIET] * 25)   # "да знаю" while it still plays: no D sent
        got += await frames_until(r, {b"F"}, timeout=5)
        got += await frames_until(r, {b"A"}, timeout=5)
        return got

    got = asyncio.run(with_bridge(tmp_path, monkeypatch, seen, scenario))
    types = [t for t, _ in got]
    assert b"F" in types                              # cut short
    assert len(seen["chat"]) == 1                     # and answered


def test_gpt_live_leaves_reminders_to_the_station(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"default": "gpt-live"}))

    async def go():
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1], {"station": "x", "reminder": "Чайник!"})
            got = await frames_until(r, {b"R", b"X"})
            server.close()
            return got
    assert asyncio.run(go()) == [(b"R", b"")]
