"""The gpt-live backend against a fake server that follows OpenAI's GPT-Live WebSocket documentation.
It says nothing about the real service."""
import asyncio
import base64
import json

import aiohttp
from aiohttp import web

import bridge
from conftest import LOUDER, frames_until, serve_app, station
from livebridge import hermes, session, wire
from livebridge.backends import gpt_live
from livebridge.routes import Routes

PCM_OUT = b"\x20\x00" * 2400


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def fake_live(seen, reject=False):
    async def sessions(req):
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        seen["headers"] = dict(req.headers)
        start = json.loads((await ws.receive()).data)
        seen["start"] = start
        if reject:
            await ws.send_json({"type": "error", "error": {"message": "nope"}})
            await ws.close()
            return ws
        await ws.send_json({"type": "session.started", "session": {"id": "s1"}})
        asked = False
        async for msg in ws:
            ev = json.loads(msg.data)
            seen.setdefault("events", []).append(ev["type"])
            if ev["type"] == "session.input_audio.append":
                seen["mic"] = seen.get("mic", 0) + len(base64.b64decode(ev["audio"]))
                if not asked and seen["mic"] > 3 * 1536:
                    asked = True
                    for part in ("какая ", "погода"):
                        await ws.send_json({"type": "session.input_transcript.delta", "delta": part})
                    await ws.send_json({"type": "session.delegation.created", "delegation": {
                        "id": "item_1", "type": "delegation", "target": "client"}})
            elif ev["type"] == "session.commentary.append":
                seen["commentary"] = ev
                await ws.send_json({"type": "session.output_transcript.delta", "delta": "Плюс десять."})
                await ws.send_json({"type": "session.output_audio.delta", "delta": b64(PCM_OUT)})
            elif ev["type"] == "session.close":
                await ws.send_json({"type": "session.closed", "usage": {}})
                break
        return ws

    app = web.Application()
    app.router.add_get("/v1/live/sessions", sessions)
    return app


def fake_hermes(seen):
    async def chat(req):
        seen.setdefault("hermes", []).append((await req.json())["messages"][-1]["content"])
        return web.json_response({"choices": [{"message": {"content": "Плюс десять, облачно."}}]})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app


async def with_bridge(tmp_path, monkeypatch, seen, scenario, reject=False, key="sk-test"):
    if key:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    lv_runner, lv_port = await serve_app(fake_live(seen, reject))
    hm_runner, hm_port = await serve_app(fake_hermes(seen))
    monkeypatch.setattr(gpt_live, "URL", f"ws://127.0.0.1:{lv_port}/v1/live/sessions")
    monkeypatch.setattr(hermes, "HERMES_URL", f"http://127.0.0.1:{hm_port}/v1/chat/completions")
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"default": "gpt-live", "stations": {"kitchen": {
        "backend": "gpt-live", "options": {"voice": "cedar"}}}}))
    try:
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            try:
                return await scenario(server.sockets[0].getsockname()[1])
            finally:
                server.close()
    finally:
        await lv_runner.cleanup()
        await hm_runner.cleanup()


def test_dialog_with_client_delegation(tmp_path, monkeypatch, fake_vad):
    seen = {}

    async def scenario(port):
        r, w = await station(port, {"station": "kitchen", "protocol": 1})
        for _ in range(8):
            w.write(wire.frame(b"A", b"\x01\x00" * 512))
        got = await frames_until(r, {b"A"})
        w.write(wire.frame(b"E", b"button"))
        return got + await frames_until(r, {b"X"})
    got = asyncio.run(with_bridge(tmp_path, monkeypatch, seen, scenario))

    s = seen["start"]
    assert s["type"] == "session.start"
    assert s["session"]["model"] == "gpt-live-1" and s["session"]["delegation"] == {"type": "client"}
    assert s["session"]["audio"] == {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "cedar"}}
    assert "делегируй" in s["session"]["instructions"]
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    # 1024 bytes of 16 kHz mic become 1536 bytes of 24 kHz
    assert seen["mic"] >= 3 * 1500
    # the agent was asked what the owner said, with the router mark; its answer went back on the delegation
    assert seen["hermes"] == ["@@LIVE@@ какая погода"]
    assert seen["commentary"]["delegation_id"] == "item_1"
    assert seen["commentary"]["content"] == "Плюс десять, облачно."
    # the model's voice reached the station untouched, with the UI states around it
    assert [p for t, p in got if t == b"A"] == [PCM_OUT]
    assert [p.decode() for t, p in got if t == b"S"][:3] == ["listening", "thinking", "speaking"]
    assert got[-1] == (b"X", b"station: button")
    assert "session.close" in seen["events"]


def test_missing_key_hands_the_dialog_off(tmp_path, monkeypatch, fake_vad):
    async def scenario(port):
        r, w = await station(port)
        return await frames_until(r, {b"R", b"X"})
    assert asyncio.run(with_bridge(tmp_path, monkeypatch, {}, scenario, key="")) == [(b"R", b"")]


def test_rejected_session_hands_the_dialog_off(tmp_path, monkeypatch, fake_vad):
    async def scenario(port):
        r, w = await station(port)
        return await frames_until(r, {b"R", b"X"})
    assert asyncio.run(with_bridge(tmp_path, monkeypatch, {}, scenario, reject=True)) == [(b"R", b"")]


# ---- barge-in: the audio the model had already produced is dropped until it pauses ----


class Out:
    def __init__(self):
        self.data = b""

    def is_closing(self):
        return False

    def write(self, d):
        self.data += d

    def get_extra_info(self, k):
        return None


def test_audio_in_flight_after_a_barge_in_is_dropped_until_the_model_pauses(monkeypatch, fake_vad):
    monkeypatch.setattr(gpt_live, "DROP_GAP_S", 0.1)
    out = Out()
    g = gpt_live.GptLive(wire.Station(None, out, {}), None)

    async def go():
        g.play(b"\x01\x00" * 100)
        assert out.data.count(b"A") >= 1
        sent = len(out.data)
        g.cancel_response(flush=True)
        assert out.data[sent:] == wire.frame(b"F")
        sent = len(out.data)
        for _ in range(4):                       # the rest of the interrupted answer arrives in a burst
            await asyncio.sleep(0.04)
            g.play(b"\x02\x00" * 100)
        assert len(out.data) == sent
        await asyncio.sleep(0.25)                # the model paused, then answers the new question
        g.play(b"\x03\x00" * 100)
        assert out.data.endswith(b"\x03\x00" * 100)
    asyncio.run(go())


def test_the_transcript_is_split_into_utterances_by_pauses(monkeypatch, fake_vad):
    monkeypatch.setattr(gpt_live, "TURN_GAP_S", 0.05)
    g = gpt_live.GptLive(wire.Station(None, Out(), {}), None)

    async def go():
        g.hear("какая ")
        g.hear("погода")
        await asyncio.sleep(0.1)
        g.hear("и завтра")
    asyncio.run(go())
    assert g.turns == ["какая погода", "и завтра"]
