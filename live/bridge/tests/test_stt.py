"""The local STT (stt.py): the transcript as the turn (made ahead, once), next to the audio, or not at all; short
commands kept, noise dropped, the sound when the STT fails."""
import asyncio
import json

import aiohttp
from aiohttp import web

import bridge
from conftest import LOUDER, QUIET, frames_until, serve_app, station
from livebridge import agent, session, stt, wire
from livebridge.backends import pipeline
from livebridge.routes import Routes


def fakes(seen, transcript, stt_status=200):
    async def chat(req):
        seen["chat"].append(await req.json())
        ev = {"choices": [{"delta": {"content": "Хорошо."}}]}
        return web.Response(text="data: " + json.dumps(ev) + "\n\ndata: [DONE]\n\n", content_type="text/event-stream")

    async def speech(req):
        return web.Response(body=b"\x10\x00" * 2400)

    async def transcriptions(req):
        form = await req.post()
        seen["stt"].append((req.headers.get("Authorization"), form["model"], len(form["file"].file.read())))
        if stt_status != 200:
            return web.Response(status=stt_status, text="boom")
        return web.json_response({"text": transcript})

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", lambda req: web.json_response({}))
    app.router.add_post("/stt/audio/transcriptions", transcriptions)
    return app


def run(tmp_path, monkeypatch, transcript, frames, stt_status=200):
    seen = {"chat": [], "stt": []}
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(agent, "AGENT", "hermes")
    monkeypatch.setattr(session, "IDLE_S", 0.4)
    monkeypatch.setattr(session, "FIRST_IDLE_S", 1.5)
    monkeypatch.setattr(stt, "KEY", "stt-key")
    monkeypatch.setattr(pipeline, "AGENT_RACE", False)  # these turns are about the transcript, not the agent

    async def go():
        runner, port = await serve_app(fakes(seen, transcript, stt_status))
        monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(stt, "URL", f"http://127.0.0.1:{port}/stt")
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])
            for f in frames:
                w.write(wire.frame(b"A", f))
            got = await frames_until(r, {b"A", b"X"}, timeout=8)
            if got[-1][0] == b"A":
                w.write(wire.frame(b"D"))  # the answer has been played
                got += await frames_until(r, {b"X"}, timeout=8)
            server.close()
            await runner.cleanup()
            return got
    return seen, asyncio.run(go())


def user_content(body):
    return [m for m in body["messages"] if m["role"] == "user"][-1]["content"]


def test_a_short_command_the_vad_would_drop_is_kept(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "Удалите их.", [LOUDER] * 7 + [QUIET] * 25)   # 224 ms voiced
    assert len(seen["stt"]) == 1 and seen["stt"][0][:2] == ("Bearer stt-key", stt.MODEL)
    assert len(seen["chat"]) == 1
    assert user_content(seen["chat"][0]) == "Удалите их."


def test_short_noise_stays_ignored(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "", [LOUDER] * 7 + [QUIET] * 25)
    assert len(seen["stt"]) == 1 and seen["chat"] == []
    assert got[-1] == (b"X", b"idle")


SPEECH = [LOUDER] * 20 + [QUIET] * 25


def test_by_default_the_turn_is_its_transcript_made_ahead_once(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "Какая погода завтра?", SPEECH)
    assert len(seen["stt"]) == 1                      # the one started at the pause, not a second one at the end
    assert user_content(seen["chat"][0]) == "Какая погода завтра?"
    assert "расшифровкой" in seen["chat"][0]["messages"][0]["content"]   # the persona knows it gets text


def test_a_pause_inside_the_utterance_restarts_the_transcript(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "Какая погода завтра?",
                    [LOUDER] * 10 + [QUIET] * 10 + [LOUDER] * 10 + [QUIET] * 25)
    assert len(seen["chat"]) == 1
    # the one started at the pause was dropped; the turn is the transcript of the whole utterance (both parts, more
    # than the ~22 frames the first part and its pause come to)
    assert seen["stt"][-1][2] > 30 * 1024


def test_both_puts_the_transcript_next_to_the_sound(tmp_path, monkeypatch, fake_vad):
    monkeypatch.setattr(stt, "INPUT", "both")
    seen, got = run(tmp_path, monkeypatch, "Какая погода завтра?", SPEECH)
    content = user_content(seen["chat"][0])
    assert [c["type"] for c in content] == ["text", "input_audio"] and "Какая погода завтра?" in content[0]["text"]


def test_audio_gives_the_model_the_sound_alone(tmp_path, monkeypatch, fake_vad):
    monkeypatch.setattr(stt, "INPUT", "audio")
    seen, got = run(tmp_path, monkeypatch, "Какая погода завтра?", SPEECH)
    assert [c["type"] for c in user_content(seen["chat"][0])] == ["input_audio"]


def test_when_the_stt_fails_the_turn_is_the_sound(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "", SPEECH, stt_status=500)
    assert [c["type"] for c in user_content(seen["chat"][0])] == ["input_audio"]


def test_a_transcript_without_words_is_not_answered(tmp_path, monkeypatch, fake_vad):
    seen, got = run(tmp_path, monkeypatch, "", SPEECH)
    assert seen["chat"] == [] and got[-1] == (b"X", b"idle")
