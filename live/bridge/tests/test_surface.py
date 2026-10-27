"""The agent's questions to the owner through the speakers (surface.py), detached agent requests, spoken answers."""
import asyncio
import json
import logging
import time

import aiohttp
from aiohttp import web

import bridge
from conftest import LOUDER, QUIET, frames_until, serve_app, station
from livebridge import agent, reminders, session, spoken, stt, surface, wire
from livebridge.backends import pipeline
from livebridge.routes import Routes

AUTH = {"Authorization": "Bearer surface-token"}


# ---- spoken answers ----

def test_codes_are_read_the_way_people_say_them():
    assert spoken.digits("четыре пять один два") == "4512"
    assert spoken.digits("сорок пять двенадцать") == "4512"
    assert spoken.digits("код сто двадцать три") == "123"
    assert spoken.digits("ноль пять семь") == "057"
    assert spoken.digits("двадцать ноль пять") == "2005"
    assert spoken.digits("45 12") == "4512"
    assert spoken.digits("ну это") == ""


def test_yes_no_prefers_no():
    assert spoken.yes_no("Да, оплачивай") == "yes"
    assert spoken.yes_no("давай") == "yes"
    assert spoken.yes_no("нет") == "no"
    assert spoken.yes_no("не надо") == "no"
    assert spoken.yes_no("да нет, стоп") == "no"
    assert spoken.yes_no("что?") == "unclear"


# ---- a whole dialog ----

def fakes(seen, transcripts, hermes_delay=0.0, release=None, reply=None, model=None):
    async def chat(req):
        body = await req.json()
        seen["chat"].append(body)
        msgs = body["messages"]
        if model is not None and (ev := model(body)) is not None:
            pass
        elif reply is not None:                    # the model answers by itself, without the agent
            ev = {"choices": [{"delta": {"content": reply}}]}
        elif any("Агент закончил фоновую просьбу" in str(m.get("content")) for m in msgs):
            ev = {"choices": [{"delta": {"content": "Заказ оформлен."}}]}
        elif any(m.get("role") == "tool" for m in msgs[-2:]):
            ev = {"choices": [{"delta": {"content": "Готово."}}]}
        else:
            ev = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "ask_hermes", "arguments": json.dumps({"request": "сделай"})}}]}}]}
        return web.Response(text="data: " + json.dumps(ev) + "\n\ndata: [DONE]\n\n", content_type="text/event-stream")

    async def speech(req):
        seen["tts"].append((await req.json())["input"])
        return web.Response(body=b"\x10\x00" * 2400)

    async def transcriptions(req):
        await req.post()
        seen["stt"] += 1
        return web.json_response({"text": transcripts[min(seen["stt"], len(transcripts)) - 1]})

    async def hermes(req):
        seen["hermes"].append((await req.json())["messages"][-1]["content"])
        sid = req.headers.get("X-Hermes-Session-Id") or "s1"
        seen.setdefault("sessions", []).append(sid)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "X-Hermes-Session-Id": sid})
        await resp.prepare(req)
        await resp.write(b"event: hermes.tool.progress\ndata: {\"tool\": \"browser_navigate\"}\n\n")
        if release is not None:
            await release.wait()
        await asyncio.sleep(hermes_delay)
        ev = {"choices": [{"delta": {"content": "Готово, заказ оформлен."}}]}
        await resp.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode())
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", lambda req: web.json_response({}))
    app.router.add_post("/stt/audio/transcriptions", transcriptions)
    app.router.add_post("/hermes/v1/chat/completions", hermes)
    return app


def setup(monkeypatch, port):
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(pipeline, "AGENT_RACE", True)
    monkeypatch.setattr(agent, "AGENT", "hermes")
    monkeypatch.setattr(agent, "HERMES_URL", f"http://127.0.0.1:{port}/hermes/v1/chat/completions")
    monkeypatch.setattr(stt, "URL", f"http://127.0.0.1:{port}/stt")
    monkeypatch.setattr(stt, "INPUT", "text")
    monkeypatch.setattr(session, "IDLE_S", 0.6)
    monkeypatch.setattr(surface, "TOKEN", "surface-token")
    monkeypatch.setattr(surface, "registry", surface.Registry())
    monkeypatch.setattr(reminders, "SOUND", "off")


def speak(w):
    for _ in range(20):
        w.write(wire.frame(b"A", LOUDER))
    for _ in range(25):
        w.write(wire.frame(b"A", QUIET))


async def until(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


def test_a_code_the_agent_asks_for_goes_to_it_alone(tmp_path, monkeypatch, fake_vad, caplog):
    caplog.set_level(logging.INFO, logger="live")
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    async def go():
        release = asyncio.Event()
        runner, port = await serve_app(fakes(seen, ["Войди на сайт.", "четыре пять один два"], release=release))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        monkeypatch.setattr(pipeline, "DETACH_S", 30)
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])
            speak(w)
            await until(lambda: "s1" in surface.registry.dialogs)
            base = f"http://127.0.0.1:{s_port}"
            async with http.post(f"{base}/ask", json={"session": "s1", "kind": "code", "text": "Скажи код из SMS."},
                                 headers=AUTH) as resp:
                ask = await resp.json()
            await frames_until(r, {b"A"})          # the question
            w.write(wire.frame(b"D"))
            await asyncio.sleep(0.1)
            speak(w)                               # the owner says the code
            async with http.get(f"{base}/ask/{ask['id']}?wait=5", headers=AUTH) as resp:
                answer = await resp.json()
            await frames_until(r, {b"A"})          # "got it"
            w.write(wire.frame(b"D"))
            chats_before_release = len(seen["chat"])
            release.set()
            got = await frames_until(r, {b"A"})     # the agent's answer
            w.write(wire.frame(b"D"))
            got += await frames_until(r, {b"X"})
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return ask, answer, chats_before_release, got
    ask, answer, chats, got = asyncio.run(go())
    assert ask["wake"] is False
    assert answer == {"status": "answered", "answer": "4512"}
    assert seen["tts"][:2] == ["Скажи код из SMS.", pipeline.ACKS["code"]]
    assert seen["tts"][-1] == "Готово, заказ оформлен."
    assert chats == 1                              # no model round for the code
    assert "4512" not in caplog.text and "четыре" not in caplog.text
    assert got[-1] == (b"X", b"idle")


def test_a_long_request_goes_on_in_the_background_and_is_told_later(tmp_path, monkeypatch, fake_vad):
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    async def go():
        runner, port = await serve_app(fakes(seen, ["Закажи пиццу."], hermes_delay=1.0))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        monkeypatch.setattr(pipeline, "DETACH_S", 0.3)
        monkeypatch.setattr(session, "IDLE_S", 3)   # the owner is still there when the result comes
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])
            speak(w)
            await frames_until(r, {b"A"})          # "this takes a while"
            w.write(wire.frame(b"D"))
            async with http.post(f"http://127.0.0.1:{s_port}/result", json={"session": "s1", "text": "x"},
                                 headers=AUTH) as resp:
                alive = await resp.json()
            await frames_until(r, {b"A"})          # the result, in the same dialog
            w.write(wire.frame(b"D"))
            got = await frames_until(r, {b"X"})
            async with http.post(f"http://127.0.0.1:{s_port}/result", json={"session": "s1", "text": "x"},
                                 headers=AUTH) as resp:
                gone = await resp.json()
            async with http.post(f"http://127.0.0.1:{s_port}/result", json={"session": "s1"}) as resp:
                unauthorized = resp.status
            async with http.post(f"http://127.0.0.1:{s_port}/result", json={"session": "ha-pipeline"},
                                 headers=AUTH) as resp:
                unknown = await resp.json()
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return alive, gone, unauthorized, unknown, got
    alive, gone, unauthorized, unknown, got = asyncio.run(go())
    assert seen["tts"] == [pipeline.DETACHED_SPEECH, "Заказ оформлен."]
    assert alive == {"announce": False} and gone == {"announce": True} and unauthorized == 401
    assert unknown == {"announce": False}          # a session no Live dialog started (HA's own pipeline)
    assert len(seen["hermes"]) == 1                # one request: the early one, detached
    assert got[-1] == (b"X", b"idle")


def test_a_dropped_request_asks_nobody_and_is_not_announced(tmp_path, monkeypatch, fake_vad):
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    async def go():
        release = asyncio.Event()                  # the agent's run is still going when the model has answered
        runner, port = await serve_app(fakes(seen, ["Спроси меня, какой цвет мне нравится."], release=release,
                                             reply="Какой цвет тебе нравится?"))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        monkeypatch.setattr(agent, "HERMES_KEY", "hk")   # sessions are named by the dialog
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])
            speak(w)
            await frames_until(r, {b"A"})          # the model's own question
            w.write(wire.frame(b"D"))
            await until(lambda: surface.registry.abandoned)
            sid = seen["sessions"][0]
            base = f"http://127.0.0.1:{s_port}"
            async with http.post(f"{base}/ask", json={"session": sid, "kind": "text", "text": "Какой цвет?"},
                                 headers=AUTH) as resp:
                ask = await resp.json()
            async with http.get(f"{base}/ask/{ask['id']}?wait=1", headers=AUTH) as resp:
                answer = await resp.json()
            async with http.post(f"{base}/result", json={"session": sid, "text": "x"}, headers=AUTH) as resp:
                result = await resp.json()
            release.set()
            got = await frames_until(r, {b"X"})
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return sid, ask, answer, result, got
    sid, ask, answer, result, got = asyncio.run(go())
    assert sid.startswith("live-") and len(seen["sessions"]) == 1
    assert ask["wake"] is False and answer == {"status": "no_answer", "answer": ""}
    assert result == {"announce": False}
    assert seen["tts"] == ["Какой цвет тебе нравится?"]
    assert got[-1] == (b"X", b"idle")


async def background_request(http, server, seen):
    """A dialog whose request goes on in the background, then ends; the agent session of that request."""
    r, w = await station(server.sockets[0].getsockname()[1])
    speak(w)
    await frames_until(r, {b"A"})                  # "this takes a while"
    w.write(wire.frame(b"D"))
    await frames_until(r, {b"X"})
    return seen["sessions"][0]


def test_a_background_request_asks_and_answers_in_the_dialog_in_progress(tmp_path, monkeypatch, fake_vad):
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    async def go():
        release = asyncio.Event()
        runner, port = await serve_app(fakes(seen, ["Закажи пиццу.", "Синий."], release=release))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        monkeypatch.setattr(agent, "HERMES_KEY", "hk")
        monkeypatch.setattr(pipeline, "DETACH_S", 0.3)
        monkeypatch.setattr(session, "FIRST_IDLE_S", 5)
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        base = f"http://127.0.0.1:{s_port}"
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            sid = await background_request(http, server, seen)
            async with http.get(f"{base}/session/{sid}", headers=AUTH) as resp:
                bg = await resp.json()
            r, w = await station(server.sockets[0].getsockname()[1])   # the owner is in another dialog now
            await until(lambda: surface.registry.current() is not None)
            async with http.post(f"{base}/ask", json={"session": sid, "kind": "text", "text": "Какой цвет коробки?"},
                                 headers=AUTH) as resp:
                ask = await resp.json()
            await frames_until(r, {b"A"})          # the question, with the lead
            w.write(wire.frame(b"D"))
            await asyncio.sleep(0.1)
            speak(w)
            async with http.get(f"{base}/ask/{ask['id']}?wait=5", headers=AUTH) as resp:
                answer = await resp.json()
            await frames_until(r, {b"A"})          # "got it"
            w.write(wire.frame(b"D"))
            async with http.post(f"{base}/result", json={"session": sid, "text": "Заказ оформлен, привезут к восьми."},
                                 headers=AUTH) as resp:
                result = await resp.json()
            await frames_until(r, {b"A"})          # the result, told in this dialog
            w.write(wire.frame(b"D"))
            got = await frames_until(r, {b"X"})
            release.set()
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return bg, ask, answer, result, got
    bg, ask, answer, result, got = asyncio.run(go())
    assert bg == {"background": True}
    assert ask["wake"] is False and answer == {"status": "answered", "answer": "Синий."}
    assert result == {"announce": False}
    assert seen["tts"][:4] == [pipeline.DETACHED_SPEECH, surface.FOREIGN_LEAD + "Какой цвет коробки?",
                               pipeline.ACKS["text"], "Заказ оформлен."]
    note = seen["chat"][-1]["messages"][-1]["content"]
    assert "Агент закончил фоновую просьбу «сделай»: Заказ оформлен, привезут к восьми." in note
    assert got[-1] == (b"X", b"idle")


def test_the_owner_hears_of_and_can_stop_a_background_request(tmp_path, monkeypatch, fake_vad):
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    def model(body):
        names = [t["function"]["name"] for t in body["tools"]]
        if "stop_background" in names and body["messages"][-1].get("role") == "user":
            return {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "s1", "function": {
                "name": "stop_background", "arguments": json.dumps({"number": 1})}}]}}]}
        return None

    async def go():
        release = asyncio.Event()
        runner, port = await serve_app(fakes(seen, ["Закажи пиццу.", "Отмени пиццу."], release=release, model=model))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        monkeypatch.setattr(agent, "HERMES_KEY", "hk")
        monkeypatch.setattr(pipeline, "AGENT_RACE", False)
        monkeypatch.setattr(pipeline, "DETACH_S", 0.3)
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            sid = await background_request(http, server, seen)
            running = [b.session for b in surface.registry.running()]
            r, w = await station(server.sockets[0].getsockname()[1])
            speak(w)
            await frames_until(r, {b"A"})          # after stop_background
            w.write(wire.frame(b"D"))
            got = await frames_until(r, {b"X"})
            release.set()
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return sid, running, got
    sid, running, got = asyncio.run(go())
    assert running == [sid]
    stop_round = seen["chat"][-2]
    assert "1. «сделай», идёт" in stop_round["messages"][0]["content"]
    assert "stop_background" in [t["function"]["name"] for t in stop_round["tools"]]
    assert seen["chat"][-1]["messages"][-1]["content"].startswith("Остановлено: «сделай»")
    assert surface.registry.running() == [] and sid in surface.registry.abandoned
    assert seen["tts"][-1] == "Готово." and got[-1] == (b"X", b"idle")


def test_with_no_dialog_the_speakers_are_woken_and_the_loudest_answers(tmp_path, monkeypatch, fake_vad):
    seen = {"chat": [], "tts": [], "stt": 0, "hermes": []}

    async def go():
        runner, port = await serve_app(fakes(seen, ["да, оплачивай"]))
        s_runner, s_port = await serve_app(surface.app())
        setup(monkeypatch, port)
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            base = f"http://127.0.0.1:{s_port}"
            async with http.post(f"{base}/ask", json={"session": "gone", "kind": "confirm", "text": "Оплатить заказ?"},
                                 headers=AUTH) as resp:
                ask = await resp.json()
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            # what Home Assistant's start_conversation makes the station do
            r, w = await station(server.sockets[0].getsockname()[1],
                                 {"station": "kitchen", "protocol": 1, "reminder": "Оплатить заказ?"})
            await frames_until(r, {b"A"})
            w.write(wire.frame(b"D"))
            await asyncio.sleep(0.5)
            speak(w)
            async with http.get(f"{base}/ask/{ask['id']}?wait=5", headers=AUTH) as resp:
                answer = await resp.json()
            got = await frames_until(r, {b"A"})    # the acknowledgement
            w.write(wire.frame(b"D"))
            got += await frames_until(r, {b"X"})
            server.close()
            await runner.cleanup()
            await s_runner.cleanup()
            return ask, answer, got
    ask, answer, got = asyncio.run(go())
    assert ask["wake"] is True
    assert answer == {"status": "answered", "answer": "yes"}
    assert seen["tts"] == ["Оплатить заказ?", pipeline.ACKS["yes"]]
    assert seen["chat"] == [] and seen["hermes"] == []   # nothing went to a model or the agent
    assert got[-1] == (b"X", b"answered")
