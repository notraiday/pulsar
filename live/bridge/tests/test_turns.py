"""Turn-taking with a person: an utterance that stops on a word leading on waits for the rest, an owner who goes on
right after their turn ends is one turn, noise over a cut answer does not lose the question, and the history keeps
what the owner actually heard of an answer they talked over. The mic is fed in real time (32 ms a frame)."""
import asyncio
import json

import aiohttp
from aiohttp import web

import bridge
from conftest import LOUDER, QUIET, serve_app, station
from livebridge import agent, session, stt, wire
from livebridge.backends import pipeline
from livebridge.routes import Routes


def fakes(seen, transcripts, reply, think_s=0.0, tts_s=0.1):
    """`reply(body)`: the model's text for a round; the first round thinks `think_s`; every TTS answer is `tts_s` of
    audio."""
    async def chat(req):
        body = await req.json()
        seen["chat"].append(body)
        if len(seen["chat"]) == 1 and think_s:
            await asyncio.sleep(think_s)
        ev = {"choices": [{"delta": {"content": reply(body)}}]}
        return web.Response(text="data: " + json.dumps(ev, ensure_ascii=False) + "\n\ndata: [DONE]\n\n",
                            content_type="text/event-stream")

    async def speech(req):
        seen["tts"].append((await req.json())["input"])
        return web.Response(body=b"\x10\x00" * int(24000 * tts_s))

    async def transcriptions(req):
        await req.post()
        seen["stt"] += 1
        return web.json_response({"text": transcripts[min(seen["stt"], len(transcripts)) - 1]})

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/audio/speech", speech)
    app.router.add_get("/key", lambda req: web.json_response({}))
    app.router.add_post("/stt/audio/transcriptions", transcriptions)
    return app


def run(tmp_path, monkeypatch, transcripts, script, reply=lambda body: "Хорошо.", **kw):
    """`script`: (frame, count) pairs fed to the station's mic one frame per 32 ms; returns what the fakes saw."""
    seen = {"chat": [], "tts": [], "stt": 0}
    monkeypatch.setattr(pipeline, "OR_KEY", "test-key")
    monkeypatch.setattr(pipeline, "AGENT_RACE", False)
    monkeypatch.setattr(pipeline, "LLM_HEDGE_S", 0)   # the slow first round is on purpose
    monkeypatch.setattr(agent, "AGENT", "hermes")
    monkeypatch.setattr(session, "IDLE_S", 0.6)
    monkeypatch.setattr(stt, "INPUT", "text")

    async def go():
        runner, port = await serve_app(fakes(seen, transcripts, reply, **kw))
        monkeypatch.setattr(pipeline, "OR_BASE", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(stt, "URL", f"http://127.0.0.1:{port}/stt")
        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"default": "pipeline"}))
        async with aiohttp.ClientSession() as http:
            server = await bridge.serve("127.0.0.1:0", Routes(routes), http)
            r, w = await station(server.sockets[0].getsockname()[1])

            async def read():
                """Plays the answer audio in real time: D once what it got has played, nothing after F."""
                loop, end, drained = asyncio.get_running_loop(), 0.0, None
                while (f := await wire.read_frame(r)) is not None and f[0] != b"X":
                    if f[0] == b"A":
                        end = max(loop.time(), end) + len(f[1]) / 48000
                        if drained:
                            drained.cancel()
                        drained = loop.call_at(end, lambda: w.write(wire.frame(b"D")))
                    elif f[0] == b"F":
                        end = 0.0
                        if drained:
                            drained.cancel()
            reader = asyncio.create_task(read())
            for fr, n in script:
                for _ in range(n):
                    w.write(wire.frame(b"A", fr))
                    await asyncio.sleep(0.032)
            while not reader.done():  # the station keeps streaming its mic until the dialog ends
                w.write(wire.frame(b"A", QUIET))
                await asyncio.sleep(0.032)
            server.close()
            await runner.cleanup()
    asyncio.run(asyncio.wait_for(go(), 20))
    return seen


def users(body):
    return [m["content"] for m in body["messages"] if m["role"] == "user"]


def test_a_pause_after_a_word_that_leads_on_waits_for_the_rest(tmp_path, monkeypatch, fake_vad):
    # "Включи свет в" ... 1 s ... "спальне": one turn (600 ms of silence would have ended it)
    seen = run(tmp_path, monkeypatch, ["Включи свет в.", "Включи свет в спальне."],
               [(LOUDER, 12), (QUIET, 31), (LOUDER, 10), (QUIET, 25)])
    assert len(seen["chat"]) == 1
    assert users(seen["chat"][0]) == ["Включи свет в спальне."]


def test_a_finished_sentence_ends_the_turn_as_before(tmp_path, monkeypatch, fake_vad):
    seen = run(tmp_path, monkeypatch, ["Включи свет в спальне.", "Спасибо."],
               [(LOUDER, 12), (QUIET, 31), (LOUDER, 10), (QUIET, 25)])
    assert users(seen["chat"][0]) == ["Включи свет в спальне."]


def test_going_on_right_after_the_turn_ended_is_one_turn(tmp_path, monkeypatch, fake_vad):
    # the model is still thinking about "Включи свет." when the owner adds "в спальне": one turn, one answer
    seen = run(tmp_path, monkeypatch, ["Включи свет.", "В спальне."],
               [(LOUDER, 12), (QUIET, 25), (LOUDER, 18), (QUIET, 25)], think_s=2.0)
    last = seen["chat"][-1]
    assert users(last) == ["Включи свет. В спальне."]
    assert not any(m["role"] == "assistant" for m in last["messages"])   # the cut answer left nothing behind
    assert seen["tts"] == ["Хорошо."]


def test_noise_over_a_cut_answer_does_not_lose_the_question(tmp_path, monkeypatch, fake_vad):
    seen = run(tmp_path, monkeypatch, ["Какая погода завтра?", ""],
               [(LOUDER, 12), (QUIET, 25), (LOUDER, 18), (QUIET, 25)], think_s=2.0)
    assert users(seen["chat"][-1]) == ["Какая погода завтра?"]   # asked again, nothing added
    assert seen["tts"] == ["Хорошо."]


def test_the_history_keeps_what_the_owner_heard_of_an_answer_they_talked_over(tmp_path, monkeypatch, fake_vad):
    answer = "Раз два три четыре пять шесть семь восемь девять десять. Одиннадцать двенадцать."

    def reply(body):
        return answer if len(users(body)) == 1 else "Поняла."
    # each sentence plays 3 s; the owner talks over the first one about 1 s into it
    seen = run(tmp_path, monkeypatch, ["Посчитай до двенадцати.", "Стоп, хватит."],
               [(LOUDER, 12), (QUIET, 50), (LOUDER, 18), (QUIET, 25)], reply=reply, tts_s=3.0)
    said = next(m["content"] for m in seen["chat"][-1]["messages"] if m["role"] == "assistant")
    assert said.startswith("Раз") and said.endswith(pipeline.CUT_MARK)
    heard = said.removesuffix(pipeline.CUT_MARK)
    assert "десять" not in heard and "Одиннадцать" not in heard and "…" in heard


def test_words_that_lead_on():
    for t in ["Включи свет в.", "Купи хлеб и?", "Сколько будет два +?", "Поставь таймер на.", "Ну и что.", "Скажи,"]:
        assert stt.leads_on(t), t
    for t in ["Какая погода завтра?", "Нет.", "Как?", "Что?", "Сколько будет 2 + 2?", "Включи свет в спальне.", ""]:
        assert not stt.leads_on(t), t


def test_a_turn_the_model_left_unanswered_takes_the_next_utterance(tmp_path, monkeypatch, fake_vad):
    # the model keeps silent on what sounds unfinished; the rest comes as a turn of its own: it joins the first
    def reply(body):
        return "[SILENT]" if users(body) == ["Сколько будет два."] else "Пять."
    seen = run(tmp_path, monkeypatch, ["Сколько будет два.", "Плюс три."],
               [(LOUDER, 12), (QUIET, 31), (LOUDER, 10), (QUIET, 25)], reply=reply)
    assert users(seen["chat"][-1]) == ["Сколько будет два. Плюс три."]
    assert seen["tts"] == ["Пять."]


def test_a_short_addition_while_the_answer_is_made_goes_on_the_turn(tmp_path, monkeypatch, fake_vad):
    # "три" is too short to cut an answer that plays, but nothing plays yet: the answer waits and the turn grows
    seen = run(tmp_path, monkeypatch, ["Сколько будет два.", "Три."],
               [(LOUDER, 12), (QUIET, 30), (LOUDER, 8), (QUIET, 25)], think_s=2.0)
    assert users(seen["chat"][-1]) == ["Сколько будет два. Три."]
    assert seen["tts"] == ["Хорошо."]


def test_short_noise_while_the_answer_is_made_leaves_it_alone(tmp_path, monkeypatch, fake_vad):
    seen = run(tmp_path, monkeypatch, ["Сколько будет два плюс два?", ""],
               [(LOUDER, 12), (QUIET, 30), (LOUDER, 8), (QUIET, 25)], think_s=2.0)
    assert len(seen["chat"]) == 1 and seen["tts"] == ["Хорошо."]
