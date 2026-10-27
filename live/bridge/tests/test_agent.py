"""The home agent's two ways (agent.py) against a fake Home Assistant and a fake Hermes on loopback."""
import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from conftest import fake_ha, serve_app
from livebridge import agent
from livebridge.util import BackendUnavailable


@pytest.fixture
def ha(monkeypatch):
    monkeypatch.setattr(agent, "HA_TOKEN", "ha-token")
    monkeypatch.setattr(agent, "HA_AGENT_ID", "conversation.extended_openai_conversation")
    monkeypatch.setattr(agent, "AGENT", "ha")


async def with_app(app, monkeypatch, scenario):
    runner, port = await serve_app(app)
    monkeypatch.setattr(agent, "HA_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(agent, "HERMES_URL", f"http://127.0.0.1:{port}/v1/chat/completions")
    try:
        async with aiohttp.ClientSession() as http:
            return await scenario(http)
    finally:
        await runner.cleanup()


def test_a_request_goes_to_the_pipeline_agent_as_the_station_and_keeps_the_conversation(monkeypatch, ha):
    seen = []

    async def scenario(http):
        a = agent.Agent(http, {"ha_satellite_id": "assist_satellite.kitchen", "ha_device_id": "dev1"})
        a.check()
        return [await a.ask("какая погода"), await a.ask("а завтра")]
    answers = asyncio.run(with_app(fake_ha(seen), monkeypatch, scenario))
    assert answers == ["Плюс десять, облачно."] * 2
    (auth1, b1), (auth2, b2) = seen
    assert auth1 == auth2 == "Bearer ha-token"
    assert b1 == {"text": "@@LIVE@@ какая погода", "language": "ru", "agent_id": "conversation.extended_openai_conversation",
                  "device_id": "dev1", "satellite_id": "assist_satellite.kitchen"}
    # the second request continues the conversation HA opened for the first
    assert b2["text"] == "@@LIVE@@ а завтра" and b2["conversation_id"] == "conv1"


def test_requests_side_by_side_do_not_share_a_conversation(monkeypatch, ha):
    seen = []

    async def scenario(http):
        a = agent.Agent(http)
        await a.ask("первая")
        await asyncio.gather(a.ask("вторая"), a.ask("третья"))
        await a.ask("четвёртая")
        return a.conversation_id
    conv = asyncio.run(with_app(fake_ha(seen, delay=0.05), monkeypatch, scenario))
    ids = {b["text"].split()[-1]: b.get("conversation_id") for _, b in seen}
    # one of the two parallel requests continues the dialog's conversation, the other gets one of its own
    assert ids["первая"] is None
    assert {ids["вторая"], ids["третья"]} == {"conv1", None}
    assert ids["четвёртая"] == "conv1" and conv == "conv1"


def test_the_conversation_is_not_kept_when_turned_off(monkeypatch, ha):
    monkeypatch.setattr(agent, "HA_KEEP_CONVERSATION", False)
    seen = []

    async def scenario(http):
        a = agent.Agent(http)
        await a.ask("раз")
        await a.ask("два")
    asyncio.run(with_app(fake_ha(seen), monkeypatch, scenario))
    assert [b.get("conversation_id") for _, b in seen] == [None, None]


@pytest.mark.parametrize("kw, err", [
    ({"status": 401}, "home assistant HTTP 401: 401: Unauthorized"),
    ({"response_type": "error", "speech": "Не поняла"}, "home assistant: no_intent_match: Не поняла"),
])
def test_ha_failures_are_errors_not_answers(monkeypatch, ha, kw, err):
    async def scenario(http):
        with pytest.raises(RuntimeError) as e:
            await agent.Agent(http).ask("что-нибудь")
        return str(e.value)
    assert asyncio.run(with_app(fake_ha([], **kw), monkeypatch, scenario)) == err


def test_tags_are_stripped_and_an_empty_answer_says_so(monkeypatch, ha):
    async def scenario(http):
        return await agent.Agent(http).ask("x")
    assert asyncio.run(with_app(fake_ha([], speech="[emo:happy] Готово."), monkeypatch, scenario)) == "Готово."
    assert asyncio.run(with_app(fake_ha([], speech=""), monkeypatch, scenario)) == "Сделано, ответ пустой."


def test_hermes_directly(monkeypatch):
    seen = []

    async def chat(req):
        seen.append((req.headers.get("Authorization"), await req.json()))
        return web.json_response({"choices": [{"message": {"content": "Сейчас 21:40."}}]})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    monkeypatch.setattr(agent, "HERMES_KEY", "hk")

    async def scenario(http):
        a = agent.Agent(http, {"agent": "hermes"})
        a.check()
        return await a.ask("который час")
    assert asyncio.run(with_app(app, monkeypatch, scenario)) == "Сейчас 21:40."
    auth, body = seen[0]
    assert auth == "Bearer hk" and body["messages"] == [{"role": "user", "content": "@@LIVE@@ который час"}]


def test_hermes_streamed_answer_is_what_follows_the_last_tool(monkeypatch):
    def sse(*parts):
        out = []
        for p in parts:
            if p.startswith("tool:"):
                out.append('event: hermes.tool.progress\ndata: {"tool": "%s"}\n\n' % p[5:])
            else:
                out.append("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}, ensure_ascii=False) + "\n\n")
        return "".join(out) + "data: [DONE]\n\n"
    bodies = [sse("I'll open Wikipedia.", "tool:browser_navigate", "Now Tomsk.", "tool:browser_navigate",
                  "Омск — 1716, ", "Томск — 1604."),
              sse("Готово.", "tool:memory")]               # nothing after the last tool: the last text it said

    async def chat(req):
        await req.json()
        return web.Response(text=bodies.pop(0), content_type="text/event-stream",
                            headers={"X-Hermes-Session-Id": "s1"})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    sessions = []

    async def scenario(http):
        a = agent.Agent(http, {"agent": "hermes"})
        return [await a.ask("x", on_session=sessions.append), await a.ask("y")]
    assert asyncio.run(with_app(app, monkeypatch, scenario)) == ["Омск — 1716, Томск — 1604.", "Готово."]
    assert sessions == ["s1"]


def test_an_agent_that_cannot_work_is_refused_before_the_dialog(monkeypatch):
    monkeypatch.setattr(agent, "AGENT", "ha")
    monkeypatch.setattr(agent, "HA_URL", "")
    monkeypatch.setattr(agent, "HA_TOKEN", "")
    with pytest.raises(BackendUnavailable, match="HA_URL"):
        agent.Agent(None).check()
    with pytest.raises(BackendUnavailable, match="unknown agent"):
        agent.Agent(None, {"agent": "telepathy"}).check()
    agent.Agent(None, {"agent": "hermes"}).check()  # needs nothing up front
