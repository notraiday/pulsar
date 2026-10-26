import asyncio
import json
import os

from livebridge import wire
from livebridge.routes import Routes


def reader(data: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    r.feed_data(data)
    r.feed_eof()
    return r


def test_frame_roundtrip():
    async def go():
        r = reader(wire.frame(b"A", b"\x01\x02") + wire.frame(b"F"))
        assert await wire.read_frame(r) == (b"A", b"\x01\x02")
        assert await wire.read_frame(r) == (b"F", b"")
        assert await wire.read_frame(r) is None
    asyncio.run(go())


def test_oversize_and_truncated_frames_are_refused():
    async def go():
        big = b"A" + (wire.MAX_FRAME + 1).to_bytes(4, "little")
        assert await wire.read_frame(reader(big)) is None
        assert await wire.read_frame(reader(b"A\x05\0\0\0ab")) is None
    asyncio.run(go())


def test_hello_is_tolerant():
    assert wire.parse_hello(b'{"station":"kitchen","wake_word":"alisa"}') == {"station": "kitchen", "wake_word": "alisa"}
    assert wire.parse_hello(b"not json") == {}
    assert wire.parse_hello(b"[1]") == {}
    assert wire.parse_hello(b"") == {}


def write(path, doc, bump):
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump))


def test_routes_defaults_without_a_file(tmp_path):
    assert Routes(None).pick("kitchen") == ("pipeline", {})
    assert Routes(tmp_path / "missing.json").pick("kitchen") == ("pipeline", {})


def test_routes_first_matching_pattern_wins(tmp_path):
    p = tmp_path / "routes.json"
    write(p, {"default": "pipeline", "stations": {
        "kitchen*": {"backend": "gpt-live", "options": {"voice": "marin"}},
        "kitchen-2": "handoff",
        "*": "handoff"}}, 1)
    r = Routes(p)
    assert r.pick("kitchen-2") == ("gpt-live", {"voice": "marin"})
    assert r.pick("bedroom") == ("handoff", {})


def test_routes_are_hot_reloaded_and_a_bad_file_keeps_the_last_good_one(tmp_path):
    p = tmp_path / "routes.json"
    write(p, {"default": "handoff"}, 1)
    r = Routes(p)
    assert r.pick("x")[0] == "handoff"
    write(p, {"default": "gpt-live", "fallback": "none"}, 2)
    assert r.pick("x")[0] == "gpt-live" and r.fallback == "none"
    write(p, "{ not json", 3)
    assert r.pick("x")[0] == "gpt-live"
    write(p, {"stations": {"a": {"options": {}}}}, 4)  # no "backend"
    assert r.pick("x")[0] == "gpt-live"
    write(p, {"default": "pipeline"}, 5)
    assert r.pick("x")[0] == "pipeline"
