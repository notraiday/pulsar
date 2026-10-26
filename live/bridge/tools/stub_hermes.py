# Stand-in for the home agent (HERMES_URL=http://127.0.0.1:18642/v1/chat/completions): read-only canned answers, no side effects.
import asyncio, os, time
from aiohttp import web
T0=time.monotonic(); D=float(os.environ.get("SCHED_DELAY","4.0"))
def log(m): print(f"[stub {time.monotonic()-T0:6.1f}s] {m}", flush=True)
async def h(req):
    body=await req.json(); txt=body["messages"][-1]["content"]; low=txt.lower()
    log(f"START {txt!r}")
    if "расписан" in low or "календар" in low or "планы" in low:
        await asyncio.sleep(D); ans="Завтра в девять лекция по спецглавам БЖД, в одиннадцать теория оптимальных систем."
    elif "погод" in low:
        await asyncio.sleep(1.0); ans="Завтра днём плюс десять, ночью плюс четыре, облачно."
    else:
        await asyncio.sleep(0.5); ans="Не понял."
    log(f"DONE  {txt!r} -> {ans!r}")
    return web.json_response({"choices":[{"message":{"content":ans}}]})
app=web.Application(); app.router.add_post("/v1/chat/completions",h)
web.run_app(app,host="127.0.0.1",port=18642,print=None)
