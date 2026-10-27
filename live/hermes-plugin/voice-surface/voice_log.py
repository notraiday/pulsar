"""A short journal of the owner's voice requests in the Obsidian vault (``<log_dir>/YYYY-MM-DD.md``, by default
``/srv/vault/Агент/Голос``): one line per request worth remembering, so the voice agent can answer "what did I ask
yesterday" with the ``voice_log`` tool.

Nothing personal goes in. A turn that read private data (any tool outside LOG_SAFE: mail, money, calendar, the
browser, the vault, memory, the owner's answers) is written from the owner's words alone: its answer and its tool
results never reach the writer. The line itself is written by the auxiliary model, whose rule is "no personal
details" (no people's names, contacts, addresses, amounts on accounts, message contents, codes, health). That is why
the model router counts ``voice_log`` as a safe tool (EXTRA_SAFE_TOOLS): reading it taints nothing.

The writer goes through the model router like every auxiliary call. For a turn that read private data the owner's
words come to it as the result of a tool outside the router's safe list (``voice_private_turn``), so the router keeps
that request on the local model, as it does the turn itself.
"""
from __future__ import annotations

import contextvars
import fcntl
import json
import logging
import os
import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("voice_surface")

LIVE_MARK = "@@LIVE@@"
# Tools whose results carry nothing private (prefixes of the last name segment, as the router matches them).
LOG_SAFE = ("hass", "getlivecontext", "getdatetime", "get_current_time", "convert_time", "clarify", "kagi_",
            "weather_forecast", "live_reminder", "web_search", "web_extract")
MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября",
          "декабря")
WRITER = (
    "Ты ведёшь короткий журнал голосовых поручений хозяина. По одной его просьбе напиши одну строку журнала: "
    "от первого лица, в прошедшем времени, до двадцати слов — что он просил и чем кончилось («Включила свет в "
    "спальне.», «Нашла билеты в Казань на пятницу, от 4 500 ₽.», «Поставила напоминание про вынос мусора на 20:00.»). "
    "Никаких личных данных: ни имён людей, ни телефонов, адресов, почты, сумм на счетах и трат, содержания писем и "
    "сообщений, кодов, паролей, здоровья — только суть дела («Посмотрела почту.», «Заказала пиццу на сайте.»). Если "
    "это мелочь, которую незачем помнить (приветствие, болтовня, спасибо, вопрос о времени, дате или погоде, "
    "арифметика, переспрос), ответь ровно SKIP. Ответ — только строка журнала или SKIP.")
PRIVATE_NOTE = ("(Для этой просьбы понадобились личные данные: ответа ты не видишь. Запиши только, что за дело, "
                "без подробностей и без итога, если он не следует из самой просьбы.)")
_lock = threading.Lock()


def log_dir(cfg: dict) -> Path | None:
    d = str(cfg.get("log_dir", "/srv/vault/Агент/Голос")).strip()
    return Path(d) if d else None


def _tool_names(turn: list) -> list[str]:
    out = []
    for m in turn:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                name = str(((tc or {}).get("function") or {}).get("name") or "")
                if name and name not in out:
                    out.append(name)
    return out


def _safe(name: str) -> bool:
    short = re.split(r"__", name)[-1].lower()
    return short.startswith(LOG_SAFE)


def this_turn(history: list) -> list:
    """The messages of the last turn: what follows its user message."""
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], dict) and history[i].get("role") == "user":
            return history[i + 1:]
    return list(history)


def writer_messages(request: str, answer: str, tools: list[str]) -> list[dict]:
    if any(not _safe(t) for t in tools):  # private: the router keeps it local (module doc); no answer, no results
        return [{"role": "system", "content": WRITER},
                {"role": "user", "content": "Строка журнала для этой просьбы."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "turn", "type": "function", "function": {
                    "name": "voice_private_turn", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "turn", "content": f"Просьба хозяина: «{request[:600]}»\n{PRIVATE_NOTE}"},
                {"role": "user", "content": "Напиши строку журнала (или SKIP)."}]
    return [{"role": "system", "content": WRITER},
            {"role": "user", "content": f"Просьба хозяина: «{request[:600]}»\nТвой ответ: «{answer[:800]}»"}]


def entry_for(request: str, answer: str, tools: list[str]) -> str:
    """The journal line for a turn ("" = not worth keeping), written by the auxiliary model."""
    from agent.auxiliary_client import call_llm
    resp = call_llm(task="voice_log", messages=writer_messages(request, answer, tools), max_tokens=400,
                    temperature=None, timeout=60, reasoning_config={"enabled": False})
    text = str(resp.choices[0].message.content or "").strip()
    text = re.sub(r"\s+", " ", re.sub(r"<think>.*?</think>", "", text, flags=re.S)).strip().strip("«»\"")
    if not text or text.upper().startswith("SKIP") or len(text) < 4:
        return ""
    return text[:240]


def append(directory: Path, line: str, when: float | None = None) -> Path:
    t = time.localtime(when)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / time.strftime("%Y-%m-%d.md", t)
    with _lock, open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(f"# Голос, {t.tm_mday} {MONTHS[t.tm_mon - 1]} {t.tm_year}\n\n")
        f.write(f"- {time.strftime('%H:%M', t)} — {line}\n")
    try:
        os.chmod(path, 0o664)
    except OSError:
        pass
    return path


def record(cfg: dict, user_message: Any, assistant_response: Any, history: list) -> None:
    """After a voice turn: its journal line, in a thread of this turn's context (the profile is context-local)."""
    directory = log_dir(cfg)
    request = str(user_message or "").replace(LIVE_MARK, "").strip()
    if directory is None or not request:
        return
    answer = str(assistant_response or "").strip()
    tools = _tool_names(this_turn(history or []))
    when = time.time()

    def write():
        try:
            line = entry_for(request, answer, tools)
            if line:
                append(directory, line, when)
                log.info("voice-surface: voice log: a line for this turn (%d chars)", len(line))
        except Exception as e:
            log.warning("voice-surface: voice log failed: %r", e)
    threading.Thread(target=contextvars.copy_context().run, args=(write,), daemon=True).start()


# ---- the voice_log tool ----------------------------------------------------------------------------------------

TOOL_DESCRIPTION = (
    "The owner's past voice requests: a short journal kept automatically, one line per request (date, time, what "
    "was asked and done; no personal details). Use it when the owner asks about something from earlier ('что я "
    "просил вчера?', 'когда мы заказывали пиццу?', 'что ты тогда нашла про билеты?'). Returns the lines of the last "
    "`days` days, newest last, those with any of the words of `query` when it is given.")
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "days": {"type": "integer", "description": "How many days back, 1-60 (default 7; 1 = today)."},
        "query": {"type": "string", "description": "Optional words to look for (any of them), e.g. 'пицца заказ'."},
    },
}


def read(cfg: dict, days: int = 7, query: str = "") -> dict:
    directory = log_dir(cfg)
    if directory is None:
        return {"status": "off"}
    days = max(1, min(int(days or 7), 60))
    words = [w for w in re.findall(r"\w+", (query or "").lower()) if len(w) > 2]
    stems = [w[:max(4, len(w) - 2)] for w in words]  # "пиццу" finds "пицца"
    out = []
    for back in range(days - 1, -1, -1):
        d = date.today() - timedelta(days=back)
        try:
            lines = (directory / f"{d.isoformat()}.md").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("- ") and (not stems or any(s in line.lower() for s in stems)):
                out.append(f"{d.isoformat()} {line[2:]}")
    text = "\n".join(out)
    if len(text) > 6000:
        text = "…\n" + text[-6000:]
    return {"status": "ok", "days": days, "lines": text or "(ничего)"}


def tool(cfg_fn, args: dict, **_: Any) -> str:
    try:
        return json.dumps(read(cfg_fn(), args.get("days") or 7, str(args.get("query") or "")), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)[:200]})
