"""The owner surface for voice sessions: questions go to the Live bridge, which asks through the speakers.

Hermes keeps its human prompts (one-time code, card fill confirmation) off for api_server sessions: nobody there
can answer. In this profile's api_server sessions the Live bridge can, so:

  * ``can_prompt_here`` is true and the code prompt asks the bridge (kind "code": only the digits come back;
    Hermes types them into the page, the model never sees them);
  * a card fill is confirmed by voice, naming the card and the site;
  * the ``ask_owner`` tool lets the agent ask a yes / no or a short question (before it orders or pays);
  * after every turn the bridge is told the answer is ready (/result); when the Live dialog that asked has ended
    (the request went on in the background) and no other one is in progress, the bridge's short spoken summary of
    it is announced through Home Assistant (``wake_script``), and the dialog that opens goes on with that job;
  * website tasks and requests that went on in the background think: their model calls carry
    ``hermes_live_think``, which the model router in front of the agent turns into its thinking model
    (litellm-taint, ``cloud-think``) - except the step right after a browser action that worked (a click, typing:
    what comes next is mechanical, usually a look at the page), which stays on the fast model ("system 1"), at most
    SYSTEM1_MAX times in a row.

When there is no live dialog for the session, the bridge says "wake" and the speakers are woken the same way, with
the question as the text; the dialog that opens takes the answer.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

log = logging.getLogger("voice_surface")

PLUGIN_ID = "voice-surface"
THINK_FIELD = "hermes_live_think"  # read (and removed) by the model router
# Browser tools that act on a page and return no page: the step after one that worked is mechanical.
ACTIONS = frozenset({"browser_click", "browser_type", "browser_press", "browser_scroll", "browser_back",
                     "browser_vault_fill", "browser_vault_enter_code", "browser_dialog"})
SYSTEM1_MAX = 3
_TAG_RE = re.compile(r"\[(?:emo:[a-z]+|SILENT)\]", re.I)
_installed = False


# ---- settings / context ----------------------------------------------------------------------------------------

def settings() -> dict:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    except Exception as e:
        log.warning("voice-surface: reading the config failed, using the defaults: %r", e)
        cfg = {}
    entry = ((cfg.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}
    raw = entry.get("settings") if isinstance(entry, dict) else None
    out = {"bridge_url": "http://192.168.1.50:10702", "profile": "voice", "wake_script": "script.live_reminder"}
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v})
    return out


def _home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _token() -> str:
    try:
        return (_home() / "voice_surface.token").read_text().strip()
    except OSError:
        return ""


def _session_env(name: str) -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, "")
    except Exception:
        return os.environ.get(name, "")


def voice_session() -> bool:
    """An api_server session of this plugin's profile (the voice agent behind the Live bridge)."""
    try:
        return _session_env("HERMES_SESSION_PLATFORM") == "api_server" and _home().name == settings()["profile"]
    except Exception:
        return False


def available() -> bool:
    return voice_session() and bool(_token())


# ---- the bridge and Home Assistant -----------------------------------------------------------------------------

def _bridge(method: str, path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(settings()["bridge_url"].rstrip("/") + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _ha() -> tuple:
    """Home Assistant's address and Authorization header: the ones the profile's homeassistant MCP server uses
    (its url and headers, with ${VAR} resolved from the profile's secrets the way Hermes resolves them), unless
    HASS_URL / HASS_TOKEN are set for the profile."""
    url, auth = "", ""
    try:
        from agent.secret_scope import get_secret
        url = (get_secret("HASS_URL", "") or "").rstrip("/")
        token = get_secret("HASS_TOKEN", "") or ""
        auth = f"Bearer {token}" if token else ""
    except Exception:
        pass
    try:
        from urllib.parse import urlsplit
        from hermes_cli.config import load_config_readonly
        from tools.mcp_tool_config import _interpolate_env_vars
        srv = ((load_config_readonly().get("mcp_servers") or {}).get("homeassistant") or {})
        if not url and srv.get("url"):
            u = urlsplit(str(srv["url"]))
            url = f"{u.scheme}://{u.netloc}"
        if not auth:
            auth = str(_interpolate_env_vars((srv.get("headers") or {}).get("Authorization") or ""))
    except Exception as e:
        log.warning("voice-surface: reading Home Assistant's address failed: %r", e)
    if "${" in auth:
        auth = ""
    return url, auth


def wake(text: str) -> bool:
    """Make the speakers start a dialog that says `text` (Home Assistant script, field "message"). Off while the
    profile has a ``voice_surface.nowake`` file (tests, quiet hours): questions then fail instead of waking anyone."""
    script = settings()["wake_script"].split(".", 1)[-1]
    if (_home() / "voice_surface.nowake").exists():
        log.info("voice-surface: not waking the speakers (script.%s, %d chars): %s/voice_surface.nowake is set",
                 script, len(text), _home().name)
        return False
    url, auth = _ha()
    if not url or not auth:
        log.warning("voice-surface: no Home Assistant address / token for this profile, cannot wake the speakers")
        return False
    log.info("voice-surface: waking the speakers with script.%s (%d chars)", script, len(text))
    req = urllib.request.Request(f"{url}/api/services/script/{script}", method="POST",
                                 data=json.dumps({"message": text}).encode(),
                                 headers={"Authorization": auth, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
        return True
    except (urllib.error.URLError, OSError) as e:
        log.warning("voice-surface: waking the speakers failed: %r", e)
        return False


def ask_owner(kind: str, text: str, timeout: float = 120.0) -> dict:
    """Ask the owner through the speakers; {"status": answered|unclear|timeout|no_answer|error, "answer"}.
    kind "code" returns digits, "confirm" yes|no, "text" the transcript."""
    session = _session_env("HERMES_SESSION_ID")
    try:
        a = _bridge("POST", "/ask", {"session": session, "kind": kind, "text": text, "timeout": timeout})
    except (urllib.error.URLError, OSError, ValueError) as e:
        log.warning("voice-surface: the bridge is unreachable: %r", e)
        return {"status": "error", "answer": ""}
    log.info("voice-surface: ask %s (%s) for %s%s", a.get("id"), kind, session, ", waking the speakers" if a.get("wake") else "")
    if a.get("wake") and not wake(a.get("text") or text):
        return {"status": "error", "answer": ""}
    deadline = time.monotonic() + timeout + 5
    while time.monotonic() < deadline:
        try:
            res = _bridge("GET", f"/ask/{a['id']}?wait=25", timeout=35)
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("voice-surface: waiting for the answer failed: %r", e)
            time.sleep(2)
            continue
        if res.get("status") != "pending":
            log.info("voice-surface: ask %s: %s", a.get("id"), res.get("status"))
            return res
    return {"status": "timeout", "answer": ""}


# ---- Hermes's prompts, rerouted -------------------------------------------------------------------------------

def _site(origin: str) -> str:
    return (origin or "").split("://", 1)[-1].split("/", 1)[0] or "сайт"


def _code_prompt(site: str, hint: str = "") -> str:
    res = ask_owner("code", f"Продиктуй код, который прислал {_site(site)}.", timeout=180)
    return res.get("answer", "") if res.get("status") == "answered" else ""


def install() -> None:
    """Patch Hermes's prompt points for voice sessions (idempotent; everything else is left as it was)."""
    global _installed
    if _installed:
        return
    _installed = True
    from agent.vault_backends import unlock

    orig_can, orig_code = unlock.can_prompt_here, unlock.get_code_prompt_callback

    def can_prompt_here() -> bool:
        return True if available() else orig_can()

    def get_code_prompt_callback():
        return _code_prompt if available() else orig_code()

    unlock.can_prompt_here = can_prompt_here
    unlock.get_code_prompt_callback = get_code_prompt_callback

    try:
        from tools import browser_vault_tool as bvt
    except Exception:
        log.warning("voice-surface: browser vault tools not importable; card confirmation by voice is off")
        return
    orig_confirm = bvt._confirm_payment_fill

    def _confirm_payment_fill(label: str, origin: str) -> bool:
        if not available():
            return orig_confirm(label, origin)
        res = ask_owner("confirm", f"Ввести карту «{label}» для оплаты на сайте {_site(origin)}? Да или нет?", timeout=120)
        return res.get("status") == "answered" and res.get("answer") == "yes"

    bvt._confirm_payment_fill = _confirm_payment_fill
    log.info("voice-surface: installed (code prompt, card confirmation by voice)")


# ---- after every turn: is anyone still listening for this answer? ----------------------------------------------

def post_llm_call(session_id: str = "", assistant_response: Any = "", platform: str = "", **_: Any):
    try:
        if platform != "api_server" or not _token() or _home().name != settings()["profile"]:
            return None
        text = _TAG_RE.sub("", str(assistant_response or "")).strip()
        if not session_id or not text:
            return None
        res = _bridge("POST", "/result", {"session": session_id, "text": text[:2000]}, timeout=15)
        if res.get("announce"):
            say = str(res.get("text") or "").strip() or text[:600]  # the bridge's spoken summary of it
            log.info("voice-surface: the dialog of %s has ended; announcing its answer", session_id)
            # the profile (Hermes home, config, secrets) is context-local: the thread runs in this turn's context
            threading.Thread(target=contextvars.copy_context().run, args=(wake, say), daemon=True).start()
    except Exception as e:  # never break the turn
        log.warning("voice-surface: post_llm_call failed: %r", e)
    return None


# ---- website tasks and requests that went on in the background think ----------------------------------------

_think: set = set()        # sessions whose model calls think (they never go back)
_streak: dict = {}         # session -> fast ("system 1") calls in a row


def _browser_used(messages) -> bool:
    return any(str((tc.get("function") or {}).get("name") or "").startswith("browser_")
               for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
               for tc in m.get("tool_calls") or [] if isinstance(tc, dict))


def _routine(messages) -> bool:
    """The last tool results answer only browser actions, all of which worked."""
    i, results = len(messages) - 1, []
    while i >= 0 and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
        results.append(messages[i])
        i -= 1
    if not results or i < 0 or messages[i].get("role") != "assistant":
        return False
    names = {tc.get("id"): (tc.get("function") or {}).get("name") for tc in messages[i].get("tool_calls") or []
             if isinstance(tc, dict)}
    for r in results:
        if names.get(r.get("tool_call_id")) not in ACTIONS:
            return False
        c = r.get("content")
        text = (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)).replace(" ", "")
        if '"success":false' in text or '"success":true' not in text:
            return False
    return True


def llm_request(request: Any = None, session_id: str = "", platform: str = "", **_: Any):
    """``llm_request`` middleware for the Live bridge's sessions: website tasks (the dialog model said so, or the
    agent has used the browser) and requests that went on in the background (nobody waits) ask the router for the
    thinking model, but for the mechanical step after a browser action."""
    try:
        if platform != "api_server" or not str(session_id).startswith("live-") or not isinstance(request, dict):
            return None
        messages = request.get("messages") or []
        if session_id not in _think:
            if _home().name != settings()["profile"] or not _token():
                return None
            if _browser_used(messages):
                why = "the agent uses the browser"
            else:
                st = _bridge("GET", f"/session/{session_id}", timeout=0.5)
                why = "a website task" if st.get("think") else "in the background" if st.get("background") else ""
            if not why:
                return None
            if len(_think) > 1000:
                _think.clear()
                _streak.clear()
            _think.add(session_id)
            log.info("voice-surface: %s thinks from now on (%s)", session_id, why)
        if _routine(messages) and _streak.get(session_id, 0) < SYSTEM1_MAX:
            _streak[session_id] = _streak.get(session_id, 0) + 1
            return None
        _streak[session_id] = 0
        req = dict(request)
        req["extra_body"] = dict(req.get("extra_body") or {}, **{THINK_FIELD: True})
        return {"request": req, "source": PLUGIN_ID, "reason": "Live request that thinks"}
    except Exception as e:  # never break the call
        log.debug("voice-surface: llm_request: %r", e)
        return None


# ---- the ask_owner tool ----------------------------------------------------------------------------------------

ASK_OWNER_DESCRIPTION = (
    "Ask the owner a short question out loud through the home speakers and wait for the spoken answer (up to two "
    "minutes). Use kind=confirm before anything that orders, books or pays: say what exactly and the total "
    "(\"Две пиццы, тысяча двести девяносто рублей, доставка к восьми. Заказываю?\") and go on only on yes. "
    "kind=text for a short choice you cannot make yourself (which item, which address). Do not use it for one-time "
    "codes: the vault code tool asks for those without you seeing them. Russian, one or two short sentences, "
    "understandable on their own: the owner may be busy with another conversation by then, so name what it is "
    "about (\"Пицца: на какой адрес везти?\").")
ASK_OWNER_PARAMETERS = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "What to say, in Russian, ending with the question."},
        "kind": {"type": "string", "enum": ["confirm", "text"], "description": "confirm = yes/no; text = a short answer."},
    },
    "required": ["question", "kind"],
}


def ask_owner_tool(args: dict, **_: Any) -> str:
    kind = args.get("kind") if args.get("kind") in ("confirm", "text") else "confirm"
    question = str(args.get("question") or "").strip()
    if not question:
        return json.dumps({"status": "error", "error": "question is empty"})
    return json.dumps(ask_owner(kind, question), ensure_ascii=False)
