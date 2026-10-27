"""Vaultwarden (the Bitwarden CLI) for sessions that cannot type a master password.

* Unlock: the owner runs ``hermes-bw-unlock`` on this host once (until reboot or ``hermes-bw-lock``); it leaves only
  the ``bw`` session token in RAM, ``/run/user/<uid>/hermes-bw/session``. Whenever Hermes's Bitwarden backend finds
  no session of its own, it takes that one; when the file is gone, the session taken from it is dropped too.
* Cards and identities: Hermes's Bitwarden backend lists logins only. Cards (type 3) become ``payment`` items and
  identities with an address (type 4) ``address`` items. Like every payment / address item they are filled only on
  the sites they are bound to: a custom field ``hermes_sites`` on the item, origins separated by commas or spaces
  (``https://dodopizza.ru, https://pay.example.com``). Without it the item is listed but not fillable.
* ``bw list items`` is cached for a few seconds: one listing serves the logins and the cards.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("voice_surface")

SITES_FIELD = "hermes_sites"
_LIST_CACHE_S = 15.0
_installed = False
_injected: Optional[str] = None  # the token taken from the file (dropped again when the file goes)


def token_file() -> Path:
    return Path(f"/run/user/{os.getuid()}/hermes-bw/session")


def _file_token() -> str:
    try:
        return token_file().read_text().strip()
    except OSError:
        return ""


def _sites(item: dict) -> List[str]:
    from agent.vault_store import normalize_origin
    out: List[str] = []
    for f in item.get("fields") or []:
        if str(f.get("name") or "").strip().lower() != SITES_FIELD:
            continue
        for raw in re.split(r"[\s,;]+", str(f.get("value") or "")):
            try:
                origin = normalize_origin(raw) if raw else ""
            except Exception:
                origin = ""
            if origin.startswith("https://") and origin not in out:
                out.append(origin)
    return out


def _meta(backend, item: dict):
    from agent.vault_store import VaultItemMeta
    kind = {3: "payment", 4: "address"}.get(item.get("type"))
    if kind is None:
        return None
    if kind == "payment":
        card = item.get("card") or {}
        if not card.get("number"):
            return None
        brand = str(card.get("brand") or "карта")
        label = f"{item.get('name') or brand} ({brand})"
    else:
        ident = item.get("identity") or {}
        if not (ident.get("address1") or ident.get("city")):
            return None
        label = str(item.get("name") or "адрес")
    sites = _sites(item)
    return VaultItemMeta(id=f"{backend.prefix}{item.get('id')}", kind=kind, label=label,
                         origin=sites[0] if sites else None, created_at=str(item.get("creationDate") or ""),
                         allowed_origins=tuple(sites))


def _secret(item: dict) -> Optional[Dict[str, str]]:
    if item.get("type") == 3:
        c = item.get("card") or {}
        return {"card_number": str(c.get("number") or "").replace(" ", ""),
                "cardholder_name": str(c.get("cardholderName") or ""),
                "exp_month": str(c.get("expMonth") or "").zfill(2), "exp_year": str(c.get("expYear") or ""),
                "cvc": str(c.get("code") or ""), "billing_postal_code": ""}
    if item.get("type") == 4:
        i = item.get("identity") or {}
        return {"address_line1": str(i.get("address1") or ""),
                "address_line2": " ".join(str(x) for x in (i.get("address2"), i.get("address3")) if x),
                "city": str(i.get("city") or ""), "state": str(i.get("state") or ""),
                "postal_code": str(i.get("postalCode") or ""), "country": str(i.get("country") or "")}
    return None


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    from agent.vault_backends import unlock
    from agent.vault_backends.bitwarden import BitwardenLoginBackend as B

    orig_live = unlock._live

    def _live(backend: str, *, touch: bool):
        global _injected
        if backend != B.name:
            return orig_live(backend, touch=touch)
        token = orig_live(backend, touch=touch)
        on_file = _file_token()
        if token and token == _injected and not on_file:  # hermes-bw-lock: the file is gone
            unlock.lock(backend)
            _injected = None
            return None
        if on_file and token != on_file and (token is None or token == _injected):
            unlock.store_session_token(backend, on_file)  # unlocked (again) by hermes-bw-unlock
            _injected = on_file
            token = orig_live(backend, touch=touch)
        return token

    unlock._live = _live

    orig_run = B._run
    cache: Dict[str, tuple] = {}

    def _run(self, *args: str) -> str:
        if args != ("list", "items"):
            return orig_run(self, *args)
        key = unlock.get_session_token(self.name) or ""
        hit = cache.get("list")
        if hit and hit[0] == key and time.monotonic() - hit[1] < _LIST_CACHE_S:
            return hit[2]
        out = orig_run(self, *args)
        cache["list"] = (key, time.monotonic(), out)
        return out

    B._run = _run

    orig_list, orig_secret = B.list_items, B.resolve_secret

    def list_items(self):
        out = orig_list(self)
        if not self.is_unlocked():
            return out
        try:
            raw = json.loads(self._run("list", "items") or "[]")
        except Exception as e:
            log.warning("voice-surface: listing Vaultwarden cards failed: %r", e)
            return out
        for item in raw if isinstance(raw, list) else []:
            meta = _meta(self, item)
            if meta is not None:
                out.append(meta)
        return out

    def resolve_secret(self, handle: str) -> Dict[str, str]:
        item = json.loads(self._run("get", "item", handle[len(self.prefix):]) or "{}")
        return _secret(item) or orig_secret(self, handle)

    B.list_items = list_items
    B.resolve_secret = resolve_secret
    log.info("voice-surface: Vaultwarden session from %s, cards and identities listed", token_file())
