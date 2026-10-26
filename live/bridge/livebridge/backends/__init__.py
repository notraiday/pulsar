"""Dialog backends. A backend is a module with

    async def run(station: Station, http: aiohttp.ClientSession, options: dict) -> None

that owns the dialog until it ends. It may raise BackendUnavailable (or any exception) before it has sent anything to
the station; the bridge then hands the dialog off to the station's own pipeline (see routes.py: "fallback").

Backends are found by name in this package (a '-' in the name is read as '_', so "gpt-live" is gpt_live.py) and are
imported on first use: dropping a new file here adds a backend without touching the rest, and routes.json (re-read
for every dialog) chooses which station gets it. Code changes of a backend that is already loaded need a restart.
"""
import importlib

from ..util import BackendUnavailable

ALIASES = {"test": "selftest"}


def load(name: str):
    mod = ALIASES.get(name, name).replace("-", "_")
    if not mod.isidentifier():
        raise BackendUnavailable(f"bad backend name {name!r}")
    try:
        m = importlib.import_module(f"{__name__}.{mod}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{mod}":
            raise  # the backend exists but one of its own imports is missing
        raise BackendUnavailable(f"no backend {name!r}") from None
    if not hasattr(m, "run"):
        raise BackendUnavailable(f"backend {name!r} has no run()")
    return m
