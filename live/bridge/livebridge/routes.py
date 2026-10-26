"""Which backend takes a dialog: routes.json, re-read whenever it changes (checked for every new dialog).

    {
      "default": "pipeline",
      "fallback": "handoff",
      "stations": {
        "kitchen*": {"backend": "gpt-live", "options": {"voice": "marin"}},
        "bedroom":  "handoff"
      }
    }

"stations" maps fnmatch patterns on the name the station sends in its hello to a backend (a string, or an object with
"options" for the backend; the agent's options - agent, ha_agent_id, ha_device_id, ha_satellite_id - go on to agent.py);
the first match wins, otherwise "default". "fallback" (default "handoff") takes the dialog
when the chosen backend cannot start; "none" ends it instead. A file that is missing keeps the built-in defaults, one
that does not parse keeps the last good version (and logs).
"""
import fnmatch
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("live")

DEFAULT_BACKEND = os.environ.get("LIVE_DEFAULT_BACKEND", "pipeline")


class Routes:
    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self.default = DEFAULT_BACKEND
        self.fallback = "handoff"
        self.stations: list[tuple[str, str, dict]] = []
        self._mtime = None

    def _load(self):
        if not self.path:
            return
        try:
            mtime = self.path.stat().st_mtime_ns
        except OSError:
            if self._mtime is not None:
                log.warning("routes file %s is gone, keeping the last version", self.path)
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            default = str(doc.get("default") or DEFAULT_BACKEND)
            fallback = str(doc.get("fallback") or "handoff")
            stations = []
            for pat, v in (doc.get("stations") or {}).items():
                if isinstance(v, str):
                    stations.append((pat, v, {}))
                else:
                    stations.append((pat, str(v["backend"]), dict(v.get("options") or {})))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log.warning("routes file %s not used: %r", self.path, e)
            return
        self.default, self.fallback, self.stations = default, fallback, stations
        log.info("routes: default %s, fallback %s, %s", default, fallback,
                 ", ".join(f"{p} -> {b}" for p, b, _ in stations) or "no per-station rules")

    def pick(self, station: str) -> tuple[str, dict]:
        """(backend name, its options) for a station name."""
        self._load()
        for pat, backend, options in self.stations:
            if fnmatch.fnmatchcase(station, pat):
                return backend, options
        return self.default, {}
