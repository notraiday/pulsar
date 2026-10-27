"""voice-surface: the Live speakers as the voice agent's way to the owner (see plugin.yaml)."""
from __future__ import annotations

from typing import Any


def register(ctx: Any) -> None:
    from . import core, vaultwarden

    core.install()
    vaultwarden.install()
    ctx.register_hook("post_llm_call", core.post_llm_call)
    ctx.register_middleware("llm_request", core.llm_request)
    ctx.register_tool(
        name="ask_owner",
        toolset="voice_surface",
        schema={"name": "ask_owner", "description": core.ASK_OWNER_DESCRIPTION, "parameters": core.ASK_OWNER_PARAMETERS},
        handler=core.ask_owner_tool,
        check_fn=core.available,
        is_async=False,
        description=core.ASK_OWNER_DESCRIPTION,
        emoji="🔔",
    )


__all__ = ["register"]
