"""Temporarily enable or disable mock ticketing inside evaluation processes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from collections.abc import Iterator

from app.core import config
from app.mcp import server as mcp_server
from app.services import ticketing as ticketing_mod
from app.services.ticketing import reset_ticket_adapter


@contextmanager
def ticketing_enabled(enabled: bool) -> Iterator[None]:
    original = config.settings
    updated = replace(original, ticketing_enabled=enabled)
    config.settings = updated
    mcp_server.settings = updated
    ticketing_mod.settings = updated
    reset_ticket_adapter()
    try:
        yield
    finally:
        config.settings = original
        mcp_server.settings = original
        ticketing_mod.settings = original
        reset_ticket_adapter()
