from __future__ import annotations

from typing import Any


def append_trace(
    state: dict[str, Any],
    *,
    agent: str,
    outcome: str,
    summary: str,
    issues: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    steps = list(state.get("agent_trace") or [])
    steps.append(
        {
            "agent": agent,
            "outcome": outcome,
            "summary": summary[:300],
            "issues": issues or [],
        }
    )
    return steps
