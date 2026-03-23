"""CLI backend abstraction layer for AI coding agents."""

from __future__ import annotations

from scripts.backends.base import CliBackend
from scripts.backends.codex import CodexBackend
from scripts.backends.copilot import CopilotBackend

SUPPORTED_BACKENDS = {"codex", "copilot"}


def get_backend(name: str) -> CliBackend:
    """Return a *CliBackend* instance by identifier.

    Supported values: ``"codex"``, ``"copilot"``.
    """
    if name == "codex":
        return CodexBackend()
    if name == "copilot":
        return CopilotBackend()
    raise ValueError(
        f"unsupported CLI backend {name!r}; choose from {sorted(SUPPORTED_BACKENDS)}"
    )


__all__ = [
    "CliBackend",
    "CodexBackend",
    "CopilotBackend",
    "SUPPORTED_BACKENDS",
    "get_backend",
]
