"""Abstract base class for AI CLI backends."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class CliBackend(ABC):
    """Unified interface for invoking AI CLI coding agents.

    Each concrete subclass wraps a specific CLI tool (Codex, Copilot, …)
    and knows how to send a prompt, capture output, and locate tool-specific
    configuration / session directories.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in CLI flags and log messages (e.g. ``"codex"``)."""

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def exec_prompt(
        self,
        prompt: str,
        *,
        summary_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Send *prompt* to the AI CLI and return the completed process.

        Parameters
        ----------
        prompt:
            The full prompt text to send (typically via *stdin*).
        summary_path:
            File where the CLI should write its final-message / summary output.
        stdout_path:
            Path to persist the CLI's standard output.
        stderr_path:
            Path to persist the CLI's standard error.
        cwd:
            Working directory for the subprocess.
        """

    # ------------------------------------------------------------------
    # Configuration directories
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def repo_config_dir(self) -> str:
        """Relative repo path containing this backend's config (e.g. ``".codex"``)."""

    @property
    @abstractmethod
    def agents_dir(self) -> str:
        """Relative repo path to agent definitions (e.g. ``".codex/agents"``)."""

    @property
    @abstractmethod
    def skills_dir(self) -> str:
        """Relative repo path to skill definitions (e.g. ``".codex/skills"``)."""

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    @property
    def supports_skills(self) -> bool:
        """Whether this backend natively discovers and invokes SKILL.md skills."""
        return True

    # ------------------------------------------------------------------
    # Debug / session logs
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def session_log_dir(self) -> Path | None:
        """Absolute path to the session-log directory, or ``None`` if unavailable."""
