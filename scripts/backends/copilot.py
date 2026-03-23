"""GitHub Copilot CLI backend implementation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.backends.base import CliBackend

# Override the base command via environment variable, e.g.
#   PTA_COPILOT_CMD="gh copilot"  (space-separated tokens)
_COPILOT_CMD = os.environ.get("PTA_COPILOT_CMD", "copilot")

# Default log directory mirrors ~/.copilot/logs/
_DEFAULT_LOG_DIR = Path.home() / ".copilot" / "logs"


class CopilotBackend(CliBackend):

    @property
    def name(self) -> str:
        return "copilot"

    def exec_prompt(
        self,
        prompt: str,
        *,
        summary_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        base_cmd = _COPILOT_CMD.split()

        summary_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            *base_cmd,
            # Non-interactive: -p exits after completion.
            "-p",
            prompt,
            # Auto-approve all tools/paths/urls so it runs unattended.
            "--allow-all",
            # Save session transcript to the summary path.
            f"--share={summary_path}",
            # Silent mode: only agent response on stdout (no TUI chrome).
            "-s",
        ]

        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        return completed

    # ---- configuration directories ----

    @property
    def repo_config_dir(self) -> str:
        return ".github"

    @property
    def agents_dir(self) -> str:
        return ".github/agents"

    @property
    def skills_dir(self) -> str:
        return ".github/skills"

    # ---- session logs ----

    @property
    def session_log_dir(self) -> Path | None:
        return _DEFAULT_LOG_DIR if _DEFAULT_LOG_DIR.exists() else None
