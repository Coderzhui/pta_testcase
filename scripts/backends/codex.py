"""Codex CLI backend implementation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.backends.base import CliBackend


class CodexBackend(CliBackend):

    @property
    def name(self) -> str:
        return "codex"

    def exec_prompt(
        self,
        prompt: str,
        *,
        summary_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "codex",
            "exec",
            "--cd",
            ".",
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-last-message",
            str(summary_path),
            "-",
        ]
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return completed

    @property
    def repo_config_dir(self) -> str:
        return ".codex"

    @property
    def agents_dir(self) -> str:
        return ".codex/agents"

    @property
    def skills_dir(self) -> str:
        return ".codex/skills"

    @property
    def session_log_dir(self) -> Path | None:
        path = Path.home() / ".codex" / "sessions"
        return path if path.exists() else None
