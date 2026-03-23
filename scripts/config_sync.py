#!/usr/bin/env python3
"""Configuration synchronization tool for Codex CLI ↔ Copilot CLI config formats.

Converts agent definitions, skills, and project-level instructions between:
  - .codex/  (Codex CLI: TOML agents, SKILL.md skills, config.toml)
  - .github/ (Copilot CLI: .agent.md agents, SKILL.md skills, copilot-instructions.md)

Usage:
    python -m scripts.config_sync --from codex --to copilot [--dry-run]
    python -m scripts.config_sync --from copilot --to codex [--dry-run]
    python -m scripts.config_sync --diff
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

_CODEX_AGENTS_DIR = ".codex/agents"
_CODEX_SKILLS_DIR = ".codex/skills"
_CODEX_AGENTS_MD = "AGENTS.md"

_COPILOT_AGENTS_DIR = ".github/agents"
_COPILOT_SKILLS_DIR = ".github/skills"
_COPILOT_INSTRUCTIONS = ".github/copilot-instructions.md"

# Tool mapping between Codex sandbox_mode and Copilot tools list
_READONLY_TOOLS = ["read", "search"]
_READWRITE_TOOLS = ["read", "edit", "shell", "search"]


# ---------------------------------------------------------------------------
# Codex → Copilot
# ---------------------------------------------------------------------------

def codex_agent_toml_to_copilot_md(toml_path: Path) -> str:
    """Convert a ``.codex/agents/*.toml`` file to a Copilot ``.agent.md`` string.

    Parameters
    ----------
    toml_path:
        Path to the Codex agent TOML file.

    Returns
    -------
    str
        The full content of the equivalent Copilot ``.agent.md`` file.
    """
    with open(toml_path, "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)

    name: str = data.get("name", toml_path.stem)
    description: str = data.get("description", "")
    sandbox_mode: str = data.get("sandbox_mode", "")
    developer_instructions: str = data.get("developer_instructions", "")

    # Determine tools list from sandbox_mode
    if sandbox_mode == "read-only":
        tools = _READONLY_TOOLS
    else:
        tools = _READWRITE_TOOLS

    # Build YAML frontmatter dict
    frontmatter: dict[str, Any] = {
        "name": name,
        "description": description,
        "tools": tools,
    }

    # Preserve optional metadata as comments or extra fields
    model = data.get("model")
    reasoning_effort = data.get("model_reasoning_effort")
    if model:
        frontmatter["model"] = model
    if reasoning_effort:
        frontmatter["model_reasoning_effort"] = reasoning_effort

    fm_text = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    ).rstrip("\n")

    # Build markdown body
    body_parts: list[str] = []
    if developer_instructions.strip():
        body_parts.append("## Instructions\n")
        body_parts.append(developer_instructions.strip())

    body = "\n".join(body_parts)

    return f"---\n{fm_text}\n---\n\n{body}\n"


# ---------------------------------------------------------------------------
# Copilot → Codex
# ---------------------------------------------------------------------------

def _parse_agent_md(text: str) -> tuple[dict[str, Any], str]:
    """Split an ``.agent.md`` file into (frontmatter_dict, markdown_body)."""
    pattern = re.compile(r"\A---\n(.*?\n)---\n(.*)", re.DOTALL)
    m = pattern.match(text)
    if not m:
        raise ValueError("Could not parse YAML frontmatter from agent.md content")
    fm_raw, body = m.group(1), m.group(2)
    fm: dict[str, Any] = yaml.safe_load(fm_raw) or {}
    return fm, body


def _extract_instructions(body: str) -> str:
    """Extract the content under ``## Instructions`` heading."""
    pattern = re.compile(r"^## Instructions\s*\n", re.MULTILINE)
    m = pattern.search(body)
    if not m:
        # Fall back: return entire body stripped
        return body.strip()
    start = m.end()
    # Find the next heading at level 2 or higher, if any
    next_heading = re.search(r"^## ", body[start:], re.MULTILINE)
    if next_heading:
        return body[start : start + next_heading.start()].strip()
    return body[start:].strip()


def _format_toml_string(value: str, multiline: bool = False) -> str:
    """Format a Python string as a TOML value."""
    if multiline or "\n" in value:
        return f'"""\n{value}\n"""'
    # Escape backslashes and double-quotes for single-line strings
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def copilot_agent_md_to_codex_toml(md_path: Path) -> str:
    """Convert a ``.github/agents/*.agent.md`` file to a Codex TOML string.

    Parameters
    ----------
    md_path:
        Path to the Copilot agent markdown file.

    Returns
    -------
    str
        The full content of the equivalent Codex agent ``.toml`` file.
    """
    text = md_path.read_text(encoding="utf-8")
    fm, body = _parse_agent_md(text)

    name: str = fm.get("name", md_path.stem.removesuffix(".agent"))
    description: str = fm.get("description", "").strip()
    tools: list[str] = fm.get("tools", [])
    model: str = fm.get("model", "")
    reasoning_effort: str = fm.get("model_reasoning_effort", "")

    instructions = _extract_instructions(body)

    # Determine sandbox_mode from tools
    tool_set = set(tools)
    is_readonly = tool_set <= {"read", "search"}

    lines: list[str] = []
    lines.append(f'name = "{name}"')
    lines.append(f"description = {_format_toml_string(description)}")
    if model:
        lines.append(f'model = "{model}"')
    if reasoning_effort:
        lines.append(f'model_reasoning_effort = "{reasoning_effort}"')
    if is_readonly:
        lines.append('sandbox_mode = "read-only"')
    if instructions:
        lines.append(f"developer_instructions = {_format_toml_string(instructions, multiline=True)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Skills sync
# ---------------------------------------------------------------------------

def sync_skills(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Copy SKILL.md files from *src_dir* to *dst_dir*.

    The directory tree under *src_dir* is mirrored into *dst_dir*.

    Parameters
    ----------
    src_dir:
        Source skills directory (e.g. ``.codex/skills``).
    dst_dir:
        Destination skills directory (e.g. ``.github/skills``).

    Returns
    -------
    list[Path]
        Paths of the copied files (destination paths).
    """
    copied: list[Path] = []
    if not src_dir.is_dir():
        return copied

    for src_file in sorted(src_dir.rglob("SKILL.md")):
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file)
    return copied


# ---------------------------------------------------------------------------
# copilot-instructions.md generation
# ---------------------------------------------------------------------------

def generate_copilot_instructions(agents_md_path: Path, output_path: Path) -> None:
    """Generate ``.github/copilot-instructions.md`` that includes AGENTS.md content.

    Parameters
    ----------
    agents_md_path:
        Path to the root ``AGENTS.md`` file.
    output_path:
        Path where ``copilot-instructions.md`` will be written.
    """
    if not agents_md_path.is_file():
        raise FileNotFoundError(f"AGENTS.md not found at {agents_md_path}")

    agents_content = agents_md_path.read_text(encoding="utf-8").strip()

    header = textwrap.dedent("""\
        <!-- Auto-generated by scripts/config_sync.py — do not edit manually. -->
        <!-- Source: AGENTS.md -->

    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + agents_content + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main sync orchestrator
# ---------------------------------------------------------------------------

def sync_configs(
    repo_root: Path,
    source: str,
    target: str,
    dry_run: bool = False,
) -> list[str]:
    """Synchronize configuration files from *source* format to *target* format.

    Parameters
    ----------
    repo_root:
        Root of the repository.
    source:
        ``"codex"`` or ``"copilot"``.
    target:
        ``"codex"`` or ``"copilot"``.
    dry_run:
        If ``True``, only report what *would* happen without writing files.

    Returns
    -------
    list[str]
        Human-readable list of actions taken (or planned if *dry_run*).
    """
    if source == target:
        raise ValueError("source and target must differ")
    if source not in ("codex", "copilot") or target not in ("codex", "copilot"):
        raise ValueError("source and target must be 'codex' or 'copilot'")

    actions: list[str] = []
    prefix = "[dry-run] " if dry_run else ""

    if source == "codex" and target == "copilot":
        actions.extend(_sync_codex_to_copilot(repo_root, dry_run, prefix))
    else:
        actions.extend(_sync_copilot_to_codex(repo_root, dry_run, prefix))

    return actions


def _sync_codex_to_copilot(repo_root: Path, dry_run: bool, prefix: str) -> list[str]:
    """Convert .codex/ → .github/."""
    actions: list[str] = []

    # --- Agents ---
    src_agents = repo_root / _CODEX_AGENTS_DIR
    dst_agents = repo_root / _COPILOT_AGENTS_DIR
    if src_agents.is_dir():
        for toml_file in sorted(src_agents.glob("*.toml")):
            md_content = codex_agent_toml_to_copilot_md(toml_file)
            dst_file = dst_agents / f"{toml_file.stem}.agent.md"
            action = f"{prefix}convert {toml_file.relative_to(repo_root)} → {dst_file.relative_to(repo_root)}"
            actions.append(action)
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(md_content, encoding="utf-8")

    # --- Skills ---
    src_skills = repo_root / _CODEX_SKILLS_DIR
    dst_skills = repo_root / _COPILOT_SKILLS_DIR
    if src_skills.is_dir():
        if dry_run:
            for skill in sorted(src_skills.rglob("SKILL.md")):
                rel = skill.relative_to(src_skills)
                actions.append(f"{prefix}copy {_CODEX_SKILLS_DIR}/{rel} → {_COPILOT_SKILLS_DIR}/{rel}")
        else:
            copied = sync_skills(src_skills, dst_skills)
            for p in copied:
                actions.append(f"{prefix}copy → {p.relative_to(repo_root)}")

    # --- copilot-instructions.md ---
    agents_md = repo_root / _CODEX_AGENTS_MD
    output_path = repo_root / _COPILOT_INSTRUCTIONS
    if agents_md.is_file():
        action = f"{prefix}generate {_COPILOT_INSTRUCTIONS} from {_CODEX_AGENTS_MD}"
        actions.append(action)
        if not dry_run:
            generate_copilot_instructions(agents_md, output_path)

    return actions


def _sync_copilot_to_codex(repo_root: Path, dry_run: bool, prefix: str) -> list[str]:
    """Convert .github/ → .codex/."""
    actions: list[str] = []

    # --- Agents ---
    src_agents = repo_root / _COPILOT_AGENTS_DIR
    dst_agents = repo_root / _CODEX_AGENTS_DIR
    if src_agents.is_dir():
        for md_file in sorted(src_agents.glob("*.agent.md")):
            toml_content = copilot_agent_md_to_codex_toml(md_file)
            stem = md_file.name.removesuffix(".agent.md")
            dst_file = dst_agents / f"{stem}.toml"
            action = f"{prefix}convert {md_file.relative_to(repo_root)} → {dst_file.relative_to(repo_root)}"
            actions.append(action)
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(toml_content, encoding="utf-8")

    # --- Skills ---
    src_skills = repo_root / _COPILOT_SKILLS_DIR
    dst_skills = repo_root / _CODEX_SKILLS_DIR
    if src_skills.is_dir():
        if dry_run:
            for skill in sorted(src_skills.rglob("SKILL.md")):
                rel = skill.relative_to(src_skills)
                actions.append(f"{prefix}copy {_COPILOT_SKILLS_DIR}/{rel} → {_CODEX_SKILLS_DIR}/{rel}")
        else:
            copied = sync_skills(src_skills, dst_skills)
            for p in copied:
                actions.append(f"{prefix}copy → {p.relative_to(repo_root)}")

    # --- AGENTS.md from copilot-instructions.md ---
    instructions_path = repo_root / _COPILOT_INSTRUCTIONS
    agents_md = repo_root / _CODEX_AGENTS_MD
    if instructions_path.is_file():
        action = f"{prefix}update {_CODEX_AGENTS_MD} from {_COPILOT_INSTRUCTIONS}"
        actions.append(action)
        if not dry_run:
            raw = instructions_path.read_text(encoding="utf-8")
            # Strip the auto-generated header comments
            content = re.sub(r"^<!--.*?-->\s*\n?", "", raw, flags=re.MULTILINE).strip()
            agents_md.write_text(content + "\n", encoding="utf-8")

    return actions


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff_configs(repo_root: Path) -> str:
    """Show differences between ``.codex/`` and ``.github/`` agent configs.

    For each agent, converts the Codex TOML to Copilot MD (or vice-versa)
    and diffs against the existing counterpart file.

    Returns
    -------
    str
        Unified diff output, or a summary message if everything matches.
    """
    lines: list[str] = []

    # Collect agent names from both sides
    codex_dir = repo_root / _CODEX_AGENTS_DIR
    copilot_dir = repo_root / _COPILOT_AGENTS_DIR

    codex_agents: dict[str, Path] = {}
    copilot_agents: dict[str, Path] = {}

    if codex_dir.is_dir():
        for f in codex_dir.glob("*.toml"):
            codex_agents[f.stem] = f
    if copilot_dir.is_dir():
        for f in copilot_dir.glob("*.agent.md"):
            copilot_agents[f.name.removesuffix(".agent.md")] = f

    all_names = sorted(set(codex_agents) | set(copilot_agents))

    for name in all_names:
        codex_path = codex_agents.get(name)
        copilot_path = copilot_agents.get(name)

        if codex_path and not copilot_path:
            lines.append(f"  {name}: only in .codex/ (no .github/ counterpart)")
        elif copilot_path and not codex_path:
            lines.append(f"  {name}: only in .github/ (no .codex/ counterpart)")
        elif codex_path and copilot_path:
            # Convert codex → copilot and compare
            generated = codex_agent_toml_to_copilot_md(codex_path)
            existing = copilot_path.read_text(encoding="utf-8")
            if generated == existing:
                lines.append(f"  {name}: in sync ✓")
            else:
                lines.append(f"  {name}: DIFFERS")
                diff = difflib.unified_diff(
                    existing.splitlines(keepends=True),
                    generated.splitlines(keepends=True),
                    fromfile=str(copilot_path.relative_to(repo_root)),
                    tofile=f"(generated from {codex_path.relative_to(repo_root)})",
                )
                lines.extend("    " + l.rstrip("\n") for l in diff)

    # Skills
    codex_skills = repo_root / _CODEX_SKILLS_DIR
    copilot_skills = repo_root / _COPILOT_SKILLS_DIR
    codex_skill_files = sorted(codex_skills.rglob("SKILL.md")) if codex_skills.is_dir() else []
    copilot_skill_files = sorted(copilot_skills.rglob("SKILL.md")) if copilot_skills.is_dir() else []

    codex_rels = {f.relative_to(codex_skills) for f in codex_skill_files}
    copilot_rels = {f.relative_to(copilot_skills) for f in copilot_skill_files}

    for rel in sorted(codex_rels | copilot_rels):
        cs = codex_skills / rel
        cp = copilot_skills / rel
        if rel in codex_rels and rel not in copilot_rels:
            lines.append(f"  skill {rel}: only in .codex/skills/")
        elif rel in copilot_rels and rel not in codex_rels:
            lines.append(f"  skill {rel}: only in .github/skills/")
        else:
            if cs.read_text(encoding="utf-8") == cp.read_text(encoding="utf-8"):
                lines.append(f"  skill {rel}: in sync ✓")
            else:
                lines.append(f"  skill {rel}: DIFFERS")

    # AGENTS.md vs copilot-instructions.md
    agents_md = repo_root / _CODEX_AGENTS_MD
    instructions_md = repo_root / _COPILOT_INSTRUCTIONS
    if agents_md.is_file() and instructions_md.is_file():
        raw = instructions_md.read_text(encoding="utf-8")
        content = re.sub(r"^<!--.*?-->\s*\n?", "", raw, flags=re.MULTILINE).strip()
        agents_content = agents_md.read_text(encoding="utf-8").strip()
        if content == agents_content:
            lines.append(f"  {_CODEX_AGENTS_MD} ↔ {_COPILOT_INSTRUCTIONS}: in sync ✓")
        else:
            lines.append(f"  {_CODEX_AGENTS_MD} ↔ {_COPILOT_INSTRUCTIONS}: DIFFERS")
    elif agents_md.is_file():
        lines.append(f"  {_COPILOT_INSTRUCTIONS}: missing")
    elif instructions_md.is_file():
        lines.append(f"  {_CODEX_AGENTS_MD}: missing")

    if not lines:
        return "No configuration files found in either .codex/ or .github/."
    return "Config diff (.codex/ ↔ .github/):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config_sync",
        description="Synchronize config between Codex CLI (.codex/) and Copilot CLI (.github/).",
    )
    parser.add_argument(
        "--from",
        dest="source",
        choices=["codex", "copilot"],
        help="Source config format.",
    )
    parser.add_argument(
        "--to",
        dest="target",
        choices=["codex", "copilot"],
        help="Target config format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without writing files.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Show differences between .codex/ and .github/ configs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.diff:
        print(diff_configs(args.repo_root))
        return 0

    if not args.source or not args.target:
        parser.error("--from and --to are required unless --diff is used")

    if args.source == args.target:
        parser.error("--from and --to must be different")

    actions = sync_configs(args.repo_root, args.source, args.target, dry_run=args.dry_run)

    if not actions:
        print("Nothing to sync.")
    else:
        for a in actions:
            print(a)
        print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(actions)} action(s) completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
