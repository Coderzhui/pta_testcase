from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test" / "api_test"
RUNS_DIR = ROOT / "runs"
DEFAULT_MANIFEST_FIELDS = ["raw_api_name", "canonical_name", "file_name", "status", "notes"]
FIX_MODES = {"off", "tests", "safe"}
FAILURE_CATEGORIES = {
    "NONE",
    "TEST_BUG",
    "UNSUPPORTED_ON_NPU",
    "ENVIRONMENT_MISSING",
    "API_BEHAVIOR_MISMATCH",
    "PYTORCH_BUG",
    "TORCH_NPU_BUG",
    "FLAKY_OR_UNSTABLE",
    "INSUFFICIENT_COVERAGE",
    "UNKNOWN",
}


@dataclass
class ManifestEntry:
    raw_api_name: str
    canonical_name: str
    file_name: str
    status: str = "pending"
    notes: str = ""

    @property
    def test_path(self) -> Path:
        return TEST_DIR / self.file_name


@dataclass
class ApiResult:
    raw_api_name: str
    canonical_name: str
    file_name: str
    stage: str = "manifest"
    final_status: str = "pending"
    pytest_outcome: str = "not_run"
    failure_category: str = "UNKNOWN"
    root_cause_summary: str = ""
    failure_messages: list[str] = field(default_factory=list)
    tests_total: int = 0
    passed_count: int = 0
    skipped_count: int = 0
    xfailed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    fix_recommendation: str = "none"
    auto_fixable: bool = False
    fix_applied: bool = False
    fix_target: str = ""
    fix_summary: str = ""
    fix_artifact: str = ""
    changed_files: list[str] = field(default_factory=list)
    rerun_status: str = "not_run"
    report_path: str = ""


def api_to_filename(api_name: str) -> str:
    name = api_name.strip()
    if not name:
        return ""
    if name.startswith("torch."):
        name = name[len("torch.") :]
    return f"test_{name.replace('.', '_')}.py"


def build_manifest_from_text_input(input_path: Path, output_path: Path) -> list[ManifestEntry]:
    rows: list[ManifestEntry] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        api_name = line.strip()
        if not api_name or api_name.startswith("#"):
            continue
        rows.append(
            ManifestEntry(
                raw_api_name=api_name,
                canonical_name=api_name,
                file_name=api_to_filename(api_name),
                status="pending",
                notes="",
            )
        )
    write_manifest(rows, output_path)
    return rows


def load_manifest(path: Path) -> list[ManifestEntry]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in DEFAULT_MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
        return [
            ManifestEntry(
                raw_api_name=row["raw_api_name"].strip(),
                canonical_name=row["canonical_name"].strip(),
                file_name=row["file_name"].strip(),
                status=(row.get("status") or "pending").strip() or "pending",
                notes=(row.get("notes") or "").strip(),
            )
            for row in reader
            if (row.get("canonical_name") or "").strip()
        ]


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "raw_api_name": entry.raw_api_name,
                    "canonical_name": entry.canonical_name,
                    "file_name": entry.file_name,
                    "status": entry.status,
                    "notes": entry.notes,
                }
            )


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(report_dir: Path | None, resume_dir: Path | None) -> Path:
    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    base_dir = (report_dir or RUNS_DIR).resolve()
    run_dir = base_dir / utc_run_id()
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{utc_run_id()}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_command(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    stdin_text: str | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_input_manifest(input_path: Path, run_dir: Path) -> tuple[list[ManifestEntry], Path]:
    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        manifest_path = run_dir / "manifest.csv"
        entries = build_manifest_from_text_input(input_path.resolve(), manifest_path)
        return entries, manifest_path
    if suffix == ".csv":
        manifest_path = run_dir / "manifest.csv"
        entries = load_manifest(input_path.resolve())
        write_manifest(entries, manifest_path)
        return entries, manifest_path
    raise ValueError(f"unsupported input type for {input_path}; expected .txt or .csv")


def codex_prompt_for_generation(manifest_path: Path, run_dir: Path, max_workers: int) -> str:
    return textwrap.dedent(
        f"""\
        使用 batch-npu-api-test skill。

        处理 CSV 文件：{relative_to_root(manifest_path)}

        执行生成阶段，不要把任务拆成需要我再次确认的多轮对话。
        要求：
        1. 只读取 CSV 中 status=pending 的 API。
        2. 启动 generator/reviewer 并行生成和审查测试文件。
        3. 可以对测试文件做最小修复，但只允许修改 test/api_test/ 下 CSV 对应的目标文件。
        4. 不要运行 pytest；外层 pipeline 会统一执行和分析。
        5. 不要修改其他目录。
        6. 最终回复写入简洁的生成摘要，包含触达的文件和静态阻塞项。
        7. 本次批处理的并发预算参考值：{max_workers}。

        生成摘要请写到最终消息。外层 pipeline 会保存到：
        {relative_to_root(run_dir / "generation_summary.md")}
        """
    )


def run_generation_stage(manifest_path: Path, run_dir: Path, max_workers: int) -> None:
    prompt = codex_prompt_for_generation(manifest_path, run_dir, max_workers)
    command = [
        "codex",
        "exec",
        "--cd",
        ".",
        "--dangerously-bypass-approvals-and-sandbox",
        "--output-last-message",
        str(run_dir / "generation_summary.md"),
        "-",
    ]
    completed = run_command(
        command,
        cwd=ROOT,
        stdin_text=prompt,
        stdout_path=run_dir / "codex_generation.stdout.log",
        stderr_path=run_dir / "codex_generation.stderr.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "generation stage failed; inspect "
            f"{relative_to_root(run_dir / 'codex_generation.stderr.log')}"
        )


def run_pytest_stage(test_files: list[Path], run_dir: Path, label: str) -> dict[str, object]:
    junit_path = run_dir / "pytest_raw" / f"{label}_junit.xml"
    stdout_path = run_dir / "pytest_raw" / f"{label}.stdout.log"
    stderr_path = run_dir / "pytest_raw" / f"{label}.stderr.log"
    junit_path.parent.mkdir(parents=True, exist_ok=True)

    if not test_files:
        junit_path.write_text("<testsuite tests=\"0\" failures=\"0\" errors=\"0\" skipped=\"0\" />\n", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        command_text = "(pytest skipped: no target files existed)"
        (run_dir / "pytest_raw" / f"{label}.command.txt").write_text(command_text, encoding="utf-8")
        return {
            "returncode": 0,
            "junit_path": junit_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "command": command_text,
        }

    cmd = [sys.executable, "-m", "pytest", "-q", "--junitxml", str(junit_path), *[str(path) for path in test_files]]
    completed = run_command(cmd, cwd=ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
    command_text = " ".join(shlex.quote(part) for part in cmd)
    (run_dir / "pytest_raw" / f"{label}.command.txt").write_text(command_text, encoding="utf-8")
    return {
        "returncode": completed.returncode,
        "junit_path": junit_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "command": command_text,
    }


def parse_testcase_outcome(testcase: ET.Element) -> tuple[str, str]:
    failure = testcase.find("failure")
    if failure is not None:
        return "failed", build_message_blob(failure)
    error = testcase.find("error")
    if error is not None:
        return "error", build_message_blob(error)
    skipped = testcase.find("skipped")
    if skipped is not None:
        message = build_message_blob(skipped)
        kind = (skipped.attrib.get("type") or "").lower()
        if "xfail" in kind or "xfail" in message.lower():
            return "xfailed", message
        return "skipped", message
    return "passed", ""


def build_message_blob(node: ET.Element) -> str:
    parts = [node.attrib.get("message", "").strip(), (node.text or "").strip()]
    return "\n".join(part for part in parts if part).strip()


def resolve_entry_for_testcase(testcase: ET.Element, by_stem: dict[str, ManifestEntry]) -> ManifestEntry | None:
    file_attr = testcase.attrib.get("file", "")
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    candidates = [file_attr, classname, name]
    for candidate in candidates:
        normalized = candidate.replace("\\", "/")
        for stem, entry in by_stem.items():
            if stem in normalized or entry.file_name in normalized:
                return entry
    return None


def parse_junit_results(
    entries: list[ManifestEntry],
    execution: dict[str, object],
) -> dict[str, dict[str, object]]:
    per_api: dict[str, dict[str, object]] = {
        entry.canonical_name: {
            "tests_total": 0,
            "passed_count": 0,
            "skipped_count": 0,
            "xfailed_count": 0,
            "failed_count": 0,
            "error_count": 0,
            "messages": [],
        }
        for entry in entries
    }

    by_stem = {Path(entry.file_name).stem: entry for entry in entries}
    junit_path = execution["junit_path"]
    if not Path(junit_path).exists():
        return per_api

    tree = ET.parse(junit_path)
    root = tree.getroot()
    for testcase in root.iter("testcase"):
        entry = resolve_entry_for_testcase(testcase, by_stem)
        if entry is None:
            continue
        bucket = per_api[entry.canonical_name]
        outcome, message = parse_testcase_outcome(testcase)
        bucket["tests_total"] += 1
        if outcome == "passed":
            bucket["passed_count"] += 1
        elif outcome == "skipped":
            bucket["skipped_count"] += 1
        elif outcome == "xfailed":
            bucket["xfailed_count"] += 1
        elif outcome == "failed":
            bucket["failed_count"] += 1
        elif outcome == "error":
            bucket["error_count"] += 1
        if message:
            bucket["messages"].append(message)
    return per_api


def first_non_empty(items: Iterable[str]) -> str:
    for item in items:
        if item:
            return item
    return ""


def derive_final_status(bucket: dict[str, object], entry: ManifestEntry) -> tuple[str, str]:
    if not entry.test_path.exists():
        return "review_failed", "not_collected"
    failed = int(bucket["failed_count"])
    errors = int(bucket["error_count"])
    passed = int(bucket["passed_count"])
    skipped = int(bucket["skipped_count"])
    xfailed = int(bucket["xfailed_count"])
    if failed or errors:
        return "pytest_failed", "failed"
    if passed:
        if skipped and not xfailed:
            return "pytest_passed", "passed_with_skips"
        if xfailed:
            return "pytest_passed", "passed_with_xfails"
        return "pytest_passed", "passed"
    if xfailed:
        return "xfailed", "xfailed"
    if skipped:
        return "skipped", "skipped"
    return "analyzed", "no_tests_recorded"


def detect_category(text: str, final_status: str) -> str:
    lowered = text.lower()
    if final_status in {"pytest_passed", "fixed"}:
        return "NONE"
    if any(token in lowered for token in ["torch_npu import failed", "no module named 'torch_npu'", "npu is not available"]):
        return "ENVIRONMENT_MISSING"
    if any(token in lowered for token in ["not supported by this npu backend", "unsupported on npu", "does not support", "not exposed in this build", "dispatchkey.npu"]):
        return "UNSUPPORTED_ON_NPU"
    if any(token in lowered for token in ["xfail", "not reliable", "unstable", "flaky"]):
        return "FLAKY_OR_UNSTABLE"
    if any(token in lowered for token in ["/ascend-pytorch/", "torch_npu/"]):
        return "TORCH_NPU_BUG"
    if any(token in lowered for token in ["/pytorch/", " aten/", "torch/csrc", "c10/"]):
        return "PYTORCH_BUG"
    if any(token in lowered for token in ["did not raise", "assertionerror", "typeerror", "attributeerror", "nameerror", "runtimeerror"]) and "test/api_test" in lowered:
        return "TEST_BUG"
    if "coverage" in lowered and any(token in lowered for token in ["missing", "insufficient", "uncovered"]):
        return "INSUFFICIENT_COVERAGE"
    if final_status in {"skipped", "xfailed"}:
        return "UNSUPPORTED_ON_NPU"
    if final_status == "pytest_failed":
        return "API_BEHAVIOR_MISMATCH"
    return "UNKNOWN"


def recommend_fix(category: str, fix_mode: str) -> tuple[str, bool, str]:
    if fix_mode == "off":
        return "manual_followup", False, ""
    if category in {"TEST_BUG", "ENVIRONMENT_MISSING", "UNSUPPORTED_ON_NPU", "FLAKY_OR_UNSTABLE", "INSUFFICIENT_COVERAGE"}:
        return "adjust_test", True, "test/api_test"
    if fix_mode == "safe" and category == "PYTORCH_BUG":
        return "patch_pytorch", True, "pytorch"
    if fix_mode == "safe" and category == "TORCH_NPU_BUG":
        return "patch_torch_npu", True, "ascend-pytorch"
    if fix_mode == "safe" and category == "API_BEHAVIOR_MISMATCH":
        return "adjust_test_or_patch_source", True, "test/api_test,pytorch,ascend-pytorch"
    return "manual_followup", False, ""


def create_results(entries: list[ManifestEntry], execution: dict[str, object], run_dir: Path, fix_mode: str) -> list[ApiResult]:
    junit_results = parse_junit_results(entries, execution)
    results: list[ApiResult] = []
    for entry in entries:
        bucket = junit_results[entry.canonical_name]
        final_status, pytest_outcome = derive_final_status(bucket, entry)
        message = first_non_empty(bucket["messages"]) or entry.notes
        category = detect_category(message, final_status)
        recommendation, auto_fixable, fix_target = recommend_fix(category, fix_mode)
        result = ApiResult(
            raw_api_name=entry.raw_api_name,
            canonical_name=entry.canonical_name,
            file_name=entry.file_name,
            stage="analysis",
            final_status=final_status,
            pytest_outcome=pytest_outcome,
            failure_category=category,
            root_cause_summary=message or "No explicit failure detail was captured.",
            failure_messages=list(bucket["messages"]),
            tests_total=int(bucket["tests_total"]),
            passed_count=int(bucket["passed_count"]),
            skipped_count=int(bucket["skipped_count"]),
            xfailed_count=int(bucket["xfailed_count"]),
            failed_count=int(bucket["failed_count"]),
            error_count=int(bucket["error_count"]),
            fix_recommendation=recommendation,
            auto_fixable=auto_fixable,
            fix_target=fix_target,
            report_path=relative_to_root(run_dir / "summary.md"),
        )
        if result.failure_category not in FAILURE_CATEGORIES:
            result.failure_category = "UNKNOWN"
        results.append(result)
    return results


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results(results: list[ApiResult], run_dir: Path) -> None:
    json_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    write_json(json_path, [asdict(result) for result in results])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else list(asdict(ApiResult("", "", "")).keys()))
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["failure_messages"] = json.dumps(result.failure_messages, ensure_ascii=False)
            row["changed_files"] = json.dumps(result.changed_files, ensure_ascii=False)
            writer.writerow(row)


def snapshot_newer_files(paths: list[Path], marker: Path) -> list[str]:
    changed: list[str] = []
    for base in paths:
        if not base.exists():
            continue
        if base.is_file():
            candidates = [base]
        else:
            candidates = [path for path in base.rglob("*") if path.is_file()]
        for candidate in candidates:
            if candidate.stat().st_mtime_ns > marker.stat().st_mtime_ns:
                changed.append(relative_to_root(candidate))
    return sorted(set(changed))


def fix_prompt(result: ApiResult, run_dir: Path, fix_mode: str) -> str:
    allowed_scopes = [f"test/api_test/{result.file_name}"]
    if fix_mode == "safe":
        allowed_scopes.extend(["pytorch/", "ascend-pytorch/"])
    category = result.failure_category
    summary = result.root_cause_summary.strip()
    failure_blob = "\n\n".join(result.failure_messages[:3]).strip() or "(no extra failure messages captured)"
    return textwrap.dedent(
        f"""\
        修复一次单 API 失败，不要等待额外确认。

        API: {result.canonical_name}
        测试文件: test/api_test/{result.file_name}
        分类: {category}
        建议动作: {result.fix_recommendation}
        当前 pytest 状态: {result.final_status} / {result.pytest_outcome}

        允许修改范围：
        {os.linesep.join(f"- {scope}" for scope in allowed_scopes)}

        失败摘要：
        {summary}

        失败细节：
        {failure_blob}

        修复要求：
        1. 只做最小修复，禁止触达未授权文件。
        2. 如果问题本质是当前环境/后端不支持，请优先在测试中使用 pytest.skip 或 pytest.xfail，并写清楚原因。
        3. 只有在 fix mode 为 safe 且确实是低风险局部问题时，才允许修改 pytorch/ 或 ascend-pytorch/。
        4. 不要做重构，不要扩散改动。
        5. 外层 pipeline 会自动 rerun pytest，因此你只需要完成修复并在最终回复里说明改了什么。
        """
    )


def run_fix_attempt(result: ApiResult, run_dir: Path, fix_mode: str) -> ApiResult:
    marker = run_dir / "fixes" / f"{Path(result.file_name).stem}.before"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    prompt = fix_prompt(result, run_dir, fix_mode)
    summary_path = run_dir / "fixes" / f"{Path(result.file_name).stem}.md"
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
    completed = run_command(
        command,
        cwd=ROOT,
        stdin_text=prompt,
        stdout_path=run_dir / "fixes" / f"{Path(result.file_name).stem}.stdout.log",
        stderr_path=run_dir / "fixes" / f"{Path(result.file_name).stem}.stderr.log",
    )
    allowed_paths = [TEST_DIR / result.file_name]
    if fix_mode == "safe":
        allowed_paths.extend([ROOT / "pytorch", ROOT / "ascend-pytorch"])
    changed_files = snapshot_newer_files(allowed_paths, marker)
    result.fix_artifact = relative_to_root(summary_path)
    result.fix_summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
    result.fix_applied = completed.returncode == 0 and bool(changed_files)
    result.changed_files = changed_files
    if any(path.startswith("pytorch/") for path in changed_files):
        result.fix_target = "pytorch"
    elif any(path.startswith("ascend-pytorch/") for path in changed_files):
        result.fix_target = "ascend-pytorch"
    elif changed_files:
        result.fix_target = "test/api_test"
    if completed.returncode != 0 and not result.fix_summary:
        result.fix_summary = "codex fix attempt exited non-zero; inspect fix logs."
    if result.fix_applied:
        rerun = run_pytest_stage([TEST_DIR / result.file_name], run_dir, f"rerun_{Path(result.file_name).stem}")
        rerun_results = create_results(
            [ManifestEntry(result.raw_api_name, result.canonical_name, result.file_name)],
            rerun,
            run_dir,
            fix_mode="off",
        )
        rerun_result = rerun_results[0]
        result.rerun_status = rerun_result.final_status
    else:
        result.rerun_status = "not_run"
    return result


def apply_auto_fixes(results: list[ApiResult], run_dir: Path, fix_mode: str) -> list[ApiResult]:
    if fix_mode == "off":
        return results
    updated: list[ApiResult] = []
    for result in results:
        if result.final_status not in {"pytest_failed", "skipped", "xfailed", "review_failed"} or not result.auto_fixable:
            updated.append(result)
            continue
        updated.append(run_fix_attempt(result, run_dir, fix_mode))
    return updated


def merge_final_batch_results(
    entries: list[ManifestEntry],
    prior_results: list[ApiResult],
    execution: dict[str, object],
    run_dir: Path,
) -> list[ApiResult]:
    fresh = {result.canonical_name: result for result in create_results(entries, execution, run_dir, fix_mode="off")}
    merged: list[ApiResult] = []
    for result in prior_results:
        current = fresh[result.canonical_name]
        result.stage = "final"
        result.final_status = "fixed" if result.fix_applied and current.final_status == "pytest_passed" else current.final_status
        result.pytest_outcome = current.pytest_outcome
        result.failure_category = current.failure_category
        result.root_cause_summary = current.root_cause_summary
        result.failure_messages = current.failure_messages
        result.tests_total = current.tests_total
        result.passed_count = current.passed_count
        result.skipped_count = current.skipped_count
        result.xfailed_count = current.xfailed_count
        result.failed_count = current.failed_count
        result.error_count = current.error_count
        merged.append(result)
    return merged


def render_summary(
    results: list[ApiResult],
    run_dir: Path,
    input_path: Path,
    fix_mode: str,
    manifest_path: Path,
    final_command: str,
) -> str:
    total = len(results)
    counts: dict[str, int] = {}
    categories: dict[str, int] = {}
    fixed = [result for result in results if result.final_status == "fixed"]
    failed = [result for result in results if result.final_status in {"pytest_failed", "review_failed"}]
    skipped = [result for result in results if result.final_status == "skipped"]
    xfailed = [result for result in results if result.final_status == "xfailed"]
    passed = [result for result in results if result.final_status in {"pytest_passed", "fixed"}]
    for result in results:
        counts[result.final_status] = counts.get(result.final_status, 0) + 1
        categories[result.failure_category] = categories.get(result.failure_category, 0) + 1

    lines = [
        f"# Pipeline Summary: {run_dir.name}",
        "",
        f"- Input: `{relative_to_root(input_path.resolve())}`",
        f"- Manifest snapshot: `{relative_to_root(manifest_path)}`",
        f"- Fix mode: `{fix_mode}`",
        f"- Command: `{final_command}`",
        f"- Total APIs: `{total}`",
        f"- Results JSON: `{relative_to_root(run_dir / 'results.json')}`",
        f"- Results CSV: `{relative_to_root(run_dir / 'results.csv')}`",
        "",
        "## Status Counts",
    ]
    for status in sorted(counts):
        lines.append(f"- `{status}`: {counts[status]}")
    lines.extend(["", "## Failure Categories"])
    for category in sorted(categories):
        lines.append(f"- `{category}`: {categories[category]}")

    lines.extend(["", "## Fixed APIs"])
    if fixed:
        for result in fixed:
            changed = ", ".join(result.changed_files) if result.changed_files else "no tracked file diff detected"
            lines.append(f"- `{result.canonical_name}` -> `{result.fix_target or 'unknown'}`; rerun `{result.rerun_status}`; changed: {changed}")
    else:
        lines.append("- None")

    lines.extend(["", "## Remaining Failures"])
    if failed:
        for result in failed:
            lines.append(f"- `{result.canonical_name}`: `{result.failure_category}`; {result.root_cause_summary or 'no summary'}")
    else:
        lines.append("- None")

    lines.extend(["", "## Skipped / Xfailed"])
    if skipped or xfailed:
        for result in skipped + xfailed:
            lines.append(f"- `{result.canonical_name}`: `{result.final_status}`; {result.root_cause_summary or 'no summary'}")
    else:
        lines.append("- None")

    lines.extend(["", "## Passing APIs"])
    if passed:
        lines.append(f"- Count: {len(passed)}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_summary(
    results: list[ApiResult],
    run_dir: Path,
    input_path: Path,
    fix_mode: str,
    manifest_path: Path,
    command: str,
) -> None:
    summary = render_summary(results, run_dir, input_path, fix_mode, manifest_path, command)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def select_target_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    pending = [entry for entry in entries if entry.status == "pending"]
    return pending or entries


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch NPU API batch pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Build api_manifest.csv from apis.txt")
    manifest_parser.add_argument("--input", required=True, type=Path, help="Path to apis.txt")
    manifest_parser.add_argument("--output", required=True, type=Path, help="Output CSV path")

    run_parser = subparsers.add_parser("run", help="Generate tests, run pytest, analyze results, and optionally fix simple issues")
    run_parser.add_argument("--input", required=True, type=Path, help="Path to apis.txt or api_manifest.csv")
    run_parser.add_argument("--report-dir", type=Path, default=RUNS_DIR, help="Base directory for run artifacts")
    run_parser.add_argument("--resume", type=Path, help="Reuse an existing run directory")
    run_parser.add_argument("--fix-mode", choices=sorted(FIX_MODES), default="tests", help="Automatic fix scope")
    run_parser.add_argument("--skip-generate", action="store_true", help="Skip generation and reuse existing tests")
    run_parser.add_argument("--max-workers", type=int, default=8, help="Generation stage worker budget hint for nested codex")
    return parser.parse_args(argv)


def do_build_manifest(args: argparse.Namespace) -> int:
    entries = build_manifest_from_text_input(args.input.resolve(), args.output.resolve())
    print(f"written: {args.output} ({len(entries)} rows)")
    return 0


def do_run(args: argparse.Namespace) -> int:
    run_dir = ensure_run_dir(args.report_dir, args.resume)
    entries, manifest_path = resolve_input_manifest(args.input, run_dir)
    target_entries = select_target_entries(entries)
    target_files = [entry.test_path for entry in target_entries]

    if not args.skip_generate:
        run_generation_stage(manifest_path, run_dir, args.max_workers)

    existing_entries = [entry for entry in target_entries if entry.test_path.exists()]
    missing_entries = [entry for entry in target_entries if not entry.test_path.exists()]
    execution = run_pytest_stage([entry.test_path for entry in existing_entries], run_dir, "initial")
    results = create_results(target_entries, execution, run_dir, args.fix_mode)
    missing_names = {entry.canonical_name for entry in missing_entries}
    for result in results:
        if result.canonical_name in missing_names:
            result.stage = "review"
            result.final_status = "review_failed"
            result.pytest_outcome = "not_generated"
            result.failure_category = "TEST_BUG"
            result.root_cause_summary = "Expected test file was not created during generation/review stage."
            result.fix_recommendation, result.auto_fixable, result.fix_target = recommend_fix(result.failure_category, args.fix_mode)

    results = apply_auto_fixes(results, run_dir, args.fix_mode)
    if any(result.fix_applied for result in results):
        rerun_files = [entry.test_path for entry in target_entries if entry.test_path.exists()]
        final_execution = run_pytest_stage(rerun_files, run_dir, "postfix_batch")
        results = merge_final_batch_results(target_entries, results, final_execution, run_dir)

    write_results(results, run_dir)
    command_parts = [sys.executable, "-m", "scripts.pipeline", "run", "--input", str(args.input), "--fix-mode", args.fix_mode]
    if args.skip_generate:
        command_parts.append("--skip-generate")
    if args.resume:
        command_parts.extend(["--resume", str(args.resume)])
    if args.report_dir:
        command_parts.extend(["--report-dir", str(args.report_dir)])
    if args.max_workers != 8:
        command_parts.extend(["--max-workers", str(args.max_workers)])
    command_text = " ".join(shlex.quote(part) for part in command_parts)
    write_summary(results, run_dir, args.input, args.fix_mode, manifest_path, command_text)
    print(relative_to_root(run_dir / "summary.md"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "build-manifest":
        return do_build_manifest(args)
    if args.command == "run":
        return do_run(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
