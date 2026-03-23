from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test" / "api_test"
RUNS_DIR = ROOT / "runs"
DEFAULT_MANIFEST_FIELDS = ["raw_api_name", "canonical_name", "file_name", "status", "notes"]
RUN_MANIFEST_FIELDS = DEFAULT_MANIFEST_FIELDS + [
    "selected_for_run",
    "run_phase",
    "stage",
    "test_file_exists",
    "final_status",
    "pytest_outcome",
    "failure_category",
    "root_cause_summary",
    "tests_total",
    "passed_count",
    "skipped_count",
    "xfailed_count",
    "failed_count",
    "error_count",
    "fix_recommendation",
    "auto_fixable",
    "fix_applied",
    "fix_target",
    "rerun_status",
    "changed_files",
    "fix_artifact",
    "report_path",
    "last_updated_utc",
]
FIX_MODES = {"off", "tests", "safe"}
RUN_ENGINES = {"local", "codex"}
ANALYSIS_ENGINES = {"heuristic", "codex"}
FAILURE_CATEGORIES = {
    "NONE",
    "TEST_BUG",
    "UNSUPPORTED_ON_NPU",
    "ENVIRONMENT_MISSING",
    "API_BEHAVIOR_MISMATCH",
    "PYTORCH_BUG",
    "TORCH_NPU_BUG",
    "OPERATOR_BUG",
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
    initial_failure_category: str = "UNKNOWN"
    initial_root_cause_summary: str = ""
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


class PipelineLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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


def csv_bool(value: bool) -> str:
    return "yes" if value else "no"


def csv_json(value: object) -> str:
    if value in ("", None, [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False)


def derive_run_manifest_status(
    entry: ManifestEntry,
    *,
    selected: bool,
    run_phase: str,
    result: ApiResult | None,
) -> str:
    if result is not None:
        return result.final_status
    if not selected:
        return entry.status
    if run_phase == "queued":
        return entry.status
    if run_phase in {"generated", "reused_existing"}:
        return "generated" if entry.test_path.exists() else "generation_missing"
    return entry.status


def write_run_manifest(
    entries: list[ManifestEntry],
    path: Path,
    *,
    selected_entries: list[ManifestEntry],
    run_phase: str,
    results: list[ApiResult] | None = None,
) -> None:
    selected_names = {entry.canonical_name for entry in selected_entries}
    results_by_name = {result.canonical_name: result for result in (results or [])}
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            selected = entry.canonical_name in selected_names
            result = results_by_name.get(entry.canonical_name)
            writer.writerow(
                {
                    "raw_api_name": entry.raw_api_name,
                    "canonical_name": entry.canonical_name,
                    "file_name": entry.file_name,
                    "status": derive_run_manifest_status(entry, selected=selected, run_phase=run_phase, result=result),
                    "notes": entry.notes,
                    "selected_for_run": csv_bool(selected),
                    "run_phase": run_phase,
                    "stage": result.stage if result is not None else ("manifest" if selected else "deferred"),
                    "test_file_exists": csv_bool(entry.test_path.exists()),
                    "final_status": result.final_status if result is not None else "",
                    "pytest_outcome": result.pytest_outcome if result is not None else "",
                    "failure_category": result.failure_category if result is not None else "",
                    "root_cause_summary": result.root_cause_summary if result is not None else "",
                    "tests_total": result.tests_total if result is not None else "",
                    "passed_count": result.passed_count if result is not None else "",
                    "skipped_count": result.skipped_count if result is not None else "",
                    "xfailed_count": result.xfailed_count if result is not None else "",
                    "failed_count": result.failed_count if result is not None else "",
                    "error_count": result.error_count if result is not None else "",
                    "fix_recommendation": result.fix_recommendation if result is not None else "",
                    "auto_fixable": csv_bool(result.auto_fixable) if result is not None else "",
                    "fix_applied": csv_bool(result.fix_applied) if result is not None else "",
                    "fix_target": result.fix_target if result is not None else "",
                    "rerun_status": result.rerun_status if result is not None else "",
                    "changed_files": csv_json(result.changed_files) if result is not None else "",
                    "fix_artifact": result.fix_artifact if result is not None else "",
                    "report_path": result.report_path if result is not None else "",
                    "last_updated_utc": timestamp,
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


def run_codex_exec(
    prompt: str,
    *,
    summary_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path = ROOT,
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
    return run_command(
        command,
        cwd=cwd,
        stdin_text=prompt,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


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
        3. 可以对测试文件做最小修复，但只允许修改 test/api_test/ 下 CSV 对应的目标文件，且禁止使用 pytest.xfail。
        4. 不要运行 pytest；外层 pipeline 会统一执行和分析。
        5. 不要修改其他目录。
        6. 最终回复写入简洁的生成摘要，包含触达的文件和静态阻塞项。
        7. 本次批处理的并发预算参考值：{max_workers}。

        生成摘要请写到最终消息。外层 pipeline 会保存到：
        {relative_to_root(run_dir / "generation_summary.md")}
        """
    )


def run_generation_stage(
    manifest_path: Path,
    run_dir: Path,
    max_workers: int,
    logger: PipelineLogger | None = None,
) -> None:
    started = time.monotonic()
    if logger is not None:
        logger.log(
            "stage=generation start "
            f"manifest={relative_to_root(manifest_path)} max_workers={max_workers} "
            f"stdout={relative_to_root(run_dir / 'codex_generation.stdout.log')} "
            f"stderr={relative_to_root(run_dir / 'codex_generation.stderr.log')}"
        )
    prompt = codex_prompt_for_generation(manifest_path, run_dir, max_workers)
    completed = run_codex_exec(
        prompt,
        summary_path=run_dir / "generation_summary.md",
        stdout_path=run_dir / "codex_generation.stdout.log",
        stderr_path=run_dir / "codex_generation.stderr.log",
    )
    if logger is not None:
        logger.log(
            "stage=generation done "
            f"returncode={completed.returncode} elapsed_s={time.monotonic() - started:.1f} "
            f"summary={relative_to_root(run_dir / 'generation_summary.md')}"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "generation stage failed; inspect "
            f"{relative_to_root(run_dir / 'codex_generation.stderr.log')}"
        )


def build_pytest_command(test_files: list[Path], junit_path: Path) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "--junitxml", str(junit_path), *[str(path) for path in test_files]]


def codex_prompt_for_execution(
    label: str,
    pytest_cmd: str,
    command_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    returncode_path: Path,
) -> str:
    command_file = shlex.quote(str(command_path))
    stdout_file = shlex.quote(str(stdout_path))
    stderr_file = shlex.quote(str(stderr_path))
    returncode_file = shlex.quote(str(returncode_path))
    parent_dir = shlex.quote(str(stdout_path.parent))
    shell_script = textwrap.dedent(
        f"""\
        mkdir -p {parent_dir}
        cat <<'EOF' > {command_file}
        {pytest_cmd}
        EOF
        set +e
        {pytest_cmd} > {stdout_file} 2> {stderr_file}
        status=$?
        printf '%s\\n' "$status" > {returncode_file}
        exit 0
        """
    ).strip()
    return textwrap.dedent(
        f"""\
        执行 pytest 阶段，不要修改任何源码、测试文件或文档。

        阶段标签: {label}
        你必须运行下面这段 bash 脚本，完整保留 pytest 的 stdout/stderr 和 return code。

        ```bash
        {shell_script}
        ```

        要求：
        1. 只执行上面的脚本，不要额外改文件。
        2. 即使 pytest 失败，也不要把这次 codex 任务判成失败；保留日志即可。
        3. 最终回复只写简洁总结，包含 return code 和产物路径。
        """
    )


def run_pytest_stage(
    test_files: list[Path],
    run_dir: Path,
    label: str,
    engine: str,
    logger: PipelineLogger | None = None,
) -> dict[str, object]:
    junit_path = run_dir / "pytest_raw" / f"{label}_junit.xml"
    stdout_path = run_dir / "pytest_raw" / f"{label}.stdout.log"
    stderr_path = run_dir / "pytest_raw" / f"{label}.stderr.log"
    command_path = run_dir / "pytest_raw" / f"{label}.command.txt"
    returncode_path = run_dir / "pytest_raw" / f"{label}.returncode.txt"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if logger is not None:
        logger.log(
            "stage=pytest start "
            f"label={label} engine={engine} test_files={len(test_files)} "
            f"stdout={relative_to_root(stdout_path)} stderr={relative_to_root(stderr_path)} "
            f"junit={relative_to_root(junit_path)}"
        )

    if not test_files:
        junit_path.write_text("<testsuite tests=\"0\" failures=\"0\" errors=\"0\" skipped=\"0\" />\n", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        command_text = "(pytest skipped: no target files existed)"
        command_path.write_text(command_text, encoding="utf-8")
        returncode_path.write_text("0\n", encoding="utf-8")
        if logger is not None:
            logger.log(f"stage=pytest done label={label} returncode=0 elapsed_s={time.monotonic() - started:.1f} skipped_no_files=true")
        return {
            "returncode": 0,
            "junit_path": junit_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "command": command_text,
        }

    cmd = build_pytest_command(test_files, junit_path)
    command_text = " ".join(shlex.quote(part) for part in cmd)
    if engine == "local":
        completed = run_command(cmd, cwd=ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
        command_path.write_text(command_text, encoding="utf-8")
        returncode_path.write_text(f"{completed.returncode}\n", encoding="utf-8")
        if logger is not None:
            logger.log(
                f"stage=pytest done label={label} returncode={completed.returncode} "
                f"elapsed_s={time.monotonic() - started:.1f}"
            )
        return {
            "returncode": completed.returncode,
            "junit_path": junit_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "command": command_text,
        }

    prompt = codex_prompt_for_execution(
        label,
        command_text,
        command_path,
        stdout_path,
        stderr_path,
        returncode_path,
    )
    completed = run_codex_exec(
        prompt,
        summary_path=run_dir / "pytest_raw" / f"{label}.codex.md",
        stdout_path=run_dir / "pytest_raw" / f"{label}.codex.stdout.log",
        stderr_path=run_dir / "pytest_raw" / f"{label}.codex.stderr.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"execution stage '{label}' failed; inspect "
            f"{relative_to_root(run_dir / 'pytest_raw' / f'{label}.codex.stderr.log')}"
        )
    if not returncode_path.exists():
        raise RuntimeError(
            f"execution stage '{label}' did not write return code; inspect "
            f"{relative_to_root(run_dir / 'pytest_raw' / f'{label}.codex.md')}"
        )
    returncode = int(returncode_path.read_text(encoding="utf-8").strip() or "1")
    if logger is not None:
        logger.log(
            f"stage=pytest done label={label} returncode={returncode} "
            f"elapsed_s={time.monotonic() - started:.1f} "
            f"codex_summary={relative_to_root(run_dir / 'pytest_raw' / f'{label}.codex.md')}"
        )
    return {
        "returncode": returncode,
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
    if failed or errors or xfailed:
        if xfailed and not failed and not errors:
            return "pytest_failed", "xfailed_not_allowed"
        return "pytest_failed", "failed"
    if passed:
        if skipped:
            return "pytest_passed", "passed_with_skips"
        return "pytest_passed", "passed"
    if skipped:
        return "skipped", "skipped"
    return "analyzed", "no_tests_recorded"


def detect_category(text: str, final_status: str) -> str:
    lowered = text.lower()
    if final_status in {"pytest_passed", "fixed"}:
        return "NONE"
    if "pytest.xfail" in lowered or "xfail" in lowered:
        return "TEST_BUG"
    if any(token in lowered for token in ["torch_npu import failed", "no module named 'torch_npu'", "npu is not available"]):
        return "ENVIRONMENT_MISSING"
    if any(token in lowered for token in ["not supported by this npu backend", "unsupported on npu", "does not support", "not exposed in this build", "dispatchkey.npu"]):
        return "UNSUPPORTED_ON_NPU"
    if any(token in lowered for token in ["not reliable", "unstable", "flaky"]):
        return "FLAKY_OR_UNSTABLE"
    if any(token in lowered for token in ["aclnn", "aclop", "op api", "opapi", "op-plugin", "kernel", "operator"]):
        return "OPERATOR_BUG"
    if any(token in lowered for token in ["/ascend-pytorch/", "torch_npu/"]):
        return "TORCH_NPU_BUG"
    if any(token in lowered for token in ["/pytorch/", " aten/", "torch/csrc", "c10/"]):
        return "PYTORCH_BUG"
    if any(token in lowered for token in ["did not raise", "assertionerror", "typeerror", "attributeerror", "nameerror", "runtimeerror"]) and "test/api_test" in lowered:
        return "TEST_BUG"
    if "coverage" in lowered and any(token in lowered for token in ["missing", "insufficient", "uncovered"]):
        return "INSUFFICIENT_COVERAGE"
    if final_status == "skipped":
        return "UNSUPPORTED_ON_NPU"
    if final_status == "pytest_failed":
        return "API_BEHAVIOR_MISMATCH"
    return "UNKNOWN"


def recommend_fix(category: str, fix_mode: str) -> tuple[str, bool, str]:
    if fix_mode == "off":
        return "manual_followup", False, ""
    if category == "TEST_BUG":
        return "adjust_test", True, "test/api_test"
    if category in {"ENVIRONMENT_MISSING", "UNSUPPORTED_ON_NPU", "FLAKY_OR_UNSTABLE", "INSUFFICIENT_COVERAGE", "OPERATOR_BUG"}:
        return "manual_followup", False, ""
    if fix_mode == "safe" and category == "PYTORCH_BUG":
        return "patch_pytorch", True, "pytorch"
    if fix_mode == "safe" and category == "TORCH_NPU_BUG":
        return "patch_torch_npu", True, "ascend-pytorch"
    if fix_mode == "safe" and category == "API_BEHAVIOR_MISMATCH":
        return "manual_followup", False, ""
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
            initial_failure_category=category,
            initial_root_cause_summary=message or "No explicit failure detail was captured.",
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


def build_analysis_inputs(results: list[ApiResult], run_dir: Path, execution: dict[str, object]) -> Path:
    analysis_items = []
    for result in results:
        if result.final_status not in {"pytest_failed", "skipped", "review_failed"}:
            continue
        analysis_items.append(
            {
                "canonical_name": result.canonical_name,
                "file_name": result.file_name,
                "test_path": relative_to_root(TEST_DIR / result.file_name),
                "final_status": result.final_status,
                "pytest_outcome": result.pytest_outcome,
                "heuristic_failure_category": result.failure_category,
                "heuristic_summary": result.root_cause_summary,
                "failure_messages": result.failure_messages,
            }
        )

    payload = {
        "run_dir": relative_to_root(run_dir),
        "generation_summary": relative_to_root(run_dir / "generation_summary.md"),
        "execution_artifacts": {
            "junit_path": relative_to_root(Path(execution["junit_path"])),
            "stdout_path": relative_to_root(Path(execution["stdout_path"])),
            "stderr_path": relative_to_root(Path(execution["stderr_path"])),
            "command": execution["command"],
        },
        "failure_taxonomy": relative_to_root(ROOT / "docs" / "failure_taxonomy.md"),
        "items": analysis_items,
    }
    path = run_dir / "analysis_inputs.json"
    write_json(path, payload)
    return path


def codex_prompt_for_analysis(analysis_input_path: Path, triage_path: Path) -> str:
    categories = ", ".join(sorted(FAILURE_CATEGORIES - {"NONE"}))
    return textwrap.dedent(
        f"""\
        执行失败分诊阶段，不要修改任何源码、测试文件或文档。

        输入文件：
        - 分析输入：{relative_to_root(analysis_input_path)}
        - 分类规则：{relative_to_root(ROOT / 'docs' / 'failure_taxonomy.md')}

        任务：
        1. 读取 analysis_inputs.json 中的所有失败/skip/review_failed API。
        2. 必要时查看对应测试文件和 pytest 日志。
        3. 为每个 API 产出一条 JSON 记录，写入 {relative_to_root(triage_path)}。

        输出 JSON 必须是数组，每一项严格包含：
        - canonical_name
        - failure_category
        - root_cause_summary

        约束：
        1. failure_category 只能取这些值：{categories}
        2. 只有确定是 test/api_test 下用例代码问题时，才标记为 TEST_BUG。
        3. 如果看到 pytest.xfail 或 xfail 痕迹，把它视为测试策略违规，优先标记为 TEST_BUG。
        4. 环境问题、PyTorch 代码问题、torch_npu/ascend-pytorch 问题、底层算子问题要区分开。
        5. 不明确时宁可保守标成 UNKNOWN 或 API_BEHAVIOR_MISMATCH，不要编造证据。
        6. 最终回复只写简洁分析总结。
        """
    )


def load_analysis_triage(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, list):
        return {}
    triage: dict[str, dict[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        canonical_name = str(item.get("canonical_name", "")).strip()
        category = str(item.get("failure_category", "")).strip()
        summary = str(item.get("root_cause_summary", "")).strip()
        if not canonical_name or category not in FAILURE_CATEGORIES:
            continue
        triage[canonical_name] = {
            "failure_category": category,
            "root_cause_summary": summary,
        }
    return triage


def render_analysis_summary(results: list[ApiResult], run_dir: Path, fix_mode: str) -> str:
    lines = [
        f"# Analysis Summary: {run_dir.name}",
        "",
        f"- Fix mode: `{fix_mode}`",
        f"- Inputs: `{relative_to_root(run_dir / 'analysis_inputs.json')}`",
        f"- Triage JSON: `{relative_to_root(run_dir / 'analysis_triage.json')}`",
        f"- Codex notes: `{relative_to_root(run_dir / 'analysis_codex.md')}`",
        "",
        "## Auto-Fix Candidates",
    ]
    candidates = [result for result in results if result.auto_fixable and result.final_status in {"pytest_failed", "skipped", "review_failed"}]
    if candidates:
        for result in candidates:
            lines.append(
                f"- `{result.canonical_name}`: `{result.failure_category}` -> `{result.fix_recommendation}`; "
                f"{result.root_cause_summary or 'no summary'}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Report-Only Failures"])
    report_only = [result for result in results if not result.auto_fixable and result.final_status in {"pytest_failed", "skipped", "review_failed"}]
    if report_only:
        for result in report_only:
            lines.append(
                f"- `{result.canonical_name}`: `{result.failure_category}`; "
                f"{result.root_cause_summary or 'no summary'}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def run_analysis_stage(
    results: list[ApiResult],
    run_dir: Path,
    execution: dict[str, object],
    fix_mode: str,
    engine: str,
    logger: PipelineLogger | None = None,
) -> list[ApiResult]:
    started = time.monotonic()
    analysis_input_path = build_analysis_inputs(results, run_dir, execution)
    triage_path = run_dir / "analysis_triage.json"
    codex_notes_path = run_dir / "analysis_codex.md"
    failing_results = [result for result in results if result.final_status in {"pytest_failed", "skipped", "review_failed"}]
    if logger is not None:
        logger.log(
            "stage=analysis start "
            f"engine={engine} failing_apis={len(failing_results)} "
            f"inputs={relative_to_root(analysis_input_path)}"
        )
    heuristic_triage = [
        {
            "canonical_name": result.canonical_name,
            "failure_category": result.failure_category,
            "root_cause_summary": result.root_cause_summary,
        }
        for result in failing_results
    ]

    if not failing_results:
        write_json(triage_path, [])
        codex_notes_path.write_text("No failing APIs required analysis.\n", encoding="utf-8")
        (run_dir / "analysis_summary.md").write_text(render_analysis_summary(results, run_dir, fix_mode), encoding="utf-8")
        if logger is not None:
            logger.log(
                f"stage=analysis done engine={engine} failing_apis=0 elapsed_s={time.monotonic() - started:.1f} "
                f"summary={relative_to_root(run_dir / 'analysis_summary.md')}"
            )
        return results

    if engine == "codex":
        prompt = codex_prompt_for_analysis(analysis_input_path, triage_path)
        completed = run_codex_exec(
            prompt,
            summary_path=codex_notes_path,
            stdout_path=run_dir / "analysis_codex.stdout.log",
            stderr_path=run_dir / "analysis_codex.stderr.log",
        )
        if completed.returncode == 0:
            triage = load_analysis_triage(triage_path)
            if triage:
                for result in results:
                    item = triage.get(result.canonical_name)
                    if not item:
                        continue
                    result.failure_category = item["failure_category"]
                    result.root_cause_summary = item["root_cause_summary"] or result.root_cause_summary
                    result.initial_failure_category = result.failure_category
                    result.initial_root_cause_summary = result.root_cause_summary
                    result.fix_recommendation, result.auto_fixable, result.fix_target = recommend_fix(result.failure_category, fix_mode)
            else:
                write_json(triage_path, heuristic_triage)
                codex_notes_path.write_text(
                    "Codex analysis did not produce valid triage JSON; falling back to heuristic classification.\n",
                    encoding="utf-8",
                )
        else:
            write_json(triage_path, heuristic_triage)
            codex_notes_path.write_text(
                "Codex analysis failed; falling back to heuristic classification.\n",
                encoding="utf-8",
            )
    else:
        write_json(triage_path, heuristic_triage)
        codex_notes_path.write_text("Analysis engine=heuristic; no nested codex review was run.\n", encoding="utf-8")

    (run_dir / "analysis_summary.md").write_text(render_analysis_summary(results, run_dir, fix_mode), encoding="utf-8")
    if logger is not None:
        logger.log(
            "stage=analysis done "
            f"engine={engine} failing_apis={len(failing_results)} elapsed_s={time.monotonic() - started:.1f} "
            f"summary={relative_to_root(run_dir / 'analysis_summary.md')}"
        )
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


def build_fix_request(result: ApiResult, fix_mode: str) -> dict[str, object]:
    allowed_scopes = [f"test/api_test/{result.file_name}"]
    if fix_mode == "safe":
        allowed_scopes.extend(["pytorch/", "ascend-pytorch/"])
    return {
        "canonical_name": result.canonical_name,
        "file_name": result.file_name,
        "fix_mode": fix_mode,
        "failure_category": result.failure_category,
        "fix_recommendation": result.fix_recommendation,
        "final_status": result.final_status,
        "pytest_outcome": result.pytest_outcome,
        "allowed_scopes": allowed_scopes,
        "root_cause_summary": result.root_cause_summary.strip(),
        "failure_messages": result.failure_messages[:3],
    }


def codex_prompt_for_fix(request_path: Path) -> str:
    return textwrap.dedent(
        f"""\
        使用 single-api-fix skill。

        处理修复请求文件：{relative_to_root(request_path)}

        执行修复阶段，不要等待额外确认。
        要求：
        1. 只修复该请求对应的单个 API。
        2. 严格遵守请求文件中的 allowed_scopes。
        3. 禁止使用 pytest.xfail。
        4. 不要运行 pytest；外层 pipeline 会自动回归验证。
        5. 最终回复写简洁修复摘要。
        """
    )


def run_fix_attempt(
    result: ApiResult,
    run_dir: Path,
    fix_mode: str,
    run_engine: str,
    logger: PipelineLogger | None = None,
) -> ApiResult:
    started = time.monotonic()
    if logger is not None:
        logger.log(
            "stage=fix start "
            f"api={result.canonical_name} category={result.failure_category} "
            f"target=test/api_test/{result.file_name}"
        )
    marker = run_dir / "fixes" / f"{Path(result.file_name).stem}.before"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    request_path = run_dir / "fixes" / f"{Path(result.file_name).stem}.request.json"
    write_json(request_path, build_fix_request(result, fix_mode))
    prompt = codex_prompt_for_fix(request_path)
    summary_path = run_dir / "fixes" / f"{Path(result.file_name).stem}.md"
    completed = run_codex_exec(
        prompt,
        summary_path=summary_path,
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
        rerun = run_pytest_stage(
            [TEST_DIR / result.file_name],
            run_dir,
            f"rerun_{Path(result.file_name).stem}",
            run_engine,
            logger,
        )
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
    if logger is not None:
        logger.log(
            "stage=fix done "
            f"api={result.canonical_name} fix_applied={result.fix_applied} "
            f"rerun_status={result.rerun_status} elapsed_s={time.monotonic() - started:.1f} "
            f"artifact={result.fix_artifact or 'none'}"
        )
    return result


def apply_auto_fixes(
    results: list[ApiResult],
    run_dir: Path,
    fix_mode: str,
    run_engine: str,
    logger: PipelineLogger | None = None,
) -> list[ApiResult]:
    if fix_mode == "off":
        if logger is not None:
            logger.log("stage=fix skip reason=fix_mode_off")
        return results
    candidates = [
        result
        for result in results
        if result.final_status in {"pytest_failed", "skipped", "review_failed"} and result.auto_fixable
    ]
    if logger is not None:
        logger.log(f"stage=fix queue candidates={len(candidates)} fix_mode={fix_mode}")
    updated: list[ApiResult] = []
    for result in results:
        if result.final_status not in {"pytest_failed", "skipped", "review_failed"} or not result.auto_fixable:
            updated.append(result)
            continue
        updated.append(run_fix_attempt(result, run_dir, fix_mode, run_engine, logger))
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
    passed = [result for result in results if result.final_status in {"pytest_passed", "fixed"}]
    for result in results:
        counts[result.final_status] = counts.get(result.final_status, 0) + 1
        categories[result.failure_category] = categories.get(result.failure_category, 0) + 1

    lines = [
        f"# Pipeline Summary: {run_dir.name}",
        "",
        f"- Input: `{relative_to_root(input_path.resolve())}`",
        f"- Manifest progress CSV: `{relative_to_root(manifest_path)}`",
        f"- Fix mode: `{fix_mode}`",
        f"- Command: `{final_command}`",
        f"- Total APIs: `{total}`",
        f"- Results JSON: `{relative_to_root(run_dir / 'results.json')}`",
        f"- Results CSV: `{relative_to_root(run_dir / 'results.csv')}`",
        f"- Summary Table CSV: `{relative_to_root(run_dir / 'summary_table.csv')}`",
        f"- Generation Summary: `{relative_to_root(run_dir / 'generation_summary.md')}`",
        f"- Analysis Summary: `{relative_to_root(run_dir / 'analysis_summary.md')}`",
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
            lines.append(
                f"- `{result.canonical_name}`: initial `{result.initial_failure_category}` -> "
                f"`{result.fix_target or 'unknown'}`; rerun `{result.rerun_status}`; changed: {changed}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Remaining Failures"])
    if failed:
        for result in failed:
            lines.append(f"- `{result.canonical_name}`: `{result.failure_category}`; {result.root_cause_summary or 'no summary'}")
    else:
        lines.append("- None")

    lines.extend(["", "## Skipped APIs"])
    if skipped:
        for result in skipped:
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

    # Generate the comprehensive CSV summary table
    csv_path = run_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["API Name", "Final Status", "Category", "Auto-Fix?", "Rerun Status", "Quick Summary"])
        for result in results:
            short_summary = result.root_cause_summary.replace('\n', ' ')
            fixed_val = "Yes" if result.fix_applied else "No"
            writer.writerow([
                result.canonical_name,
                result.final_status,
                result.failure_category,
                fixed_val,
                result.rerun_status,
                short_summary
            ])


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
    run_parser.add_argument("--run-engine", choices=sorted(RUN_ENGINES), default="codex", help="How pytest is executed")
    run_parser.add_argument("--analysis-engine", choices=sorted(ANALYSIS_ENGINES), default="codex", help="How failure triage is performed")
    run_parser.add_argument("--skip-generate", action="store_true", help="Skip generation and reuse existing tests")
    run_parser.add_argument("--max-workers", type=int, default=8, help="Generation stage worker budget hint for nested codex")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug mode to retain all intermediate subagent logs and full codex traces")
    return parser.parse_args(argv)


def do_build_manifest(args: argparse.Namespace) -> int:
    entries = build_manifest_from_text_input(args.input.resolve(), args.output.resolve())
    print(f"written: {args.output} ({len(entries)} rows)")
    return 0


def do_run(args: argparse.Namespace) -> int:
    start_time = time.time()
    run_dir = ensure_run_dir(args.report_dir, args.resume)
    logger = PipelineLogger(run_dir / "pipeline.log")
    
    if args.debug:
        logger.log("debug mode enabled: full codex and subagent traces will be collected")

    logger.log(
        "pipeline start "
        f"input={relative_to_root(args.input.resolve())} fix_mode={args.fix_mode} "
        f"run_engine={args.run_engine} analysis_engine={args.analysis_engine} "
        f"run_dir={relative_to_root(run_dir)}"
    )
    entries, manifest_path = resolve_input_manifest(args.input, run_dir)
    target_entries = select_target_entries(entries)
    write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="queued")
    logger.log(
        "manifest ready "
        f"entries={len(entries)} target_entries={len(target_entries)} "
        f"manifest={relative_to_root(manifest_path)}"
    )

    if not args.skip_generate:
        run_generation_stage(manifest_path, run_dir, args.max_workers, logger)
        write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="generated")
    else:
        (run_dir / "generation_summary.md").write_text(
            "Generation stage was skipped because --skip-generate was set.\n",
            encoding="utf-8",
        )
        logger.log("stage=generation skip reason=skip_generate")
        write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="reused_existing")

    existing_entries = [entry for entry in target_entries if entry.test_path.exists()]
    missing_entries = [entry for entry in target_entries if not entry.test_path.exists()]
    logger.log(
        f"pytest targets ready existing_files={len(existing_entries)} missing_files={len(missing_entries)}"
    )
    execution = run_pytest_stage([entry.test_path for entry in existing_entries], run_dir, "initial", args.run_engine, logger)
    results = create_results(target_entries, execution, run_dir, args.fix_mode)
    missing_names = {entry.canonical_name for entry in missing_entries}
    for result in results:
        if result.canonical_name in missing_names:
            result.stage = "review"
            result.final_status = "review_failed"
            result.pytest_outcome = "not_generated"
            result.failure_category = "TEST_BUG"
            result.root_cause_summary = "Expected test file was not created during generation/review stage."
            result.initial_failure_category = result.failure_category
            result.initial_root_cause_summary = result.root_cause_summary
            result.fix_recommendation, result.auto_fixable, result.fix_target = recommend_fix(result.failure_category, args.fix_mode)
    write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="initial_pytest", results=results)

    results = run_analysis_stage(results, run_dir, execution, args.fix_mode, args.analysis_engine, logger)
    write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="analysis", results=results)
    results = apply_auto_fixes(results, run_dir, args.fix_mode, args.run_engine, logger)
    write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="fix", results=results)
    if any(result.fix_applied for result in results):
        rerun_files = [entry.test_path for entry in target_entries if entry.test_path.exists()]
        logger.log(f"stage=pytest rerun start files={len(rerun_files)}")
        final_execution = run_pytest_stage(rerun_files, run_dir, "postfix_batch", args.run_engine, logger)
        results = merge_final_batch_results(target_entries, results, final_execution, run_dir)
    else:
        logger.log("stage=pytest rerun skip reason=no_fix_applied")
    write_run_manifest(entries, manifest_path, selected_entries=target_entries, run_phase="final", results=results)

    write_results(results, run_dir)
    command_parts = [sys.executable, "-m", "scripts.pipeline", "run", "--input", str(args.input), "--fix-mode", args.fix_mode]
    if args.run_engine != "codex":
        command_parts.extend(["--run-engine", args.run_engine])
    if args.analysis_engine != "codex":
        command_parts.extend(["--analysis-engine", args.analysis_engine])
    if args.skip_generate:
        command_parts.append("--skip-generate")
    if args.resume:
        command_parts.extend(["--resume", str(args.resume)])
    if args.report_dir:
        command_parts.extend(["--report-dir", str(args.report_dir)])
    if args.max_workers != 8:
        command_parts.extend(["--max-workers", str(args.max_workers)])
    if args.debug:
        command_parts.append("--debug")
    command_text = " ".join(shlex.quote(part) for part in command_parts)
    write_summary(results, run_dir, args.input, args.fix_mode, manifest_path, command_text)
    
    if args.debug:
        import shutil
        debug_dir = run_dir / "debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        codex_sessions_dir = Path.home() / ".codex" / "sessions"
        if codex_sessions_dir.exists():
            count = 0
            for jsonl_file in codex_sessions_dir.rglob("*.jsonl"):
                if jsonl_file.is_file() and jsonl_file.stat().st_mtime >= start_time:
                    shutil.copy2(jsonl_file, debug_dir / jsonl_file.name)
                    count += 1
            logger.log(f"debug_logs_collected count={count} in {relative_to_root(debug_dir)}")

    logger.log(
        "pipeline done "
        f"results_json={relative_to_root(run_dir / 'results.json')} "
        f"results_csv={relative_to_root(run_dir / 'results.csv')} "
        f"summary={relative_to_root(run_dir / 'summary.md')}"
    )
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
