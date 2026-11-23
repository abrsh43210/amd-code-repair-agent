"""Reading a repair task, and writing the answer.

ASSUMPTION (the official Match 5 contract is not published yet). Section 2 of the
shared rules fixes the invocation shape: the harness execs
    python3 /app/app.py --input-<something> <path>
once per test item and reads /app/output/<stem>_output.json.
Only the *task-specific* fields are guesswork, so every guess lives in this one
file. When the real spec appears, edit FIELD_* and OUTPUT_KEYS below and nothing
else changes.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import paths

# Keys we accept for each piece of a task, in order of preference.
FIELD_CODE = ("code", "buggy_code", "source", "src", "file_content", "content", "program", "snippet")
FIELD_PATH = ("file", "file_path", "path", "filename", "source_file", "target_file")
FIELD_TESTS = ("tests", "test_code", "test", "unit_tests", "test_content", "spec")
FIELD_TEST_PATH = ("test_file", "test_path", "tests_file", "test_filename")
FIELD_ERROR = ("error", "error_message", "traceback", "stderr", "failure", "test_output", "log")
FIELD_DESC = ("description", "issue", "problem", "prompt", "instruction", "task", "bug_report", "summary")
FIELD_LANG = ("language", "lang", "ext")
FIELD_ENTRY = ("entrypoint", "entry_point", "function", "func", "symbol")
FIELD_ID = ("instance_id", "task_id", "id", "name", "problem_id", "case_id", "bug_id")

# Repo-level tasks (SWE-bench / Kaggle Gemma 4 Developer Agent shape): an issue text
# plus a whole repository, answered with a unified diff.
FIELD_PROBLEM = ("problem_statement", "issue", "issue_text", "bug_report", "problem", "description", "prompt")
FIELD_HINTS = ("hints_text", "hints", "comments")
FIELD_REPO_DIR = ("repo_path", "repo_dir", "repo_root", "workspace", "snapshot", "repo_snapshot",
                  "snapshot_path", "repository_path", "project_dir")
OUTPUT_KEYS_REPO_PATCH = tuple(
    k for k in os.environ.get("MC5_OUTPUT_KEYS_REPO_PATCH", "patch,prediction,model_patch").split(",") if k
)
NO_PATCH = os.environ.get("MC5_NO_PATCH", "")          # Kaggle's harness uses the literal "NO_PATCH"

# What we write. "patch" (unified diff) and "code" (whole fixed file) are both emitted
# because we do not yet know which one is graded; extra keys have never been penalised
# in this league, a missing one scores zero.
OUTPUT_KEYS_CODE = tuple(
    k for k in os.environ.get("MC5_OUTPUT_KEYS_CODE", "code,fixed_code,source").split(",") if k
)
OUTPUT_KEYS_PATCH = tuple(
    k for k in os.environ.get("MC5_OUTPUT_KEYS_PATCH", "patch,diff").split(",") if k
)

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".go": "go", ".rs": "rust",
}


class Task:
    def __init__(self, stem: str, code: str, tests: str = "", error: str = "",
                 description: str = "", language: str = "", entrypoint: str = "",
                 file_name: str = "", test_file_name: str = "",
                 repo_dir: "Path | None" = None, problem: str = "", hints: str = ""):
        self.repo_dir = repo_dir                 # set => repo-level task
        self.problem = problem or description
        self.hints = hints
        self.stem = stem
        self.code = code
        self.tests = tests
        self.error = error
        self.description = description
        self.entrypoint = entrypoint
        self.file_name = file_name or ("solution" + (".py" if (language or "python") == "python" else ""))
        self.test_file_name = test_file_name or "test_solution.py"
        self.language = language or LANG_BY_EXT.get(Path(self.file_name).suffix.lower(), "python")

    def __repr__(self) -> str:
        return (f"Task({self.stem!r}, lang={self.language}, code={len(self.code)}B, "
                f"tests={len(self.tests)}B, error={bool(self.error)})")


def _pick(d: dict, keys) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            return "\n".join(v)
    return ""


def _safe_stem(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._")[:120]
    return s or "task"


def _read_sibling(base: Path, name: str) -> str:
    """A task may name files instead of inlining them; they sit beside the task file."""
    if not name:
        return ""
    for cand in (base / name, base / Path(name).name, paths.INPUT_DIR / Path(name).name):
        try:
            if cand.is_file():
                return cand.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return ""


def _extract_snapshot(archive: Path, name: str) -> "Path | None":
    """Unpack a .tgz/.tar/.zip repo snapshot into the work dir and return the repo root."""
    import shutil
    dst = paths.WORK_DIR / "snapshots" / _safe_stem(name)
    try:
        if dst.is_dir() and any(dst.iterdir()):
            root = dst
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(archive), str(dst), filter="data") if archive.suffix != ".zip" \
                else shutil.unpack_archive(str(archive), str(dst))
            root = dst
        kids = [k for k in root.iterdir() if not k.name.startswith(".")]
        if len(kids) == 1 and kids[0].is_dir() and not (root / ".git").exists():
            root = kids[0]                           # archive had a single top-level folder
        return root
    except Exception as e:  # noqa: BLE001
        paths.log(f"snapshot {archive}: {type(e).__name__}: {e}")
        return None


def _repo_dir(d: dict, tid: str, base: Path) -> "Path | None":
    named = _pick(d, FIELD_REPO_DIR)
    cands = []
    if named:
        cands += [Path(named), base / named, paths.INPUT_DIR / named]
    for root in (base, paths.INPUT_DIR):
        cands += [root / tid, root / "snapshots" / tid, root / "repos" / tid,
                  root / "snapshots" / f"{tid}.tgz", root / f"{tid}.tgz", root / f"{tid}.tar.gz",
                  root / f"{tid}.zip"]
    for c in cands:
        try:
            if c.is_dir():
                return c
            if c.is_file() and c.name.endswith((".tgz", ".tar.gz", ".tar", ".zip")):
                return _extract_snapshot(c, tid)
        except OSError:
            continue
    env = os.environ.get("MC5_REPO_DIR")
    return Path(env) if env and Path(env).is_dir() else None


def _from_dict(d: dict, stem: str, base: Path) -> Task:
    tid0 = _pick(d, FIELD_ID) or stem
    if not _pick(d, FIELD_CODE) and (_pick(d, FIELD_REPO_DIR) or "problem_statement" in d or "base_commit" in d):
        repo = _repo_dir(d, tid0, base)
        if repo is not None:
            return Task(stem=_safe_stem(tid0), code="", repo_dir=repo,
                        problem=_pick(d, FIELD_PROBLEM), hints=_pick(d, FIELD_HINTS),
                        error=_pick(d, FIELD_ERROR), language="python")
    file_name = _pick(d, FIELD_PATH)
    test_file_name = _pick(d, FIELD_TEST_PATH)
    code = _pick(d, FIELD_CODE) or _read_sibling(base, file_name)
    tests = _pick(d, FIELD_TESTS) or _read_sibling(base, test_file_name)
    tid = _pick(d, FIELD_ID) or stem
    return Task(
        stem=_safe_stem(tid), code=code, tests=tests,
        error=_pick(d, FIELD_ERROR), description=_pick(d, FIELD_DESC),
        language=_pick(d, FIELD_LANG).lstrip("."), entrypoint=_pick(d, FIELD_ENTRY),
        file_name=file_name, test_file_name=test_file_name,
    )


def load(path: Path) -> list[Task]:
    """A task file: JSON object, JSON list, or a bare source file to repair."""
    text = path.read_text(encoding="utf-8", errors="replace")
    base = path.parent
    if path.suffix.lower() not in (".json", ".jsonl", ""):
        return [Task(stem=_safe_stem(path.stem), code=text, file_name=path.name)]
    try:
        data = json.loads(text)
    except ValueError:
        if "\n" in text and text.lstrip().startswith("{"):          # jsonl
            items = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            return [_from_dict(d, f"task_{i + 1:02d}", base) for i, d in enumerate(items)]
        return [Task(stem=_safe_stem(path.stem), code=text, file_name=path.name)]

    if isinstance(data, dict):
        for k in ("tasks", "items", "problems", "cases", "data", "bugs"):
            if isinstance(data.get(k), list):
                return [_from_dict(x, f"task_{i + 1:02d}", base)
                        for i, x in enumerate(data[k]) if isinstance(x, dict)]
        return [_from_dict(data, path.stem, base)]
    if isinstance(data, list):
        return [_from_dict(x, f"task_{i + 1:02d}", base)
                for i, x in enumerate(data) if isinstance(x, dict)]
    if isinstance(data, str):
        return [Task(stem=_safe_stem(path.stem), code=data, file_name=path.name)]
    return []


def write_repo_output(out_dir: Path, task: Task, result, to_stdout: bool = False) -> Path:
    """Repo-level answer: the patch under every plausible key, plus the changed files in
    full for a harness that would rather overwrite files than apply a diff."""
    rec: dict = {"instance_id": task.stem, "id": task.stem}
    for k in OUTPUT_KEYS_REPO_PATCH:
        rec[k] = result.patch or NO_PATCH
    rec["files"] = dict(result.changed)
    rec["language"] = "python"
    rec["tests_pass"] = bool(result.verified)
    rec["confidence"] = round(float(result.confidence), 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_safe_stem(task.stem)}_output.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    if to_stdout:
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    return out


def unified_diff(before: str, after: str, name: str) -> str:
    import difflib
    if before == after:
        return ""
    d = difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                             fromfile=f"a/{name}", tofile=f"b/{name}", n=3)
    out = "".join(d)
    if out and not out.endswith("\n"):
        out += "\n"
    return out


def write_output(out_dir: Path, task: Task, fixed: str, confidence: float,
                 verified: bool, to_stdout: bool = False) -> Path:
    """One file per task, written whole (atomic rename). On total failure we still
    write a valid file containing the original code: a parseable answer that may be
    wrong beats a missing file, which the harness reads as a crash."""
    rec: dict = {}
    for k in OUTPUT_KEYS_CODE:
        rec[k] = fixed
    patch = unified_diff(task.code, fixed, task.file_name)
    for k in OUTPUT_KEYS_PATCH:
        rec[k] = patch
    rec["language"] = task.language
    rec["tests_pass"] = bool(verified)
    rec["confidence"] = round(float(confidence), 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_safe_stem(task.stem)}_output.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    if to_stdout:
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    return out
