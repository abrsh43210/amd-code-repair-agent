#!/usr/bin/env python3
"""Rebuild SWE-bench-style practice tasks from public repo history, in the same shape
as the Kaggle Gemma 4 Developer Agent tasks.jsonl:

  instance_id, repo, base_commit, problem_statement, hints_text, patch, test_patch, created_at

Pipeline (the one the Kaggle data page describes):
  1. commits that change both core .py logic and tests
  2. small, non-mechanical changes only
  3. fail-to-pass: base + test_patch must FAIL
  4. pass-to-pass: base + patch + test_patch must PASS
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOS = {
    "requests": "psf/requests",
    "httpx": "encode/httpx",
    "rich": "Textualize/rich",
    "fastapi": "fastapi/fastapi",
}


def git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    return p.stdout


def is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (path.startswith(("tests/", "test/")) or "/tests/" in path
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def is_source(path: str) -> bool:
    if not path.endswith(".py") or is_test(path):
        return False
    return not path.startswith(("docs/", "docs_src/", "examples/", "scripts/", "tools/", "benchmarks/", ".github/"))


def candidates(repo: Path, limit: int, scan: int):
    out = []
    shas = git(repo, "log", "--first-parent", "--no-merges", f"-n{scan}", "--format=%H").split()
    for sha in shas:
        stat = git(repo, "show", "--numstat", "--format=", sha).strip().splitlines()
        src, tests, src_lines = [], [], 0
        for ln in stat:
            parts = ln.split("\t")
            if len(parts) != 3 or parts[0] == "-":
                continue
            add, rem, path = int(parts[0]), int(parts[1]), parts[2]
            if "=>" in path:
                src = []
                break
            if is_source(path):
                src.append(path)
                src_lines += add + rem
            elif is_test(path) and path.endswith(".py"):
                tests.append(path)
        if not src or not tests or len(src) > 3 or not (1 <= src_lines <= 80):
            continue
        out.append((sha, src, tests))
        if len(out) >= limit:
            break
    return out


def run_tests(tree: Path, tests: list[str], timeout: int = 240) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": f"{tree}:{tree / 'src'}", "PYTHONDONTWRITEBYTECODE": "1",
           "COLUMNS": "120", "TERM": "dumb"}
    env.pop("FORCE_COLOR", None)
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "-o", "addopts=", *tests],
                           cwd=str(tree), env=env, capture_output=True, text=True, errors="replace",
                           timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def build(short: str, repo: Path, work: Path, limit: int, scan: int, keep: int):
    tasks = []
    for sha, src, tests in candidates(repo, limit, scan):
        base = git(repo, "rev-parse", f"{sha}^").strip()
        patch = git(repo, "diff", base, sha, "--", *src)
        test_patch = git(repo, "diff", base, sha, "--", *tests)
        if not patch.strip() or not test_patch.strip():
            continue
        tree = work / f"{short}_{sha[:10]}"
        shutil.rmtree(tree, ignore_errors=True)
        git(repo, "worktree", "add", "--detach", "--force", str(tree), base)
        try:
            (tree / "t.diff").write_text(test_patch)
            (tree / "p.diff").write_text(patch)
            if subprocess.run(["git", "-C", str(tree), "apply", "t.diff"], capture_output=True).returncode:
                continue
            rc_before, _ = run_tests(tree, tests)
            if rc_before in (0, 124, 2, 3, 4, 5):       # must FAIL on tests, not on collection/usage
                # rc 2 = interrupted/collection error. Accept collection errors only if
                # they come from a missing symbol the fix adds (ImportError) - too loose, skip.
                if rc_before != 1:
                    continue
            if subprocess.run(["git", "-C", str(tree), "apply", "p.diff"], capture_output=True).returncode:
                continue
            rc_after, out_after = run_tests(tree, tests)
            if rc_after != 0:
                continue
            msg = git(repo, "show", "-s", "--format=%B", sha).strip()
            msg = re.sub(r"(?im)^co-authored-by:.*$", "", msg).strip()
            m = re.search(r"\(#(\d+)\)", msg.splitlines()[0])
            num = m.group(1) if m else sha[:8]
            tasks.append({
                "instance_id": f"{short}_{num}",
                "repo": REPOS[short],
                "base_commit": base,
                "problem_statement": msg,
                "hints_text": "",
                "patch": patch,
                "test_patch": test_patch,
                "created_at": git(repo, "show", "-s", "--format=%cI", sha).strip(),
                "test_files": tests,
                "source_files": src,
            })
            print(f"  + {short}_{num}  src={src}  ({len(patch.splitlines())} patch lines)", flush=True)
            if len(tasks) >= keep:
                break
        finally:
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(tree)],
                           capture_output=True)
    return tasks


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    keep = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    work = Path(tempfile.mkdtemp(prefix="mine-"))
    all_tasks = []
    for short in REPOS:
        print(f"== {short}", flush=True)
        all_tasks += build(short, root / short, work, limit=80, scan=1500, keep=keep)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for t in all_tasks:
            fh.write(json.dumps(t) + "\n")
    print(f"\n{len(all_tasks)} verified tasks -> {out}")


if __name__ == "__main__":
    main()
