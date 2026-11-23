#!/usr/bin/env python3
"""Offline end-to-end test for the Match 5 practice entry. No GPU, no network.

Runs app.py exactly as the harness would - one process per task, a path in, a JSON
file out - over the synthetic cases in tests/cases/, then checks:

  * every task produced /app/output/<id>_output.json with the required keys,
  * the fixed code really passes the task's tests (re-verified here, independently),
  * stdout stayed clean (an earlier entry scored OUTPUT_MALFORMED on stdout noise),
  * odd invocations (bare positional, --input-dir, --code/--tests, unknown flags)
    all still produce output,
  * an unverified model patch never displaces a verified fix (MC5_FAKE_LLM).

Usage:
    python3 tests/run_local.py              # model-free search (the fast path)
    MC5_FAKE_LLM=1 python3 tests/run_local.py   # also exercise the model path
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "app.py"
CASES = Path(__file__).resolve().parent / "cases"
sys.path.insert(0, str(ROOT / "app"))

FAKE = os.environ.get("MC5_FAKE_LLM") == "1"
# MC5_REAL_LLM=1 runs against a real model instead of the fake one (CI does this with
# a tiny model on CPU). Either way the model path is exercised rather than skipped.
REAL = os.environ.get("MC5_REAL_LLM") == "1"
USE_MODEL = FAKE or REAL


def run_app(args: list[str], out_dir: Path, work: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "MC5_OUTPUT_DIR": str(out_dir),
        "MC5_WORK_DIR": str(work),
        "MC5_SOCKET": str(work / "model.sock"),
        "MC5_INPUT_DIR": str(CASES),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if FAKE:
        env["MC5_FAKE_ORACLE"] = str(Path(__file__).resolve().parent / "oracle.json")
    if not USE_MODEL:
        args = args + ["--no-llm"]
    return subprocess.run([sys.executable, str(APP)] + args, capture_output=True, text=True,
                          env=env, timeout=240)


def recheck(case: dict, fixed: str) -> bool:
    """Independent verification: does the returned code pass the case's tests?"""
    tmp = Path(tempfile.mkdtemp(prefix="mc5-recheck-"))
    try:
        (tmp / "solution.py").write_text(fixed, encoding="utf-8")
        if not case.get("tests"):
            # No tests came with the task: the bar is that the module runs at all.
            p = subprocess.run([sys.executable, "solution.py"], cwd=str(tmp), timeout=30,
                               capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(tmp)})
            return p.returncode == 0
        (tmp / "test_solution.py").write_text(case["tests"], encoding="utf-8")
        p = subprocess.run([sys.executable, "test_solution.py"], cwd=str(tmp), timeout=30,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(tmp)})
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mc5-test-"))
    out_dir, work = tmp / "output", tmp / "work"
    out_dir.mkdir(parents=True)
    work.mkdir(parents=True)

    failures: list[str] = []
    fixed_count = 0
    case_files = sorted(CASES.glob("*.json"))
    print(f"{len(case_files)} cases, model={'fake-llm' if FAKE else 'real' if REAL else 'off'}\n")

    for cf in case_files:
        case = json.loads(cf.read_text(encoding="utf-8"))
        tid = case["task_id"]
        proc = run_app(["--input", str(cf)], out_dir, work)
        if proc.stdout.strip():
            failures.append(f"{tid}: wrote to stdout: {proc.stdout[:200]!r}")
        out = out_dir / f"{tid}_output.json"
        if not out.is_file():
            failures.append(f"{tid}: no output file (rc={proc.returncode}) {proc.stderr[-300:]}")
            continue
        rec = json.loads(out.read_text(encoding="utf-8"))
        for key in ("code", "patch", "confidence"):
            if key not in rec:
                failures.append(f"{tid}: output missing {key!r}")
        passed = recheck(case, rec.get("code", ""))
        fixed_count += bool(passed)
        expected = case.get("heuristic_fixable", True) or FAKE
        if not passed and (case.get("heuristic_fixable", True) or FAKE):
            # A real small model may legitimately fail the model-only case; only the
            # fake oracle is expected to fix it.
            failures.append(f"{tid}: returned code still fails its tests (label in log)")
        if rec.get("tests_pass") and not passed:
            failures.append(f"{tid}: reported tests_pass but the tests fail here")
        if not expected and passed:
            print(f"  (note: {tid} was not expected to be repairable without a model)")
        flag = "fixed" if passed else ("not fixed (expected)" if not expected else "NOT fixed")
        print(f"  {tid:28s} {flag:10s} conf={rec.get('confidence')} patch={'yes' if rec.get('patch') else 'no'}")

    # Odd invocations: all of these must still write a file.
    sample = case_files[0]
    odd = [
        (["--input-task", str(sample)], "input-task alias"),
        ([str(sample)], "bare positional"),
        (["--input-dir", str(CASES)], "input-dir"),
        (["--input", str(sample), "--unknown-flag", "x"], "unknown flag tolerated"),
    ]
    print()
    for args, label in odd:
        o = tmp / ("odd-" + label.replace(" ", "-"))
        o.mkdir()
        proc = run_app(args + (["--output-dir", str(o)] if "--output-dir" not in args else []), o, work)
        files = list(o.glob("*_output.json"))
        ok = bool(files) and all(json.loads(f.read_text())["code"] for f in files)
        print(f"  {label:28s} {'ok' if ok else 'FAILED'} ({len(files)} file(s))")
        if not ok:
            failures.append(f"odd invocation {label!r}: no usable output ({proc.stderr[-200:]})")

    # Code files on disk instead of a JSON task.
    direct = tmp / "direct"
    direct.mkdir()
    (direct / "buggy.py").write_text(json.loads(case_files[0].read_text())["code"], encoding="utf-8")
    (direct / "test_buggy.py").write_text(json.loads(case_files[0].read_text())["tests"]
                                          .replace("from solution import", "from buggy import"),
                                          encoding="utf-8")
    proc = run_app(["--code", str(direct / "buggy.py"), "--tests", str(direct / "test_buggy.py"),
                    "--output-dir", str(direct)], direct, work)
    got = list(direct.glob("*_output.json"))
    print(f"  {'--code/--tests on disk':28s} {'ok' if got else 'FAILED'}")
    if not got:
        failures.append(f"--code/--tests: no output ({proc.stderr[-200:]})")

    print(f"\n{fixed_count}/{len(case_files)} cases repaired and independently re-verified")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  - " + f)
    log = work / "app.log"
    if log.is_file():
        print(f"\nlog tail ({log}):")
        print("\n".join(log.read_text(errors="replace").splitlines()[-25:]))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
