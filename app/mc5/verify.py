"""Does a candidate fix actually work?

This is the scoring lever. Any model, large or small, writes plausible-looking
patches; the ones that count are the ones whose tests pass. So every candidate is
run in a throwaway directory in a separate process, and only a candidate that
compiles (and, where tests came with the task, passes them) is allowed to win.

Python is run for real. Other languages are compile-checked where a compiler
happens to be present, and otherwise only syntax-screened, because the graded
container is not guaranteed to have a toolchain for them.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import paths

# What a verification attempt concluded.
PASS = "pass"            # tests ran and passed
COMPILES = "compiles"    # no tests to run, but the code is valid
FAIL = "fail"            # tests ran and failed
BROKEN = "broken"        # does not even parse / compile
UNKNOWN = "unknown"      # we could not tell (no toolchain)

RANK = {PASS: 4, COMPILES: 3, UNKNOWN: 2, FAIL: 1, BROKEN: 0}


def syntax_ok(code: str, language: str) -> bool:
    if language == "python":
        try:
            compile(code, "<candidate>", "exec")
            return True
        except SyntaxError:
            return False
    # Cheap brace/paren balance screen for the C-family and friends.
    depth = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    in_s: str | None = None
    prev = ""
    for ch in code:
        if in_s:
            if ch == in_s and prev != "\\":
                in_s = None
        elif ch in "\"'":
            in_s = ch
        elif ch in depth:
            depth[ch] += 1
        elif ch in pairs:
            depth[pairs[ch]] -= 1
            if depth[pairs[ch]] < 0:
                return False
        prev = ch
    return all(v == 0 for v in depth.values())


def _run(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, capture_output=True,
                           text=True, errors="replace",
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONPATH": paths.test_pythonpath(cwd), "MPLBACKEND": "Agg"})
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as e:
        return 127, f"{type(e).__name__}: {e}"


# pytest's own summary line. Without it a non-zero exit is pytest failing to start
# (not installed, a broken plugin), which says nothing about the candidate.
PYTEST_VERDICT = re.compile(r"\b\d+ (failed|errors?|passed)\b")

# Stdlib stand-in for pytest: import the test file and call every zero-argument test.
# Exit 0 = all passed, 1 = a test failed, 5 = nothing we could run (so: no verdict).
_MINI_RUNNER = r"""
import importlib.util, inspect, sys, traceback
spec = importlib.util.spec_from_file_location("mc5_tests", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except ModuleNotFoundError as e:
    traceback.print_exc(); sys.exit(5 if e.name in ("pytest", "_pytest") else 1)
except BaseException:
    traceback.print_exc(); sys.exit(1)
def _noargs(fn):
    try:
        return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                   for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
ran = skipped = 0
def _call(fn):
    global ran
    ran += 1
    try:
        fn()
    except BaseException:
        traceback.print_exc(); sys.exit(1)
for name, obj in list(vars(mod).items()):
    if name.startswith("test") and inspect.isfunction(obj):
        if _noargs(obj): _call(obj)
        else: skipped += 1
    elif name.startswith("Test") and inspect.isclass(obj):
        try:
            inst = obj()
        except BaseException:
            skipped += 1; continue
        for mname in dir(inst):
            m = getattr(inst, mname)
            if mname.startswith("test") and callable(m):
                if _noargs(m):
                    for hook in ("setup_method", "setUp"):
                        if hasattr(inst, hook) and _noargs(getattr(inst, hook)): getattr(inst, hook)()
                    _call(m)
                else: skipped += 1
sys.exit(0 if ran and not skipped else (5 if not ran else 6))
"""


def _python_test_cmd(cwd: Path, test_name: str, tests: str) -> tuple[list[str], str]:
    """(command, kind). pytest when a test subprocess can import it and the tests look
    like pytest; unittest for unittest files; our stdlib runner for bare `def test_`
    files when pytest is missing (executing such a file proves nothing: it defines the
    tests and exits 0 without calling them); else just execute the script."""
    looks_pytest = "pytest" in tests or re.search(r"^\s*def test", tests, re.M) is not None
    if "unittest" in tests and "pytest" not in tests:
        return [sys.executable, "-m", "unittest", "-q", Path(test_name).stem], "unittest"
    if looks_pytest and paths.have_pytest():
        return [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", test_name], "pytest"
    if looks_pytest:
        return [sys.executable, "-c", _MINI_RUNNER, test_name], "mini"
    return [sys.executable, test_name], "script"


def verify(code: str, task, timeout: float | None = None) -> tuple[str, str]:
    """Returns (verdict, output). Never raises."""
    timeout = paths.VERIFY_TIMEOUT_S if timeout is None else timeout
    if not code.strip():
        return BROKEN, "empty"
    if not syntax_ok(code, task.language):
        return BROKEN, "syntax"

    try:
        paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="mc5-verify-", dir=str(paths.WORK_DIR)))
    except OSError:
        tmp = Path(tempfile.mkdtemp(prefix="mc5-verify-"))
    try:
        src = tmp / (Path(task.file_name).name or "solution.py")
        src.write_text(code, encoding="utf-8")

        if task.language != "python":
            return _verify_other(code, task, tmp, timeout)

        if not task.tests.strip():
            rc, out = _run([sys.executable, "-c",
                            f"import importlib.util,sys;"
                            f"spec=importlib.util.spec_from_file_location('cand',r'{src}');"
                            f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)"],
                           tmp, timeout)
            # An import-time crash is a real defect; a module that imports cleanly is
            # the best we can say without tests.
            return (COMPILES, out) if rc == 0 else (FAIL, out)

        test_name = Path(task.test_file_name).name or "test_solution.py"
        tests = task.tests
        # Tests usually import the module by name; make both the task's name and the
        # common ones resolve to this candidate.
        (tmp / test_name).write_text(tests, encoding="utf-8")
        stem = src.stem
        for alias in ("solution", "main", "module_under_test", "buggy", "code"):
            # only the names the tests really import: a stray code.py shadows the stdlib
            # module of that name and breaks pytest itself
            if alias != stem and re.search(rf"^\s*(from|import)\s+{alias}\b", tests, re.M):
                try:
                    (tmp / f"{alias}.py").write_text(f"from {stem} import *  # noqa\n", encoding="utf-8")
                except OSError:
                    pass
        cmd, kind = _python_test_cmd(tmp, test_name, tests)
        rc, out = _run(cmd, tmp, timeout)
        if rc == 0:
            return PASS, out
        if rc == 124:
            return FAIL, out               # timed out: an endless loop is a real defect
        if kind == "pytest" and (rc in (3, 4, 5) or not PYTEST_VERDICT.search(out)):
            return UNKNOWN, out            # the runner broke or found nothing: not a verdict
        if kind == "mini" and rc in (5, 6):
            # 5: nothing runnable; 6: what we could run passed, but some tests need fixtures
            return UNKNOWN, out or "tests need pytest"
        return FAIL, out
    except Exception as e:  # noqa: BLE001
        paths.log(f"verify crashed: {type(e).__name__}: {e}")
        return UNKNOWN, str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verify_other(code: str, task, tmp: Path, timeout: float) -> tuple[str, str]:
    lang = task.language
    src = tmp / (Path(task.file_name).name or "main.txt")
    checks = {
        "c": (["gcc", "-fsyntax-only", src.name], "gcc"),
        "cpp": (["g++", "-fsyntax-only", src.name], "g++"),
        "javascript": (["node", "--check", src.name], "node"),
        "go": (["gofmt", "-e", src.name], "gofmt"),
        "rust": (["rustc", "--edition", "2021", "--emit=metadata", src.name], "rustc"),
        "java": (["javac", "-d", ".", src.name], "javac"),
    }
    cmd_tool = checks.get(lang)
    if not cmd_tool or not shutil.which(cmd_tool[1]):
        return UNKNOWN, f"no toolchain for {lang}"
    rc, out = _run(cmd_tool[0], tmp, timeout)
    return (COMPILES, out) if rc == 0 else (BROKEN, out)


def better(a: str, b: str) -> bool:
    """Is verdict a strictly better than verdict b?"""
    return RANK.get(a, 0) > RANK.get(b, 0)
