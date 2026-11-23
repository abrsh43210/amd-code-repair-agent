#!/usr/bin/env python3
"""The test runner must work when pytest lives only in app/vendor (as in the graded
image) and must never read "pytest is missing" as a test verdict.

    python3 tests/test_runner_env.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = "def total_of(values):\n    total = 0\n    for i in range(len(values) - 1):\n        total += values[i]\n    return total\n"
GOOD = CODE.replace("len(values) - 1", "len(values)")
PT = "import pytest\nfrom solution import total_of\n\ndef test_total():\n    assert total_of([1, 2, 3]) == 6\n"

PROBE = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.append(sys.argv[1] + "/vendor")
from mc5 import paths, repo, task, verify
CODE, GOOD, PT = sys.argv[2], sys.argv[3], sys.argv[4]
t = task.Task(stem="pt", code=CODE, tests=PT, error="", language="python",
              file_name="solution.py", test_file_name="test_solution.py")
out = {"have": paths.have_pytest(), "v_bad": verify.verify(CODE, t)[0], "v_good": verify.verify(GOOD, t)[0]}
t2 = task.Task(stem="pt", code=CODE, tests=PT.replace("import pytest\n", ""), error="", language="python",
               file_name="solution.py", test_file_name="test_solution.py")
out["m_bad"] = verify.verify(CODE, t2)[0]; out["m_good"] = verify.verify(GOOD, t2)[0]
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "solution.py").write_text(CODE); (Path(td) / "test_solution.py").write_text(PT)
    out["r_bad"] = repo.run_pytest(Path(td), ["test_solution.py"], 30)[0]
    (Path(td) / "solution.py").write_text(GOOD)
    out["r_good"] = repo.run_pytest(Path(td), ["test_solution.py"], 30)[0]
print(json.dumps(out))
'''

fails = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def probe(app_dir: Path, python: list[str]) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MC5_WORK_DIR"] = tempfile.mkdtemp(prefix="mc5-t-")
    p = subprocess.run(python + ["-c", PROBE, str(app_dir), CODE, GOOD, PT], capture_output=True, text=True,
                       env=env, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    import pytest
    tmp = Path(tempfile.mkdtemp(prefix="mc5-vendor-"))
    # A copy of the app with pytest (and what it needs) ONLY in app/vendor, run by an
    # interpreter that cannot see site-packages: the shape of the graded image.
    app = tmp / "app"
    shutil.copytree(ROOT / "app", app, ignore=shutil.ignore_patterns("__pycache__", "vendor"))
    site = Path(pytest.__file__).resolve().parent.parent
    (app / "vendor").mkdir()
    for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments", "exceptiongroup", "tomli", "py"):
        for cand in (site / name, site / f"{name}.py"):
            if cand.is_dir():
                shutil.copytree(cand, app / "vendor" / cand.name, ignore=shutil.ignore_patterns("__pycache__"))
            elif cand.is_file():
                shutil.copy2(cand, app / "vendor" / cand.name)
    for d in site.glob("*.dist-info"):
        if d.name.lower().startswith(("pytest", "pluggy", "iniconfig", "packaging", "pygments")):
            shutil.copytree(d, app / "vendor" / d.name)
    import venv
    venv.create(tmp / "venv", with_pip=False)   # an interpreter with no pytest of its own
    py = [str(tmp / "venv" / "bin" / "python")]
    r = probe(app, py)
    check(r["have"], "vendor-only: pytest is found")
    check(r["v_bad"] == "fail", f"vendor-only single-file: failing test fails ({r['v_bad']})")
    check(r["v_good"] == "pass", f"vendor-only single-file: fixed code passes ({r['v_good']})")
    check(r["r_bad"] == "fail", f"vendor-only repo mode: failing test fails ({r['r_bad']})")
    check(r["r_good"] == "pass", f"vendor-only repo mode: fixed code passes ({r['r_good']})")

    shutil.rmtree(app / "vendor")        # no pytest anywhere
    r = probe(app, py)
    check(not r["have"], "no pytest: reported as missing")
    check(r["r_bad"] == "error" and r["r_good"] == "error",
          f"no pytest, repo mode: verdict is 'error', not a test result ({r['r_bad']}, {r['r_good']})")
    check(r["v_bad"] == "unknown" and r["v_good"] == "unknown",
          f"no pytest, tests import pytest: no verdict, never a pass ({r['v_bad']}, {r['v_good']})")
    check(r["m_bad"] == "fail" and r["m_good"] == "pass",
          f"no pytest, plain def test_: the stdlib runner tells them apart ({r['m_bad']}, {r['m_good']})")

    # argparse rejecting the command line must still leave an output file
    out = tmp / "out"
    task_file = tmp / "t1.json"
    task_file.write_text(json.dumps({"task_id": "t1", "file": "solution.py", "code": CODE,
                                     "test_file": "test_solution.py",
                                     "tests": "from solution import total_of\nassert total_of([1, 2, 3]) == 6\n"}))
    env = {**os.environ, "MC5_OUTPUT_DIR": str(out), "MC5_INPUT_DIR": str(tmp / "none"),
           "MC5_WORK_DIR": str(tmp / "work"), "MC5_SOCKET": str(tmp / "s.sock")}
    p = subprocess.run([sys.executable, str(ROOT / "app" / "app.py"), "--no-llm", "--language", "--input", str(task_file)],
                       capture_output=True, text=True, env=env, timeout=120)
    p = subprocess.run([sys.executable, str(ROOT / "app" / "app.py"), "--no-llm", str(task_file), "--task-id"],
                       capture_output=True, text=True, env=env, timeout=120)
    check((out / "t1_output.json").is_file(), "a command line argparse rejects still gets an output file")
    check(not p.stdout.strip(), "and stdout stays clean")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
