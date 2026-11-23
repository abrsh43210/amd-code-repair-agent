#!/usr/bin/env python3
"""Run this INSIDE the built container, before submitting.

It is the pre-submission checklist from Section 2 of the rules, automated:

  1. code at the mandated paths (/app/app.py, /app/requirements.txt, /models)
  2. the model server came up inside the startup budget, on the GPU
  3. a task is answered inside the per-item budget, for every input shape
  4. the output file exists and has the required keys, for every task
  5. peak VRAM stays between 1 GiB and 48 GiB while a task runs
  6. stdout stays clean
  7. no secrets in the image
  8. the test runner can tell a failing pytest test from a passing one

    docker run --rm -it --device=/dev/kfd --device=/dev/dri <image> \
        bash -c 'python3 /app/mc5/server.py >/tmp/s.out 2>&1 & python3 /app/tests/selfcheck.py'

(tests/ is not copied into the image by the build; mount it or paste this file in.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

APP = Path("/app/app.py")
OUT = Path("/app/output")
IN = Path("/app/input")
PER_ITEM_BUDGET_S = 30.0
STARTUP_BUDGET_S = 600.0

TASK = {
    "task_id": "selfcheck_01",
    "description": "total_of drops the last element.",
    "file": "solution.py",
    "code": "def total_of(values):\n    total = 0\n    for i in range(len(values) - 1):\n        total += values[i]\n    return total\n",
    "test_file": "test_solution.py",
    "tests": "from solution import total_of\nassert total_of([1, 2, 3]) == 6\nassert total_of([]) == 0\nprint('ok')\n",
}

fails: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> None:
    (notes if cond else fails).append(("ok   " if cond else "FAIL ") + msg)


# 1 - paths
check(APP.is_file(), "/app/app.py exists")
check(Path("/app/requirements.txt").is_file(), "/app/requirements.txt exists")
check(Path("/models").is_dir(), "/models exists")
check(Path("/app/mc5/server.py").is_file(), "/app/mc5 package present")


# 5 - VRAM sampler (the harness samples every 3 s and takes the peak)
def read_vram_bytes() -> int:
    for cmd, pat in ((["amd-smi", "metric", "--mem"], r"USED_VRAM[^0-9]*(\d+)"),
                     (["rocm-smi", "--showmeminfo", "vram", "--csv"], r"(\d{6,})")):
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        m = re.search(pat, out)
        if m:
            v = int(m.group(1))
            return v * 1024 * 1024 if v < 10 ** 6 else v      # MiB or bytes
    return -1


peak = {"v": -1}
stop = threading.Event()


def sampler() -> None:
    while not stop.is_set():
        v = read_vram_bytes()
        if v > peak["v"]:
            peak["v"] = v
        stop.wait(1.0)


threading.Thread(target=sampler, daemon=True).start()

# 2 - warm up (the server should already be running; this waits for it)
t0 = time.time()
warm = subprocess.run([sys.executable, str(APP), "--warmup"], capture_output=True, text=True,
                      timeout=STARTUP_BUDGET_S + 30)
startup = time.time() - t0
check(startup < STARTUP_BUDGET_S, f"model ready in {startup:.0f}s (budget {STARTUP_BUDGET_S:.0f}s)")
check(not warm.stdout.strip(), f"warmup wrote nothing to stdout ({warm.stdout[:80]!r})")

# 3 + 4 + 6 - answer a task in each input shape
IN.mkdir(parents=True, exist_ok=True)
task_path = IN / "selfcheck_01.json"
task_path.write_text(json.dumps(TASK), encoding="utf-8")
shapes = [
    (["--input", str(task_path)], "selfcheck_01"),
    (["--input-task", str(task_path)], "selfcheck_01"),
    ([str(task_path)], "selfcheck_01"),
]
for args, stem in shapes:
    out_file = OUT / f"{stem}_output.json"
    if out_file.exists():
        out_file.unlink()
    t = time.time()
    p = subprocess.run([sys.executable, str(APP)] + args, capture_output=True, text=True,
                       timeout=PER_ITEM_BUDGET_S + 15)
    took = time.time() - t
    label = " ".join(args[:1]) or "positional"
    check(took < PER_ITEM_BUDGET_S, f"{label}: answered in {took:.1f}s (budget {PER_ITEM_BUDGET_S:.0f}s)")
    check(not p.stdout.strip(), f"{label}: stdout clean")
    if not out_file.is_file():
        check(False, f"{label}: no {out_file}")
        continue
    rec = json.loads(out_file.read_text(encoding="utf-8"))
    check(bool(rec.get("code")), f"{label}: output has non-empty 'code'")
    check("patch" in rec, f"{label}: output has 'patch'")
    check(isinstance(rec.get("confidence"), (int, float)), f"{label}: confidence is a number")
    check(bool(rec.get("tests_pass")), f"{label}: the fix passes the task's tests")

# 8 - the test runner inside this image must be able to FAIL. If pytest cannot be
# imported by a test subprocess, a pytest-style test "passes" for every candidate.
sys.path.insert(0, "/app")
sys.path.append("/app/vendor")
from mc5 import paths as _paths, repo as _repo, task as _task, verify as _verify  # noqa: E402
check(_paths.have_pytest(), "pytest can be imported by test subprocesses")
PT = "import pytest\nfrom solution import total_of\n\ndef test_total():\n    assert total_of([1, 2, 3]) == 6\n"
bad = _task.Task(stem="pt", code=TASK["code"], tests=PT, error="", language="python",
                 file_name="solution.py", test_file_name="test_solution.py")
v_bad = _verify.verify(TASK["code"], bad)[0]
v_good = _verify.verify(TASK["code"].replace("len(values) - 1", "len(values)"), bad)[0]
check(v_bad == _verify.FAIL, f"single-file: a known-failing pytest test fails ({v_bad})")
check(v_good == _verify.PASS, f"single-file: the fixed code passes the pytest test ({v_good})")
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "solution.py").write_text(TASK["code"])
    (Path(td) / "test_solution.py").write_text(PT)
    r_bad = _repo.run_pytest(Path(td), ["test_solution.py"], 20)[0]
    (Path(td) / "solution.py").write_text(TASK["code"].replace("len(values) - 1", "len(values)"))
    r_good = _repo.run_pytest(Path(td), ["test_solution.py"], 20)[0]
check(r_bad == "fail", f"repo mode: a known-failing pytest test fails ({r_bad})")
check(r_good == "pass", f"repo mode: the fixed code passes ({r_good})")

stop.set()
time.sleep(0.2)
gib = peak["v"] / 2 ** 30 if peak["v"] > 0 else -1
if gib < 0:
    notes.append("WARN  could not read VRAM (no amd-smi/rocm-smi); check by hand")
else:
    check(1.0 <= gib <= 48.0, f"peak VRAM {gib:.1f} GiB (must be 1-48 GiB)")

# 7 - secrets
leaks = []
for root, dirs, files in os.walk("/app"):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", "output", "input")]
    for f in files:
        if f in (".env", ".netrc", "credentials", "id_rsa") or f.endswith((".pem", ".key")):
            leaks.append(os.path.join(root, f))
check(not leaks, f"no credential files under /app ({leaks})")

print("\n".join(notes))
if fails:
    print("\n".join(fails))
    print(f"\n{len(fails)} CHECK(S) FAILED - do not submit")
    sys.exit(1)
print(f"\nall {len(notes)} checks passed")
