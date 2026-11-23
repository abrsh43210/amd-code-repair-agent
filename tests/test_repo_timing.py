#!/usr/bin/env python3
"""Repo mode must come back inside its deadline whatever the model and the tests do.

    python3 tests/test_repo_timing.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from mc5 import repo  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        fails.append(msg)


SRC = "def total_of(values):\n    total = 0\n    for i in range(len(values) - 1):\n        total += values[i]\n    return total\n"
EDIT = ("shop/cart.py\n<<<<<<< SEARCH\n    for i in range(len(values) - 1):\n=======\n"
        "    for i in range(len(values)):\n>>>>>>> REPLACE\n")
SLOW_TEST = "import time\nfrom shop.cart import total_of\n\ndef test_slow():\n    time.sleep(60)\n    assert total_of([1]) == 1\n"


class SlowLLM:
    def __init__(self, delay):
        self.delay, self.calls = delay, 0

    def chat(self, system, text, max_new_tokens=120, max_time=None):
        self.calls += 1
        time.sleep(self.delay)
        return EDIT if "SEARCH/REPLACE" in system else ""


def make_repo(tmp: Path) -> Path:
    r = tmp / "repo"
    (r / "shop").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "shop" / "__init__.py").write_text("")
    (r / "shop" / "cart.py").write_text(SRC)
    (r / "tests" / "test_cart.py").write_text(SLOW_TEST)
    subprocess.run("git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm base",
                   shell=True, cwd=r, check=True)
    return r


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mc5-timing-"))
    r = make_repo(tmp)
    issue = "total_of in shop/cart.py drops the last element of values"

    # 1. tests that hang: every test run is clamped to the time left
    for budget in (12.0, 25.0):
        t = time.time()
        res = repo.repair_repo(r, issue, "", SlowLLM(1.0), budget)
        took = time.time() - t
        check(took <= budget + 0.5, f"hanging tests, {budget:.0f}s budget: back in {took:.1f}s")
        check(bool(res.patch), f"hanging tests, {budget:.0f}s budget: still ships the patch ({res.label})")

    # 2. a model that answers late in the budget: no test run may start that cannot finish
    t = time.time()
    res = repo.repair_repo(r, issue, "", SlowLLM(8.0), 12.0)
    took = time.time() - t
    check(took <= 12.5, f"slow model, 12s budget: back in {took:.1f}s")
    check(bool(res.patch), f"slow model: still ships the patch ({res.label})")

    # 3. a big repo on a different filesystem from WORK_DIR: hard links fail there, so the
    #    throwaway copy for the test run is a real copy and must be time-boxed
    import os
    from mc5 import paths
    shm = Path("/dev/shm")
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if shm.is_dir() and os.access(shm, os.W_OK) and os.stat(shm).st_dev != os.stat(paths.WORK_DIR).st_dev:
        big = Path(tempfile.mkdtemp(prefix="mc5-bigrepo-", dir=str(shm)))
        try:
            rb = make_repo(big)
            (rb / "tests" / "test_cart.py").write_text("from shop.cart import total_of\n\ndef test_one():\n    assert total_of([]) == 0\n")
            filler = "def f(x):\n    return x + 1\n" * 400
            for d in range(41):
                (rb / "pkg" / f"m{d}").mkdir(parents=True)
                for k in range(100):
                    (rb / "pkg" / f"m{d}" / f"mod{k}.py").write_text(filler)
            before = set(os.listdir(paths.WORK_DIR))
            t = time.time()
            try:
                repo._overlay(rb, {}, budget_s=0.05)
                timed_out = False
            except repo.OverlayTimeout:
                timed_out = True
            check(timed_out and time.time() - t < 1.5, f"cross-filesystem copy of 4000 files stops at its budget ({time.time() - t:.2f}s)")
            check(set(os.listdir(paths.WORK_DIR)) == before, "the partial copy is removed")
            old = repo.OVERLAY_MAX_S
            repo.OVERLAY_MAX_S = 0.05
            t = time.time()
            res = repo.repair_repo(rb, issue, "", SlowLLM(0.5), 25.0)
            took = time.time() - t
            repo.OVERLAY_MAX_S = old
            check(bool(res.patch) and "compiles" in res.label, f"copy over budget: no test verdict, patch still shipped ({res.label})")
            t = time.time()
            res = repo.repair_repo(rb, issue, "", SlowLLM(0.5), 25.0)
            took = time.time() - t
            check(took <= 25.5 and bool(res.patch), f"big repo on another filesystem, 25s budget: back in {took:.1f}s ({res.label})")
            check(res.localized[:1] == ["shop/cart.py"], f"the file the issue names is found among 4000 others ({res.localized[:2]})")
        finally:
            shutil.rmtree(big, ignore_errors=True)
    else:
        print("skip  cross-filesystem test (no second filesystem here)")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
