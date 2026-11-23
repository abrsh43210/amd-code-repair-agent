#!/usr/bin/env python3
"""Offline test of the repo-level mode. No GPU, no network, no third-party packages.

Builds a tiny repository on the fly, then checks the pieces a model's answer has to
survive: localization, SEARCH/REPLACE parsing (clean, re-indented, fenced, chatty),
ambiguity refusal, patch generation that `git apply` accepts, the regression gate,
and the end-to-end app.py invocation writing <id>_output.json with nothing on stdout.
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
sys.path.insert(0, str(ROOT / "app"))

PKG = {
    "shop/__init__.py": "",
    "shop/cart.py": '''"""Shopping cart."""


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty=1):
        self.items.append((name, price, qty))

    def total(self):
        total = 0
        for name, price, qty in self.items:
            total += price
        return total

    def count(self):
        return sum(qty for _, _, qty in self.items)
''',
    "shop/pricing.py": '''def apply_discount(amount, percent):
    """Return amount reduced by percent."""
    return amount - amount * percent / 100


def format_price(amount):
    return f"${amount:.2f}"
''',
    "shop/users.py": '''def display_name(first, last):
    return f"{first} {last}".strip()
''',
    "tests/test_cart.py": '''from shop.cart import Cart


def test_count():
    c = Cart()
    c.add("pen", 2, qty=3)
    assert c.count() == 3
''',
}
ISSUE = ("Cart.total() ignores quantity\n\nAdding 3 pens at 2 each gives `cart.total()` == 2, "
         "expected 6. The `qty` passed to `Cart.add` is dropped in `total`.")
GOLD = ("shop/cart.py\n<<<<<<< SEARCH\n            total += price\n=======\n"
        "            total += price * qty\n>>>>>>> REPLACE")
HIDDEN_TEST = '''from shop.cart import Cart


def test_total_uses_qty():
    c = Cart()
    c.add("pen", 2, qty=3)
    assert c.total() == 6
'''

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def make_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    for rel, text in PKG.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    return repo


class Scripted:
    def __init__(self, *replies, rewrite=""):
        self.replies = list(replies)
        self.calls = 0
        self.rewrite = rewrite
        self.rewrites = 0

    def chat(self, system, text, max_new_tokens=120, max_time=None):
        if "single Python code block" in system and "fixing an issue" in system:
            self.rewrites += 1
            return self.rewrite
        if "SEARCH/REPLACE" not in system:
            return ""
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


def main() -> int:
    from mc5 import localize, paths, repo as R

    tmp = Path(tempfile.mkdtemp(prefix="mc5-repotest-"))
    paths.WORK_DIR = tmp / "work"
    paths.LOG_FILE = paths.WORK_DIR / "app.log"
    repo = make_repo(tmp)
    have_git = shutil.which("git") is not None
    try:
        import pytest  # noqa: F401
        have_pytest = True
    except ImportError:
        have_pytest = False

    print("localization")
    chunks = localize.rank(repo, ISSUE)
    check(bool(chunks) and chunks[0].path == "shop/cart.py" and chunks[0].name.startswith("Cart"),
          f"top chunk is the Cart class in shop/cart.py (got {chunks[0] if chunks else None})")
    check(not any(localize.is_test_path(c.path) for c in chunks), "tests are never offered as a place to edit")

    print("edit parsing and applying")
    changed, probs = R.apply_edits(repo, R.parse_edits(GOLD))
    check(list(changed) == ["shop/cart.py"] and "price * qty" in changed["shop/cart.py"], "clean block applies")

    chatty = "Sure! Here is the fix:\n\n```\n" + GOLD + "\n```\nThis multiplies by qty."
    changed2, _ = R.apply_edits(repo, R.parse_edits(chatty))
    check(changed2 == changed, "block inside a fence with prose around it applies")

    dedented = ("shop/cart.py\n<<<<<<< SEARCH\nfor name, price, qty in self.items:\n    total += price\n=======\n"
                "for name, price, qty in self.items:\n    total += price * qty\n>>>>>>> REPLACE")
    changed3, _ = R.apply_edits(repo, R.parse_edits(dedented))
    check(changed3 == changed, "block with the wrong indentation is re-indented to match the file")

    basename = GOLD.replace("shop/cart.py", "cart.py")
    changed4, _ = R.apply_edits(repo, R.parse_edits(basename))
    check(changed4 == changed, "a bare file name is resolved to its path")

    amb = "shop/cart.py\n<<<<<<< SEARCH\n        return total\n=======\n        return total + 0\n>>>>>>> REPLACE"
    ch_amb, probs_amb = R.apply_edits(repo, R.parse_edits(amb))
    check(bool(ch_amb), "unique SEARCH text applies")
    amb2 = "shop/pricing.py\n<<<<<<< SEARCH\n    return\n=======\n    return 1\n>>>>>>> REPLACE"
    ch_amb2, probs2 = R.apply_edits(repo, R.parse_edits(amb2))
    check(not ch_amb2 and bool(probs2), "SEARCH text that is not present is refused, not guessed")

    typo = ("shop/cart.py\n<<<<<<< SEARCH\n        for name, price, qty in self.item:\n            total += price\n=======\n"
            "        for name, price, qty in self.items:\n            total += price * qty\n>>>>>>> REPLACE")
    ch_typo, pr_typo = R.apply_edits(repo, R.parse_edits(typo))
    check(ch_typo == changed, f"SEARCH text with a small retyping slip still lands on the one close match {pr_typo}")

    twin_repo = tmp / "twins"
    (twin_repo / "pkg").mkdir(parents=True)
    (twin_repo / "pkg" / "client.py").write_text(
        "class Client:\n    def send(self, r):\n        self.log(r)\n        return self.go(r)\n\n"
        "class AsyncClient:\n    async def send(self, r):\n        self.log(r)\n        return self.go(r)\n")
    twin = "pkg/client.py\n<<<<<<< SEARCH\n        self.log(r)\n        return self.go(r)\n=======\n        self.log(r)\n        self.check(r)\n        return self.go(r)\n>>>>>>> REPLACE"
    ch_t, pr_t = R.apply_edits(twin_repo, R.parse_edits(twin + "\n\n" + twin))
    check(ch_t.get("pkg/client.py", "").count("self.check(r)") == 2, "two identical blocks edit the sync and async twins in order")
    ch_t1, pr_t1 = R.apply_edits(twin_repo, R.parse_edits(twin))
    check(not ch_t1 and "occurs 2 times" in " ".join(pr_t1), "one block matching two places is refused with a useful message")

    print("fallback matching")
    (repo / "shop/gaps.py").write_text("def build(p):\n    url = []\n    path = p.path\n    if not path:\n        path = '/'\n\n    url.append(path)\n\n    return url\n")
    noblank = ("shop/gaps.py\n<<<<<<< SEARCH\n    path = p.path\n    if not path:\n        path = '/'\n    url.append(path)\n=======\n"
               "    path = p.path\n    url.append(path)\n>>>>>>> REPLACE")
    ch_b, pr_b = R.apply_edits(repo, R.parse_edits(noblank))
    check(ch_b.get("shop/gaps.py") == "def build(p):\n    url = []\n    path = p.path\n    url.append(path)\n\n    return url\n",
          f"SEARCH that left out a blank line still matches {pr_b}")
    (repo / "shop/gaps.py").unlink()
    shown = {"shop/cart.py": [(1, 20)], "shop/pricing.py": [(1, 8)]}
    elided = ("shop/cart.py\n<<<<<<< SEARCH\n    def total(self):\n        ...\n        return total\n=======\n"
              "    def total(self):\n        return sum(price * qty for _, price, qty in self.items)\n>>>>>>> REPLACE")
    ch_e, pr_e = R.apply_edits(repo, R.parse_edits(elided), shown)
    check("price * qty for" in ch_e.get("shop/cart.py", "") and "total += price" not in ch_e.get("shop/cart.py", "")
          and "def count" in ch_e.get("shop/cart.py", ""), f"SEARCH abbreviated with ... replaces the whole span {pr_e}")
    elided2 = ("shop/cart.py\n<<<<<<< SEARCH\n    def total(self):\n        total = 0\n        # ...\n        return total\n=======\n"
               "    def total(self):\n        total = 0.0\n        # ...\n        return round(total, 2)\n>>>>>>> REPLACE")
    ch_e2, pr_e2 = R.apply_edits(repo, R.parse_edits(elided2), shown)
    t_e2 = ch_e2.get("shop/cart.py", "")
    check("total = 0.0" in t_e2 and "total += price" in t_e2 and "round(total, 2)" in t_e2 and R.compiles(ch_e2)[0],
          f"... on both sides keeps the middle of the file {pr_e2}")
    wrongfile = ("shop/users.py\n<<<<<<< SEARCH\n            total += price\n=======\n"
                 "            total += price * qty\n>>>>>>> REPLACE")
    ch_w, pr_w = R.apply_edits(repo, R.parse_edits(wrongfile), shown)
    check(list(ch_w) == ["shop/cart.py"], f"right text under the wrong file name lands in the one shown file that has it {pr_w}")
    loose = ("shop/cart.py\n<<<<<<< SEARCH\n    def total(self):\n        total = 0\n        for item_name, item_price, quantity in self.items:\n"
             "            total += price\n        return total\n=======\n    def total(self):\n        total = 0\n"
             "        for name, price, qty in self.items:\n            total += price * qty\n        return total\n>>>>>>> REPLACE")
    ch_l, pr_l = R.apply_edits(repo, R.parse_edits(loose), shown)
    check(ch_l == changed, f"a badly retyped SEARCH with the right first and last line still lands {pr_l}")
    invented = ("shop/cart.py\n<<<<<<< SEARCH\n    def total(self):\n        return sum(p for _, p, _ in self.items)\n=======\n"
                "    def total(self):\n        return 1\n>>>>>>> REPLACE")
    ch_i, pr_i = R.apply_edits(repo, R.parse_edits(invented), shown)
    check(not ch_i and "closest real lines" in " ".join(pr_i) and "def total(self):" in " ".join(pr_i),
          "invented SEARCH text is refused and the feedback quotes the real lines")

    print("whole-function rewrite fallback")
    good_rw = "```python\ndef total(self):\n    total = 0\n    for name, price, qty in self.items:\n        total += price * qty\n    return total\n```"
    llm_rw = Scripted(invented, rewrite=good_rw)
    res = R.repair_repo(repo, ISSUE, llm=llm_rw, deadline_s=20)
    check("price * qty" in res.patch and llm_rw.rewrites == 1 and "rewrite" in res.label,
          f"unappliable edit -> the function is rewritten whole and re-indented (label={res.label})")
    if have_git and res.patch:
        (tmp / "rw.diff").write_text(res.patch, encoding="utf-8")
        rc = subprocess.run(["git", "apply", "--check", str(tmp / "rw.diff")], cwd=repo, capture_output=True).returncode
        check(rc == 0, "the rewrite patch applies with git")
    llm_rw2 = Scripted(invented, rewrite="```python\ndef total(self):\n    total = 0\n```")
    res = R.repair_repo(repo, ISSUE, llm=llm_rw2, deadline_s=20)
    check(res.patch == "", "a rewrite that lost half the function is rejected")
    llm_rw3 = Scripted(invented, rewrite="```python\ndef other(self):\n    total = 0\n    for a in self.items:\n        pass\n    return 1\n```")
    res = R.repair_repo(repo, ISSUE, llm=llm_rw3, deadline_s=20)
    check(res.patch == "", "a rewrite of a different function is rejected")
    llm_ok = Scripted(GOLD, rewrite=good_rw)
    res = R.repair_repo(repo, ISSUE, llm=llm_ok, deadline_s=25)
    check(llm_ok.rewrites == 0 and "price * qty" in res.patch, "no rewrite when the edit applied")

    print("patch")
    patch = R.make_patch(repo, changed)
    check(patch.startswith("diff --git a/shop/cart.py b/shop/cart.py"), "patch has a git header")
    if have_git:
        work = tmp / "apply"
        shutil.copytree(repo, work)
        subprocess.run(["git", "init", "-q"], cwd=work)
        (work / "p.diff").write_text(patch, encoding="utf-8")
        rc = subprocess.run(["git", "apply", "--check", "p.diff"], cwd=work, capture_output=True).returncode
        check(rc == 0, "`git apply --check` accepts the patch")

    print("pipeline")
    res = R.repair_repo(repo, ISSUE, "", Scripted(GOLD), deadline_s=60)
    check("price * qty" in res.patch, f"correct fix is returned as a patch (label={res.label})")
    check((repo / "shop/cart.py").read_text() == PKG["shop/cart.py"], "the repository on disk is left untouched")

    bad_syntax = GOLD.replace("total += price * qty", "total += price * (qty")
    res = R.repair_repo(repo, ISSUE, "", Scripted(bad_syntax, GOLD), deadline_s=60)
    check("price * qty" in res.patch and "(qty" not in res.patch, "a syntax error is rejected and the retry wins")

    res = R.repair_repo(repo, ISSUE, "", Scripted("I think the bug is in total().", "no blocks here", ""), deadline_s=60)
    check(res.patch == "", "no usable answer -> empty patch, never a guess")

    if have_pytest:
        breaker = ("shop/cart.py\n<<<<<<< SEARCH\n        return sum(qty for _, _, qty in self.items)\n=======\n"
                   "        return len(self.items)\n>>>>>>> REPLACE")
        res = R.repair_repo(repo, ISSUE, "", Scripted(breaker, breaker, breaker), deadline_s=90)
        check("regression" in res.label and res.confidence < 0.2,
              "a change that breaks an existing test is flagged and shipped only as a low-confidence last resort")
        res = R.repair_repo(repo, ISSUE, "", Scripted(GOLD, breaker), deadline_s=90)
        check("price * qty" in res.patch, "a clean candidate always beats one with a regression")
        res = R.repair_repo(repo, ISSUE, "", Scripted(breaker, GOLD), deadline_s=90)
        check("price * qty" in res.patch and "len(self.items)" not in res.patch,
              f"after a regression the corrected retry wins (label={res.label})")
    else:
        print("  skip  regression gate (pytest not installed here)")

    print("app.py end to end")
    task = tmp / "in" / "shop_1.json"
    task.parent.mkdir()
    task.write_text(json.dumps({"instance_id": "shop_1", "repo": "x/shop", "base_commit": "0" * 40,
                                "problem_statement": ISSUE, "hints_text": "", "repo_path": str(repo)}))
    oracle = tmp / "oracle.json"
    oracle.write_text(json.dumps({"__repo__": GOLD}))
    out = tmp / "out"
    env = {**os.environ, "MC5_OUTPUT_DIR": str(out), "MC5_WORK_DIR": str(tmp / "w2"),
           "MC5_SOCKET": str(tmp / "m.sock"), "MC5_FAKE_LLM": "1", "MC5_FAKE_ORACLE": str(oracle),
           "MC5_REPO_DEADLINE_S": "60", "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, str(ROOT / "app" / "app.py"), "--input", str(task)],
                       capture_output=True, text=True, env=env, timeout=180)
    check(not p.stdout.strip(), "stdout stays clean")
    f = out / "shop_1_output.json"
    check(f.is_file(), "shop_1_output.json written")
    if f.is_file():
        rec = json.loads(f.read_text())
        check("price * qty" in rec.get("patch", "") and rec.get("prediction") == rec.get("patch"),
              "output carries the patch under patch / prediction / model_patch")
        check(rec.get("files", {}).get("shop/cart.py", "").count("price * qty") == 1, "output carries the full fixed file")
        if have_git and have_pytest:
            g = tmp / "grade"
            shutil.copytree(repo, g)
            subprocess.run(["git", "init", "-q"], cwd=g)
            (g / "a.diff").write_text(rec["patch"])
            subprocess.run(["git", "apply", "a.diff"], cwd=g)
            (g / "tests" / "test_hidden.py").write_text(HIDDEN_TEST)
            rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                                cwd=g, capture_output=True, env={**os.environ, "PYTHONPATH": str(g)}).returncode
            check(rc == 0, "the hidden test passes after applying the patch")
    p2 = subprocess.run([sys.executable, str(ROOT / "app" / "app.py"), "--repo", str(repo), "--issue", ISSUE,
                         "--task-id", "shop_2"], capture_output=True, text=True, env=env, timeout=180)
    check((out / "shop_2_output.json").is_file() and not p2.stdout.strip(), "--repo / --issue flags work too")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
