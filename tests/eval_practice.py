#!/usr/bin/env python3
"""Score the repo-level mode on SWE-bench-style practice tasks.

Task file: JSONL in the Kaggle "Gemma 4 Developer Agent" shape
(instance_id, repo, base_commit, problem_statement, hints_text, patch, test_patch).
The reference `patch` and the `test_patch` are NEVER shown to the agent; they are
used here only to grade, exactly as the hidden harness would:
    apply agent patch -> apply test_patch -> run the tests -> PASS / FAIL

Modes
  localize  model-free: is the file / function the reference fix touches in our top-k?
  oracle    plumbing: a stand-in "model" answers with the reference fix as SEARCH/REPLACE
            blocks. Anything short of 100% here is OUR bug (parsing, applying, diffing,
            a regression check that rejects a correct fix), not a model weakness.
  llm       the real model through the resident server (needs a GPU; not run in CI)

    python3 tests/eval_practice.py --tasks practice/tasks.jsonl --repos /path/to/clones --mode localize
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))


def sh(args: list[str], cwd: Path, timeout: float = 300, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, errors="replace",
                           timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def materialise(task: dict, repos: Path | None, snapshots: Path | None, work: Path) -> Path | None:
    dst = work / task["instance_id"]
    shutil.rmtree(dst, ignore_errors=True)
    if snapshots and (snapshots / f"{task['instance_id']}.tgz").is_file():
        dst.mkdir(parents=True)
        shutil.unpack_archive(str(snapshots / f"{task['instance_id']}.tgz"), str(dst), filter="data")
        tree = dst
        for _ in range(4):                      # archives may wrap the repo in one or more folders
            kids = list(tree.iterdir())
            if len(kids) == 1 and kids[0].is_dir() and kids[0].name != ".git":
                tree = kids[0]
            else:
                break
        if (tree / ".git").exists() and not any(tree.glob("*.py")) and not any(tree.glob("*/*.py")):
            sh(["git", "checkout", "-f", "HEAD"], cwd=tree)     # archive held only the git data
        sh(["git", "config", "--global", "--add", "safe.directory", "*"], cwd=tree)
        if os.environ.get("MC5_EVAL_DEBUG"):
            n = sum(1 for _ in tree.rglob("*.py"))
            print(f"    [debug] tree={tree} entries={sorted(k.name for k in tree.iterdir())[:25]} py_files={n}")
        return tree
    if repos:
        src = repos / task["repo"].split("/")[-1]
        if src.is_dir():
            rc, out = sh(["git", "-C", str(src), "worktree", "add", "--detach", "--force", str(dst),
                          task["base_commit"]], cwd=src)
            return dst if rc == 0 else None
    return None


def release(task: dict, repos: Path | None, tree: Path) -> None:
    if repos:
        src = repos / task["repo"].split("/")[-1]
        sh(["git", "-C", str(src), "worktree", "remove", "--force", str(tree)], cwd=src)
    shutil.rmtree(tree, ignore_errors=True)


def _nl(patch: str) -> str:
    return patch if not patch or patch.endswith("\n") else patch + "\n"


def with_git_headers(patch: str) -> str:
    return _nl(_with_git_headers(patch))


def _with_git_headers(patch: str) -> str:
    """Some task sets ship plain unified diffs (---/+++ only). Add the `diff --git` lines we key on."""
    if not patch or re.search(r"^diff --git ", patch, re.M):
        return patch
    out, lines = [], patch.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            new = lines[i + 1][4:].split("\t")[0].strip()
            old_ = ln[4:].split("\t")[0].strip()
            name = new if new != "/dev/null" else old_
            name = re.sub(r"^[ab]/", "", name)
            out.append(f"diff --git a/{name} b/{name}")
        out.append(ln)
    return "\n".join(out)


def patch_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/(\S+) b/", patch, re.M)


def patch_hunks(patch: str) -> list[tuple[str, int, int]]:
    """[(file, first_old_line, last_old_line)] for each hunk of a unified diff."""
    out, cur = [], None
    for ln in patch.split("\n"):
        m = re.match(r"^diff --git a/(\S+) b/", ln)
        if m:
            cur = m.group(1)
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+", ln)
        if m and cur:
            a, n = int(m.group(1)), int(m.group(2) or 1)
            out.append((cur, a, a + max(n, 1) - 1))
    return out


def gold_as_edit_blocks(patch: str) -> str:
    """The reference fix rewritten as the SEARCH/REPLACE blocks we ask a model for."""
    blocks, cur, search, repl = [], None, [], []

    def flush():
        if cur and (search or repl):
            blocks.append(f"{cur}\n<<<<<<< SEARCH\n" + "\n".join(search) + "\n=======\n" + "\n".join(repl)
                          + "\n>>>>>>> REPLACE")

    in_hunk = False
    for ln in patch.split("\n"):
        m = re.match(r"^diff --git a/(\S+) b/", ln)
        if m:
            flush()
            cur, search, repl, in_hunk = m.group(1), [], [], False
            continue
        if ln.startswith("@@"):
            flush()
            search, repl, in_hunk = [], [], True
            continue
        if not in_hunk or ln.startswith("\\"):
            continue
        if ln.startswith("-"):
            search.append(ln[1:])
        elif ln.startswith("+"):
            repl.append(ln[1:])
        elif ln.startswith(" ") or ln == "":
            search.append(ln[1:])
            repl.append(ln[1:])
    flush()
    return "\n\n".join(blocks)


class OracleLLM:
    def __init__(self, patch: str):
        self.reply = gold_as_edit_blocks(patch)

    def chat(self, system, text, max_new_tokens=120, max_time=None):
        if "SEARCH/REPLACE" not in system:
            return ""                                    # the oracle writes no reproduction test
        return self.reply


def diagnose(gold: str, agent: str, localized: list[str]) -> str:
    """Why a miss missed, by comparing where the agent edited with where the reference did."""
    gold_files = set(patch_files(gold))
    if not agent.strip():
        return "no-patch" if gold_files & set(localized[:6]) else "no-patch+file-not-found"
    mine = set(patch_files(agent))
    if not (mine & gold_files):
        return "wrong-file"
    g, a = patch_hunks(gold), patch_hunks(agent)
    touched = sum(1 for f, s0, e0 in g if any(f == f2 and s2 <= e0 + 3 and s0 - 3 <= e2 for f2, s2, e2 in a))
    if touched == 0:
        return "right-file-wrong-place"
    if touched < len(g):
        return f"partial-{touched}-of-{len(g)}-hunks"
    return "right-place-wrong-fix"


def write_details(rows: list[dict], path: str, cap: int = 45) -> None:
    """Readable account of every miss: what the agent did next to what the reference did."""
    def cut(text):
        ls = (text or "(none)").rstrip().split("\n")
        return "\n".join(ls[:cap] + ([f"... ({len(ls) - cap} more lines)"] if len(ls) > cap else []))
    out = []
    for r in rows:
        if r.get("pass"):
            continue
        out += [f"##### {r['id']}  {r.get('kind')}  {r.get('label')}  {r.get('secs')}s",
                "--- steps", *[ln[-200:] for ln in r.get("log", []) if "repo:" in ln],
                "--- test output", (r.get("why") or "")[-500:],
                "--- agent patch", cut(r.get("agent_patch")), "--- reference patch", cut(r.get("gold_patch")), ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")


def grade(task: dict, tree: Path, agent_patch: str, test_timeout: float) -> tuple[bool, str]:
    if not agent_patch.strip():
        return False, "no patch"
    (tree / ".mc5_agent.diff").write_text(agent_patch, encoding="utf-8")
    rc, out = sh(["git", "apply", "--whitespace=nowarn", ".mc5_agent.diff"], cwd=tree)
    if rc != 0:
        return False, "agent patch does not apply: " + out[-200:]
    tp = task["test_patch"]
    (tree / ".mc5_tests.diff").write_text(tp if tp.endswith("\n") else tp + "\n", encoding="utf-8")
    rc, out = sh(["git", "apply", "--whitespace=nowarn", ".mc5_tests.diff"], cwd=tree)
    if rc != 0:
        # Some task sets ship diffs git calls corrupt (no final newline, loose hunk counts);
        # patch(1) is more forgiving.
        rc, out2 = sh(["patch", "-p1", "-f", "--no-backup-if-mismatch", "-i", ".mc5_tests.diff"], cwd=tree)
        if rc != 0:
            return False, "test patch does not apply: " + out[-200:] + " | patch: " + out2[-200:]
    tests = task.get("test_files") or patch_files(task["test_patch"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(x for x in (str(tree), str(tree / 'src'), os.environ.get("MC5_TEST_PYTHONPATH")) if x), "PYTHONDONTWRITEBYTECODE": "1",
           "COLUMNS": "120", "TERM": "dumb"}
    env.pop("FORCE_COLOR", None)
    rc, out = sh([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                  *tests], cwd=tree, timeout=test_timeout, env=env)
    return rc == 0, out[-300:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--repos", help="directory of full git clones named requests/ httpx/ rich/ fastapi/")
    ap.add_argument("--snapshots", help="directory of <instance_id>.tgz (the Kaggle dataset layout)")
    ap.add_argument("--mode", choices=["localize", "oracle", "llm"], default="localize")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=120)
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--ids-from", help="only run tasks that PASSED in this earlier report (e.g. an oracle "
                                       "run, to keep tasks this environment can actually grade)")
    args = ap.parse_args()

    from mc5 import localize, paths, repo

    tasks = [json.loads(ln) for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    for t in tasks:
        t["patch"] = with_git_headers(t.get("patch") or "")
        t["test_patch"] = with_git_headers(t.get("test_patch") or "")
    if args.ids_from:
        keep = {r["id"] for r in json.loads(Path(args.ids_from).read_text())["rows"] if r.get("pass")}
        tasks = [t for t in tasks if t["instance_id"] in keep]
    if args.limit:
        tasks = tasks[:args.limit]
    repos = Path(args.repos).resolve() if args.repos else None
    snaps = Path(args.snapshots).resolve() if args.snapshots else None
    work = Path(tempfile.mkdtemp(prefix="mc5-eval-"))
    os.environ.setdefault("MC5_WORK_DIR", str(work / "work"))
    paths.WORK_DIR = work / "work"
    paths.LOG_FILE = paths.WORK_DIR / "app.log"

    rows = []
    log_seen = 0
    for t in tasks:
        tree = materialise(t, repos, snaps, work)
        if tree is None:
            print(f"  {t['instance_id']:22s} SKIP (no snapshot)")
            continue
        row = {"id": t["instance_id"]}
        try:
            gold_files = [f for f in patch_files(t["patch"])]
            issue = t["problem_statement"] + "\n" + (t.get("hints_text") or "")
            t0 = time.time()
            if args.mode == "localize":
                ranked = localize.rank_all(tree, issue)
                files = localize.rank_files(ranked[:14])
                chunks = localize.focus(ranked, repo.CONTEXT_CHARS)
                row["secs"] = round(time.time() - t0, 2)
                ranks = [files.index(g) + 1 if g in files else 99 for g in gold_files]
                row["file_rank"] = max(ranks) if ranks else 99          # ALL gold files must be found
                hunks = patch_hunks(t["patch"])
                hit = 0
                for f, a, b in hunks:
                    if any(c.path == f and c.start <= b and a <= c.end for c in chunks):
                        hit += 1
                row["hunks"] = len(hunks)
                row["hunks_in_context"] = hit
                print(f"  {t['instance_id']:22s} file_rank={row['file_rank']:<3} hunks {hit}/{len(hunks)}  {row['secs']}s  gold={gold_files}")
            else:
                if args.mode == "oracle":
                    llm = OracleLLM(t["patch"])
                else:
                    sys.path.insert(0, str(ROOT / "app"))
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("mc5_app", ROOT / "app" / "app.py")
                    app = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(app)
                    app.start_server()
                    llm = app.RemoteLLM(time.time() + args.deadline, reserve=3.0)
                res = repo.repair_repo(tree, t["problem_statement"], t.get("hints_text") or "", llm, args.deadline)
                if os.environ.get("MC5_EVAL_DEBUG") and not res.patch.strip():
                    print("    [debug] gold patch head: " + repr(t["patch"][:300]))
                    print(f"    [debug] localized={res.localized[:5]} label={res.label}")
                    try:
                        print("    [debug] log tail: " + " | ".join(paths.LOG_FILE.read_text().splitlines()[-6:])[:900])
                    except OSError as e:
                        print(f"    [debug] no log: {e}")
                row["secs"] = round(time.time() - t0, 1)
                row["label"] = res.label
                ok, why = grade(t, tree, res.patch, 300)
                row["pass"] = ok
                row["why"] = "" if ok else why[-600:]
                row["kind"] = "resolved" if ok else diagnose(t["patch"], res.patch, res.localized)
                print(f"  {t['instance_id']:22s} {'PASS' if ok else 'FAIL'}  {row['label']:28s} {row['secs']}s  "
                      f"{row['kind']:22s} {row['why'][-90:]!r}")
                if args.mode == "llm":
                    row["agent_patch"], row["gold_patch"] = res.patch, t["patch"]
                    try:
                        lines = paths.LOG_FILE.read_text(errors="replace").splitlines()
                        row["log"] = lines[log_seen:]
                        log_seen = len(lines)
                    except OSError:
                        row["log"] = []
        finally:
            release(t, repos, tree)
        rows.append(row)

    n = len(rows) or 1
    print()
    if args.mode == "localize":
        for k in (1, 3, 5, 10):
            print(f"  all gold files in top {k:<2}: {sum(r['file_rank'] <= k for r in rows)}/{len(rows)}")
        th, hh = sum(r["hunks"] for r in rows), sum(r["hunks_in_context"] for r in rows)
        print(f"  gold hunks inside the context shown to the model: {hh}/{th}")
        full = sum(r["hunks"] == r["hunks_in_context"] for r in rows)
        print(f"  tasks with EVERY gold hunk in context: {full}/{len(rows)}")
    else:
        print(f"  resolved: {sum(bool(r.get('pass')) for r in rows)}/{len(rows)}  ({args.mode})")
        kinds: dict[str, int] = {}
        for r in rows:
            k = re.sub(r"-\d+-of-\d+-hunks", "", r.get("kind", "?"))
            kinds[k] = kinds.get(k, 0) + 1
        print("  breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
        print(f"  seconds per task: median {sorted(r['secs'] for r in rows)[len(rows) // 2] if rows else 0}, "
              f"total {sum(r['secs'] for r in rows):.0f}")
        if args.report and args.mode == "llm":
            write_details(rows, args.report + ".details.txt")
    if args.report:
        Path(args.report).write_text(json.dumps({"mode": args.mode, "rows": rows}, indent=1))
    if args.report:
        try:
            shutil.copyfile(paths.LOG_FILE, args.report + ".log")
        except OSError:
            pass
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
