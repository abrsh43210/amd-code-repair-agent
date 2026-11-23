#!/usr/bin/env python3
"""AMD AI League - Match 5 ("Code Repair"). Practice entry.

The official Match 5 I/O contract is NOT published yet. Section 2 of the shared
rules PDF fixes everything that is not task-specific: the harness execs this
script once per test item inside an already-running container, passes a path, and
reads /app/output/<stem>_output.json. So this entry point accepts every plausible
flag name for "here is the task" and maps them all to one pipeline:

    python3 /app/app.py --warmup                             # load the model once
    python3 /app/app.py --input-task /app/input/task_01.json
    python3 /app/app.py --input /app/input/task_01.json       # same thing
    python3 /app/app.py --input-dir /app/input                # every task file in it
    python3 /app/app.py --code buggy.py --tests test_buggy.py # files on disk
    python3 /app/app.py /app/input/task_01.json               # bare positional

Output, one file per task, written whole (atomic rename):
/app/output/<task-id>_output.json
    {"code": "<whole fixed file>", "patch": "<unified diff>",
     "language": "python", "tests_pass": true, "confidence": 0.95}

Nothing is printed to stdout unless --stdout is given; logs go to /tmp/mc5/app.log.
(An earlier entry scored OUTPUT_MALFORMED purely because of stdout noise.)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE / "vendor"))   # pip deps from the app layer; base-image packages win

from mc5 import paths  # noqa: E402

SERVER = HERE / "mc5" / "server.py"
STARTUP_DEADLINE_S = 540                # the shared rules give 10 minutes for startup

TASK_SUFFIXES = (".json", ".jsonl")


# --------------------------------------------------------------------------- model server

def _request(req: dict, timeout: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(paths.SOCKET))
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    finally:
        s.close()


def server_alive() -> bool:
    try:
        return bool(_request({"op": "ping"}, timeout=2).get("ok"))
    except (OSError, ValueError):
        return False


def _starting() -> bool:
    try:
        return time.time() - (paths.WORK_DIR / "server.started").stat().st_mtime < STARTUP_DEADLINE_S
    except OSError:
        return False


def start_server() -> None:
    if server_alive() or _starting():
        return
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (paths.WORK_DIR / "server.started").write_text(str(time.time()))
    subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=open(paths.WORK_DIR / "server.stderr", "ab"),
        start_new_session=True, close_fds=True, cwd=str(HERE),
    )


class RemoteLLM:
    """Client for the resident server. Fails fast if the model is not ready; the
    pipeline then keeps whatever its model-free search found."""

    def __init__(self, deadline: float, reserve: float = 1.5):
        self.deadline = deadline
        self.reserve = reserve              # seconds kept back for the work after a reply

    def chat(self, system: str, text: str, max_new_tokens: int = 120, max_time: float | None = None) -> str:
        while not server_alive():
            if time.time() > self.deadline - 4:
                raise RuntimeError("model server not ready")
            time.sleep(0.5)
        resp = _request({"op": "chat", "system": system, "text": text,
                         "max_new_tokens": max_new_tokens,
                         "deadline": min(self.deadline - self.reserve,
                                         time.time() + max_time if max_time else float("inf"))},
                        timeout=max(2.0, self.deadline - time.time()))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "server error"))
        return str(resp.get("text", ""))


def warmup() -> None:
    t0 = time.time()
    start_server()
    while time.time() - t0 < STARTUP_DEADLINE_S:
        if server_alive():
            paths.log(f"model ready after {time.time() - t0:.0f}s")
            return
        time.sleep(1.0)
    paths.log("model server did not come up in time")


# --------------------------------------------------------------------------- work

def solve(task, out_dir: Path, to_stdout: bool, no_llm: bool) -> None:
    from mc5 import repair, task as task_mod
    t0 = time.time()
    if task.repo_dir is not None:                       # repo-level task: issue + repository -> patch
        from mc5 import repo
        try:
            llm = None if no_llm else RemoteLLM(t0 + repo.REPO_DEADLINE_S, reserve=3.0)
            result = repo.repair_repo(Path(task.repo_dir), task.problem, task.hints, llm)
        except Exception as e:  # noqa: BLE001 - always write a valid file
            paths.log(f"{task.stem}: repo repair failed {type(e).__name__}: {e}")
            result = repo.Result()
        task_mod.write_repo_output(out_dir, task, result, to_stdout)
        paths.log(f"{task.stem}: {time.time() - t0:.1f}s repo label={result.label} "
                  f"patch={len(result.patch)}B top={result.localized[:3]}")
        return
    try:
        llm = None if no_llm else RemoteLLM(t0 + paths.DEADLINE_S)
        fixed, conf, verified, label = repair.repair(task, paths.DEADLINE_S, llm)
    except Exception as e:  # noqa: BLE001 - always write a valid file
        paths.log(f"{task.stem}: failed {type(e).__name__}: {e}")
        fixed, conf, verified, label = task.code, 0.0, False, "error"
    task_mod.write_output(out_dir, task, fixed, conf, verified, to_stdout)
    paths.log(f"{task.stem}: {time.time() - t0:.1f}s label={label} verified={verified}")


def tasks_from_dir(d: Path):
    from mc5 import task as task_mod
    out = []
    for p in sorted(d.iterdir() if d.is_dir() else []):
        if p.is_file() and p.suffix.lower() in TASK_SUFFIXES:
            out.extend(task_mod.load(p))
    return out


def main() -> int:
    from mc5 import task as task_mod

    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--warmup", "--index", "--setup", "--init", dest="warm",
                    nargs="?", const=True, default=None)
    ap.add_argument("--input", "--input-task", "--input_task", "--input-file", "--input-json",
                    "--task", "--task-file", "--problem", "--bug", "-i", dest="input")
    ap.add_argument("--input-dir", "--input_dir", "--tasks-dir", dest="input_dir")
    ap.add_argument("--code", "--source", "--file", "--buggy-file", dest="code_file")
    ap.add_argument("--tests", "--test-file", "--test", dest="test_file")
    ap.add_argument("--error", "--error-file", "--traceback", dest="error_file")
    ap.add_argument("--repo", "--repo-dir", "--repo_dir", "--repo-path", "--workspace", "--project",
                    dest="repo_dir", help="repo-level task: path to the repository")
    ap.add_argument("--issue", "--problem-statement", "--issue-file", "--issue-text", dest="issue",
                    help="repo-level task: the issue text, or a file containing it")
    ap.add_argument("--task-id", "--task_id", "--id", "--instance-id", "--problem-id", dest="tid")
    ap.add_argument("--language", "--lang", dest="language")
    ap.add_argument("--output-dir", "--output", "--out", "-o", dest="out")
    ap.add_argument("--input-image", dest="unused_image",
                    help="accepted and ignored (Match 2 shape)")
    ap.add_argument("--stdout", action="store_true", help="debug: also print each result")
    ap.add_argument("--no-llm", action="store_true", help="debug: model-free search only")
    try:
        args, rest = ap.parse_known_args()
    except SystemExit:
        # argparse rejected the command line (a flag we know, used in a shape we do not).
        # Never leave the harness without an output file: treat every existing path on
        # the command line as a task and answer it.
        paths.log(f"argparse rejected {sys.argv[1:]!r}; falling back to bare paths")
        args, _ = ap.parse_known_args([])
        rest = [a for a in sys.argv[1:] if not a.startswith("-")]

    out_dir = paths.OUTPUT_DIR
    if args.out:
        out_dir = Path(args.out).parent if args.out.endswith(".json") else Path(args.out)

    if not args.no_llm:
        start_server()                     # idempotent; weights load in the background
    if args.warm is not None and not (args.input or args.input_dir or args.code_file or args.repo_dir or rest):
        if not args.no_llm:
            warmup()
        return 0

    jobs = []
    try:
        if args.input:
            jobs += task_mod.load(Path(args.input))
        if args.input_dir:
            jobs += tasks_from_dir(Path(args.input_dir))
        if args.code_file:
            p = Path(args.code_file)
            jobs.append(task_mod.Task(
                stem=args.tid or p.stem,
                code=p.read_text(encoding="utf-8", errors="replace"),
                tests=(Path(args.test_file).read_text(encoding="utf-8", errors="replace")
                       if args.test_file and Path(args.test_file).is_file() else ""),
                error=(Path(args.error_file).read_text(encoding="utf-8", errors="replace")
                       if args.error_file and Path(args.error_file).is_file() else ""),
                language=args.language or "", file_name=p.name,
                test_file_name=Path(args.test_file).name if args.test_file else "",
            ))
        if args.repo_dir and Path(args.repo_dir).is_dir():
            issue = args.issue or ""
            if issue and len(issue) < 400 and Path(issue).is_file():
                issue = Path(issue).read_text(encoding="utf-8", errors="replace")
            if jobs and all(j.repo_dir is None and not j.code for j in jobs):
                for j in jobs:                          # task file gave the issue, flag gave the repo
                    j.repo_dir = Path(args.repo_dir)
            else:
                jobs.append(task_mod.Task(stem=args.tid or Path(args.repo_dir).name, code="",
                                          repo_dir=Path(args.repo_dir), problem=issue, language="python"))
        for r in rest:                      # bare positional path
            if not r.startswith("-") and Path(r).is_file():
                jobs += task_mod.load(Path(r))
    except Exception as e:  # noqa: BLE001
        paths.log(f"bad input: {type(e).__name__}: {e}")

    if not jobs and paths.INPUT_DIR.is_dir():
        jobs = tasks_from_dir(paths.INPUT_DIR)      # last resort: repair everything given
    if not jobs:
        paths.log("no task found in arguments")
        return 0

    for t in jobs:
        if args.tid and len(jobs) == 1:
            t.stem = args.tid
        if args.language:
            t.language = args.language
        solve(t, out_dir, args.stdout, args.no_llm)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException as e:  # noqa: BLE001 - a crash must leave a trace, just never on stdout
        import traceback
        paths.log("FATAL " + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.stderr.write(f"mc5: {type(e).__name__}: {e}\n")
    finally:
        # Hard exit: a stray verifier thread past its deadline must not keep the
        # process alive after the output files are written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if isinstance(rc, int) else 0)
