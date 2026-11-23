"""Repo-level repair: issue text + a repository in, a unified diff out.

This is the SWE-bench shape (and the shape of Kaggle's Gemma 4 Developer Agent
tasks): the tests that grade the fix are hidden, so the single-file trick of
"run the task's tests on every candidate" is not available. What we can still
verify, in decreasing strength:

  1. a reproduction test the model writes from the issue, kept only if it FAILS
     on the untouched repo (then a candidate must make it pass)
  2. no regression in the existing tests nearest to the edited code
  3. the edited files still compile and import

The repo's dependencies may not be installed where we run. Every test-based check
is therefore best effort: if the baseline run cannot even collect, that check is
dropped rather than trusted.
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import localize, paths
from .verify import PYTEST_VERDICT

# Section 2 of the shared rules: 30 s per item, and one overrun zeroes the whole run.
# Until the Match 5 spec says repo tasks get longer, stay inside that. Practice runs
# raise both through the environment.
REPO_DEADLINE_S = float(os.environ.get("MC5_REPO_DEADLINE_S", "25"))
CONTEXT_CHARS = int(os.environ.get("MC5_REPO_CONTEXT_CHARS", "28000"))
PICK_SYMBOLS = os.environ.get("MC5_REPO_PICK_SYMBOLS", "1") == "1"
# Reproduction-test step. On 40 FastAPI task runs it never produced a usable test (it
# passed or errored on the untouched repo every time), so it can be switched off.
WRITE_REPRO = os.environ.get("MC5_REPO_REPRO", "1") == "1"
TEST_TIMEOUT_S = float(os.environ.get("MC5_REPO_TEST_TIMEOUT_S", "8"))
MAX_ATTEMPTS = int(os.environ.get("MC5_REPO_ATTEMPTS", "3"))
REWRITE = os.environ.get("MC5_REPO_REWRITE", "1") == "1"
# seconds that must be left for the symbol-pick call to run (it takes 2-5 s on the target GPU)
PICK_MIN_S = float(os.environ.get("MC5_REPO_PICK_MIN_S", "18"))
# most seconds the model-free search of the repo may take before the first model call
LOCALIZE_MAX_S = float(os.environ.get("MC5_REPO_LOCALIZE_MAX_S", "5"))
# most seconds the throwaway copy of the repo for a test run may take
OVERLAY_MAX_S = float(os.environ.get("MC5_REPO_OVERLAY_MAX_S", "3"))
REWRITE_MAX_LINES = int(os.environ.get("MC5_REPO_REWRITE_MAX_LINES", "80"))
DEBUG_REPLIES = os.environ.get("MC5_REPO_DEBUG_REPLIES") == "1"


# --------------------------------------------------------------------------- edits

_BLOCK = re.compile(
    r"(?:^|\n)[ \t]*([^\n`<>]*?\.py)[ \t]*\n(?:```[^\n]*\n)?[ \t]*<{5,}[ \t]*SEARCH[ \t]*\n(.*?)\n?[ \t]*={5,}[ \t]*\n(.*?)\n?[ \t]*>{5,}[ \t]*REPLACE",
    re.S)


def parse_edits(reply: str) -> list[tuple[str, str, str]]:
    """[(path, search, replace)] from SEARCH/REPLACE blocks."""
    out = []
    for m in _BLOCK.finditer(reply or ""):
        path = m.group(1).strip().strip("*#` ").lstrip("./")
        out.append((path, m.group(2), m.group(3)))
    return out


def _in_shown(text: str, pos: int, shown) -> bool:
    line = text.count("\n", 0, pos) + 1
    return any(a <= line <= b for a, b in shown)


def _fuzzy_block(t_lines: list[str], s_lines: list[str]) -> int | None:
    """Start line of the one window that is almost the SEARCH text (a model that retyped a
    line slightly wrong). None unless one window clearly wins."""
    want = [x.strip() for x in s_lines]
    n = len(want)
    if n < 2 or n > 60 or len(t_lines) < n:
        return None
    joined = "\n".join(want)
    anchors = {w for w in want if len(w) > 8}
    scored = []
    for k in range(len(t_lines) - n + 1):
        win = [x.strip() for x in t_lines[k:k + n]]
        if anchors and not anchors.intersection(win):
            continue                                      # cheap filter: share one exact line
        r = difflib.SequenceMatcher(None, joined, "\n".join(win), autojunk=False).ratio()
        if r >= 0.9:
            scored.append((r, k))
        if len(scored) > 50:
            return None
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.03:
        return None
    return scored[0][1]


def _find_block(text: str, search: str, after: int = 0, allow_repeat: bool = False,
                shown=None) -> tuple[int, int, str] | None:
    """Locate `search` in `text`. Returns (start, end, indent_fix) or None.
    Tries exact, then trailing-whitespace-insensitive, then indentation-insensitive.

    Text that occurs more than once is refused rather than guessed at, with one
    exception (`allow_repeat`): when the reply carries as many blocks with this same
    SEARCH text as the file has occurrences - a sync and an async twin, typically -
    they are matched in order, each after the previous edit."""
    if not search.strip():
        return None
    i = text.find(search)
    if i != -1:
        if text.find(search, i + 1) != -1:
            if not allow_repeat:
                # Ambiguous. The model only ever saw some parts of the file; if exactly one
                # occurrence lies in what it was shown, that is the one it meant.
                if shown:
                    places, j = [], text.find(search)
                    while j != -1:
                        places.append(j)
                        j = text.find(search, j + 1)
                    seen = [j for j in places if _in_shown(text, j, shown)]
                    if len(seen) == 1:
                        return seen[0], seen[0] + len(search), ""
                return None                               # refuse to guess
            j = text.find(search, after)
            return (j, j + len(search), "") if j != -1 else None
        return i, i + len(search), ""
    t_lines = text.split("\n")
    s_lines = search.split("\n")
    while s_lines and not s_lines[-1].strip():
        s_lines.pop()
    while s_lines and not s_lines[0].strip():
        s_lines.pop(0)
    if not s_lines:
        return None
    for norm in (str.rstrip, str.strip):
        want = [norm(x) for x in s_lines]
        hits = [k for k in range(len(t_lines) - len(want) + 1)
                if [norm(x) for x in t_lines[k:k + len(want)]] == want]
        if len(hits) > 1 and shown:
            seen = [k for k in hits if any(a <= k + 1 <= b for a, b in shown)]
            if len(seen) == 1:
                hits = seen
        if len(hits) == 0 and norm is str.strip:
            k = _fuzzy_block(t_lines, s_lines)
            if k is not None:
                hits = [k]
        if len(hits) == 1:
            k = hits[0]
            start = sum(len(x) + 1 for x in t_lines[:k])
            end = start + sum(len(x) + 1 for x in t_lines[k:k + len(want)]) - 1
            indent = ""
            if norm is str.strip:
                have = re.match(r"[ \t]*", t_lines[k]).group(0)
                given = re.match(r"[ \t]*", s_lines[0]).group(0)
                if have != given:
                    indent = f"{len(given)}>{have}"
            return start, end, indent
    # Blank lines left out (or added): the commonest slip of all. Compare only the lines
    # that carry code; the span runs from the first matched line to the last.
    want = [x.strip() for x in s_lines if x.strip()]
    if len(want) >= 2:
        idx = [i for i, x in enumerate(t_lines) if x.strip()]
        code = [t_lines[i].strip() for i in idx]
        hits = [k for k in range(len(code) - len(want) + 1) if code[k:k + len(want)] == want]
        if len(hits) > 1 and shown:
            seen = [k for k in hits if any(a <= idx[k] + 1 <= b for a, b in shown)]
            if len(seen) == 1:
                hits = seen
        if len(hits) == 1:
            a, b = idx[hits[0]], idx[hits[0] + len(want) - 1]
            start = sum(len(x) + 1 for x in t_lines[:a])
            end = start + sum(len(x) + 1 for x in t_lines[a:b + 1]) - 1
            have = re.match(r"[ \t]*", t_lines[a]).group(0)
            given = re.match(r"[ \t]*", next(x for x in s_lines if x.strip())).group(0)
            return start, end, (f"{len(given)}>{have}" if have != given else "")
    return None


def _reindent(replace: str, fix: str) -> str:
    if not fix:
        return replace
    given_n, have = fix.split(">", 1)
    given_n = int(given_n)
    out = []
    for ln in replace.split("\n"):
        if not ln.strip():
            out.append(ln)
            continue
        cur = len(ln) - len(ln.lstrip(" \t"))
        extra = ln[given_n:cur] if cur >= given_n else ""
        out.append(have + extra + ln.lstrip(" \t"))
    return "\n".join(out)


_ELLIPSIS = re.compile(r"^[ \t]*(?:#[ \t]*)?(?:\.\.\.|…)[ \t]*(?:#.*)?$|^[ \t]*#[ \t]*(?:\.\.\.|…|rest of|existing code|unchanged).*$", re.I)


def _best_window(t_lines: list[str], s_lines: list[str], shown=None, same_start: float = 0.0) -> tuple[float, int]:
    """(ratio, start line index) of the window of the file most like the SEARCH text.
    Looks only inside `shown` line ranges when given. (0.0, -1) if nothing is comparable."""
    want = [x.strip() for x in s_lines if x.strip()]
    n = len(s_lines)
    if not want or n > 80 or len(t_lines) < n:
        return 0.0, -1
    joined = "\n".join(x.strip() for x in s_lines)
    words = set(re.findall(r"[A-Za-z_]\w{2,}", joined))
    best = (0.0, -1)
    starts = range(len(t_lines) - n + 1)
    if shown:
        starts = [k for k in starts if any(a - 1 <= k and k + n <= b + 1 for a, b in shown)] or starts
    checked = 0
    for k in starts:
        win = [x.strip() for x in t_lines[k:k + n]]
        if words and len(words.intersection(re.findall(r"[A-Za-z_]\w{2,}", " ".join(win)))) < max(1, len(words) // 3):
            continue
        checked += 1
        if checked > 400:
            break
        r = difflib.SequenceMatcher(None, joined, "\n".join(win), autojunk=False).ratio()
        if same_start and win[0] == want[0]:
            r = min(1.0, r + same_start)                  # starts on the very same line
        if r > best[0]:
            best = (r, k)
    return best


def _loose_block(text: str, search: str, shown=None) -> tuple[int, int, str] | None:
    """Last resort for a SEARCH the model retyped badly: the one clearly best window among
    the lines it was shown (>= 0.72 alike and well ahead of any other place)."""
    t_lines = text.split("\n")
    s_lines = [x for x in search.split("\n")]
    while s_lines and not s_lines[-1].strip():
        s_lines.pop()
    while s_lines and not s_lines[0].strip():
        s_lines.pop(0)
    n = len(s_lines)
    if n < 2:
        return None
    r, k = _best_window(t_lines, s_lines, shown)
    if k < 0 or r < 0.72:
        return None
    # the first and last lines must be the real ones, or the edit could cut a statement in half
    if t_lines[k].strip() != s_lines[0].strip() or t_lines[k + n - 1].strip() != s_lines[-1].strip():
        return None
    joined = "\n".join(x.strip() for x in s_lines)
    for k2 in range(len(t_lines) - n + 1):
        if abs(k2 - k) < n:
            continue
        if t_lines[k2].strip() != s_lines[0].strip():
            continue
        r2 = difflib.SequenceMatcher(None, joined, "\n".join(x.strip() for x in t_lines[k2:k2 + n]),
                                     autojunk=False).ratio()
        if r2 > r - 0.08:
            return None                                   # a rival place: refuse to guess
    start = sum(len(x) + 1 for x in t_lines[:k])
    end = start + sum(len(x) + 1 for x in t_lines[k:k + n]) - 1
    have = re.match(r"[ \t]*", t_lines[k]).group(0)
    given = re.match(r"[ \t]*", s_lines[0]).group(0)
    return start, end, (f"{len(given)}>{have}" if have != given else "")


def _elided(text: str, search: str, replace: str, shown=None) -> tuple[int, int, str] | None:
    """SEARCH written as `first lines / ... / last lines`. Returns (start, end, new_body):
    the span from the head to the tail, and the replacement with its own `...` (if it has
    one) filled in from the file."""
    s_lines = search.split("\n")
    cut = [i for i, ln in enumerate(s_lines) if _ELLIPSIS.match(ln)]
    if not cut:
        return None
    head = "\n".join(s_lines[:cut[0]]).strip("\n")
    tail = "\n".join(s_lines[cut[-1] + 1:]).strip("\n")
    if not head.strip() or not tail.strip():
        return None
    h = _find_block(text, head, shown=shown)
    if h is None:
        return None
    rest = text[h[1]:]
    t = _find_block(rest, tail)
    if t is None:
        i = rest.find(tail)
        if i == -1:
            return None
        t = (i, i + len(tail), "")                        # first one after the head
    start, end = h[0], h[1] + t[1]
    if text.count("\n", start, end) > 200:
        return None
    middle = text[h[1]:h[1] + t[0]]
    r_lines = replace.split("\n")
    rcut = [i for i, ln in enumerate(r_lines) if _ELLIPSIS.match(ln)]
    if len(rcut) == 1 and len(cut) == 1:
        a = "\n".join(r_lines[:rcut[0]]).rstrip("\n")
        b = "\n".join(r_lines[rcut[0] + 1:]).lstrip("\n")
        return start, end, _reindent(a, h[2]) + middle + _reindent(b, h[2])
    if rcut:
        return None                                       # several gaps: cannot line them up
    return start, end, _reindent(replace, h[2])


def nearest_lines(text: str, search: str, shown=None, pad: int = 1) -> str:
    """The real lines of the file closest to a SEARCH that did not match (for feedback)."""
    t_lines = text.split("\n")
    s_lines = [x for x in search.split("\n") if x.strip() and not _ELLIPSIS.match(x)]
    if not s_lines:
        return ""
    first = s_lines[0].strip()
    at = [i for i, x in enumerate(t_lines) if x.strip() == first] if len(first) > 6 else []
    if len(at) == 1:                                      # its first line is real: quote from there
        return "\n".join(t_lines[at[0]:min(len(t_lines), at[0] + len(s_lines[:40]) + 4)])
    r, k = _best_window(t_lines, s_lines[:40], shown, same_start=0.25)
    if k < 0 or r < 0.4:
        return ""
    a, b = max(0, k - pad), min(len(t_lines), k + len(s_lines[:40]) + pad)
    return "\n".join(t_lines[a:b])


def apply_edits(root: Path, edits: list[tuple[str, str, str]],
                shown: dict[str, list[tuple[int, int]]] | None = None) -> tuple[dict[str, str], list[str]]:
    """Returns ({path: new_text}, problems). Nothing is written to disk."""
    new: dict[str, str] = {}
    problems: list[str] = []
    cursor: dict[str, int] = {}
    known = None
    for path, search, replace in edits:
        target = root / path
        if not target.is_file():
            if known is None:
                known = localize.source_files(root, include_tests=True)
            cands = [k for k in known if k.endswith("/" + path) or k.rsplit("/", 1)[-1] == path.rsplit("/", 1)[-1]]
            if len(cands) == 1:
                path, target = cands[0], root / cands[0]
            elif not search.strip() and path.endswith(".py"):
                new[path] = replace if replace.endswith("\n") else replace + "\n"      # new file
                continue
            else:
                problems.append(f"{path}: no such file")
                continue
        text = new.get(path)
        if text is None:
            text = target.read_text(encoding="utf-8", errors="replace")
        original = target.read_text(encoding="utf-8", errors="replace")
        # Blocks aimed at this file. When their number equals the number of places this
        # SEARCH text occurs, they are the twins (sync and async client, say) in order.
        same_file = sum(1 for p2, _, _ in edits if p2.rsplit("/", 1)[-1] == path.rsplit("/", 1)[-1])
        repeat_ok = same_file > 1 and original.count(search) == same_file
        # Line numbers of `shown` refer to the untouched file, so only use them on it.
        hit = _find_block(text, search, after=cursor.get(path, 0), allow_repeat=repeat_ok,
                          shown=(shown or {}).get(path) if text == original else None)
        body = None
        if hit is None and text.count(search) <= 1:
            how = ""
            sh = (shown or {}).get(path) if text == original else None
            el = _elided(text, search, replace, sh)
            if el is not None:
                hit, body, how = (el[0], el[1], ""), el[2], "elided"
            if hit is None:
                # Right text, wrong file name: the one other shown file that holds it.
                others = []
                for p2 in (shown or {}):
                    if p2 == path or p2 in new or not (root / p2).is_file():
                        continue
                    t2 = (root / p2).read_text(encoding="utf-8", errors="replace")
                    h2 = _find_block(t2, search, shown=shown.get(p2))
                    if h2 is not None:
                        others.append((p2, t2, h2))
                if len(others) == 1:
                    path, text, hit = others[0]
                    how = "other-file"
            if hit is None:
                hit = _loose_block(text, search, sh)
                how = "loose" if hit else ""
            if hit is not None:
                paths.log(f"repo: SEARCH matched by fallback '{how}' in {path}")
        if hit is None:
            n = text.count(search)
            if n > 1:
                problems.append(f"{path}: SEARCH text occurs {n} times; include more surrounding lines")
            else:
                near = nearest_lines(text, search, (shown or {}).get(path) if text == original else None)
                problems.append(f"{path}: SEARCH text was not found in the file" +
                                (f". The closest real lines in {path} are:\n```\n{near[:1200]}\n```\n"
                                 "Copy SEARCH from these exactly" if near else ""))
                first = next((x.strip() for x in search.split("\n") if x.strip()), "")
                paths.log(f"repo: SEARCH miss in {path}: {len(search.splitlines())} lines, starts {first[:80]!r}, "
                          f"nearest {'found' if near else 'none'}")
            continue
        s, e, fix = hit
        if body is None:
            body = _reindent(replace, fix)
        new[path] = text[:s] + body + text[e:]
        cursor[path] = s + len(body)
    return {p: t for p, t in new.items()
            if not (root / p).is_file() or (root / p).read_text(encoding="utf-8", errors="replace") != t}, problems


def make_patch(root: Path, changed: dict[str, str]) -> str:
    """git-style unified diff, applicable with `git apply` / `patch -p1`."""
    parts = []
    for path in sorted(changed):
        old_file = root / path
        old = old_file.read_text(encoding="utf-8", errors="replace") if old_file.is_file() else ""
        new = changed[path]
        a = f"a/{path}" if old_file.is_file() else "/dev/null"
        body = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                            fromfile=a, tofile=f"b/{path}", n=3))
        if not body:
            continue
        # difflib omits the "\ No newline at end of file" marker; normalise both ends instead.
        fixed = []
        for ln in body.split("\n"):
            fixed.append(ln)
        body = "\n".join(fixed)
        if not body.endswith("\n"):
            body += "\n"
        head = f"diff --git a/{path} b/{path}\n"
        if not old_file.is_file():
            head += "new file mode 100644\n"
        parts.append(head + body)
    return "".join(parts)


# --------------------------------------------------------------------------- checks

def compiles(changed: dict[str, str]) -> tuple[bool, str]:
    for path, text in changed.items():
        try:
            compile(text, path, "exec")
        except SyntaxError as e:
            return False, f"{path}:{e.lineno}: {e.msg}"
    return True, ""


class OverlayTimeout(Exception):
    """The throwaway copy of the repo could not be made inside its time budget."""


def _overlay(root: Path, changed: dict[str, str], budget_s: float | None = None) -> Path:
    """A throwaway copy of the repo with the candidate applied (hard links where possible).

    When the repo and WORK_DIR are on different filesystems (a mounted input volume and
    /tmp, say) hard links fail and every file is really copied, which on a large repo
    takes many seconds. `budget_s` bounds that: past it the partial copy is removed and
    OverlayTimeout is raised, which callers treat as "no test verdict"."""
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="mc5-repo-", dir=str(paths.WORK_DIR)))
    shutil.rmtree(dst)
    t_end = None if budget_s is None else time.time() + max(0.0, budget_s)
    state = {"link": True, "n": 0}

    def put(src, dst_file):
        state["n"] += 1
        if t_end is not None and state["n"] % 50 == 0 and time.time() > t_end:
            raise OverlayTimeout(f"{state['n']} files in {budget_s:.1f}s")
        if state["link"]:
            try:
                os.link(src, dst_file)
                return
            except OSError:
                state["link"] = False                     # other filesystem: stop trying
        shutil.copyfile(src, dst_file)

    try:
        shutil.copytree(root, dst, symlinks=True, copy_function=put,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".tox", ".venv",
                                                      "node_modules"))
        if t_end is not None and time.time() > t_end:
            raise OverlayTimeout(f"{state['n']} files in more than {budget_s:.1f}s")
    except OverlayTimeout:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    for path, text in changed.items():
        f = dst / path
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            f.unlink()                                    # break the hard link before writing
        f.write_text(text, encoding="utf-8")
    return dst


def run_pytest(tree: Path, targets: list[str], timeout: float) -> tuple[str, str]:
    """('pass' | 'fail' | 'error', tail of output). 'error' = could not run at all."""
    env = {**os.environ, "PYTHONPATH": paths.test_pythonpath(tree, tree / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "COLUMNS": "120", "TERM": "dumb"}
    env.pop("FORCE_COLOR", None)
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "-o", "addopts=", *targets], cwd=str(tree), env=env, capture_output=True,
                           text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return "error", "timeout"
    except OSError as e:
        return "error", str(e)
    tail = (p.stdout + p.stderr)[-2500:]
    if p.returncode == 1 and not PYTEST_VERDICT.search(tail):
        return "error", "pytest did not run: " + tail[-300:]   # e.g. No module named pytest
    if p.returncode == 0:
        return "pass", tail
    if p.returncode == 1:
        return "fail", tail
    return "error", tail                                   # 2-5: collection, usage, no tests, internal


def nearby_tests(root: Path, files: list[str], symbols: list[str], limit: int = 3) -> list[str]:
    """Existing test files most likely to exercise the edited code."""
    mods = {Path(f).stem for f in files}
    names = {s.rsplit(".", 1)[-1] for s in symbols if s and not s.startswith("<")}
    scored = []
    for rel in localize.source_files(root, include_tests=True):
        if not localize.is_test_path(rel) or rel.endswith("conftest.py"):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        s = sum(4 for m in mods if f"test_{m}" in rel)
        s += sum(min(text.count(nm), 5) for nm in names)
        s += sum(1 for m in mods if re.search(rf"\b{re.escape(m)}\b", text))
        if s:
            scored.append((s, rel))
    return [rel for _, rel in sorted(scored, reverse=True)[:limit]]


# --------------------------------------------------------------------------- prompts

SYSTEM = (
    "You are a senior Python engineer fixing an issue in an existing repository. "
    "You make the smallest change that resolves the issue and keep the project's style. "
    "You answer only with SEARCH/REPLACE edit blocks."
)

EDIT_FORMAT = """Reply with one or more edit blocks, exactly in this form and nothing else:

path/to/file.py
<<<<<<< SEARCH
exact lines copied from the file
=======
the lines that replace them
>>>>>>> REPLACE

Rules: SEARCH must be copied character for character from the code shown, keep indentation. Keep SEARCH short: only the lines that change plus one or two unchanged lines around them so it is unique. Never abbreviate with "...". Do not edit tests. Do not add explanations."""


def build_context(chunks: list[localize.Chunk], budget: int) -> str:
    out, used = [], 0
    for c in chunks:
        block = f"### {c.path} (lines {c.start}-{c.end}) {c.name}\n```python\n{c.text}\n```\n"
        if used + len(block) > budget * 1.1 and out:
            continue
        out.append(block)
        used += len(block)
    return "\n".join(out)


PICK_SYSTEM = ("You locate the code that must change to resolve an issue. "
               "You answer only with a list of symbols, one per line.")


def pick_prompt(problem: str, skeletons: dict[str, str]) -> str:
    parts = [f"## Issue\n{problem.strip()[:4000]}", "## Candidate files (signatures only)"]
    for path, sk in skeletons.items():
        parts.append(f"### {path}\n```python\n{sk}\n```")
    parts.append("List up to 6 functions, methods or classes that most likely need to be read or edited to "
                 "fix the issue, most likely first, one per line, as `path/to/file.py::Name` or "
                 "`path/to/file.py::Class.method`. Nothing else.")
    return "\n\n".join(parts)


def parse_picks(reply: str, all_chunks: list[localize.Chunk]) -> list[localize.Chunk]:
    """Chunks named by the model. Unknown names are ignored; never trusted blindly."""
    by_file: dict[str, list[localize.Chunk]] = {}
    for c in all_chunks:
        by_file.setdefault(c.path, []).append(c)
    out: list[localize.Chunk] = []
    for m in re.finditer(r"([\w./-]+\.py)\s*(?:::|:|#)\s*`?([A-Za-z_][\w.]*)", reply or ""):
        path, name = m.group(1).lstrip("./"), m.group(2)
        cands = by_file.get(path) or next((v for k, v in by_file.items() if k.endswith("/" + path)), [])
        hit = ([c for c in cands if c.name == name]
               or [c for c in cands if c.name.rsplit(".", 1)[-1] == name.rsplit(".", 1)[-1]]
               or [c for c in cands if c.name == name.split(".", 1)[0]])
        for c in hit[:2]:
            if c not in out:
                out.append(c)
        if len(out) >= 8:
            break
    return out


def fix_prompt(problem: str, hints: str, context: str, feedback: str = "") -> str:
    parts = [f"## Issue\n{problem.strip()[:6000]}"]
    if hints.strip():
        parts.append(f"## Maintainer hints\n{hints.strip()[:1500]}")
    parts.append(f"## Most relevant code\n{context}")
    if feedback:
        parts.append(f"## Your previous attempt was rejected\n{feedback.strip()[:1500]}\nTry again with a different or corrected edit.")
    parts.append(EDIT_FORMAT)
    return "\n\n".join(parts)


REWRITE_SYSTEM = ("You are a senior Python engineer fixing an issue in an existing repository. "
                  "You answer only with a single Python code block.")


def rewrite_prompt(problem: str, hints: str, chunk: localize.Chunk) -> str:
    parts = [f"## Issue\n{problem.strip()[:5000]}"]
    if hints.strip():
        parts.append(f"## Maintainer hints\n{hints.strip()[:1000]}")
    parts.append(f"## Code to fix: `{chunk.name}` in {chunk.path}\n```python\n{chunk.text}\n```")
    parts.append("Write the complete corrected version of exactly this code (the whole definition, from its "
                 "first line to its last, same indentation), with the smallest change that resolves the issue. "
                 "Reply with one ```python block and nothing else.")
    return "\n\n".join(parts)


def rewrite_target(root: Path, chunks: list[localize.Chunk], failed: list[tuple[str, str, str]],
                   problem: str = "") -> localize.Chunk | None:
    """The function the model was trying to edit: among the functions and methods it was
    shown, the one sharing most words with its failed SEARCH text (the named file
    preferred); with no failed edit, the one sharing most words with the issue."""
    import ast
    cands: list[localize.Chunk] = []
    for path in dict.fromkeys(c.path for c in chunks):
        if localize.is_test_path(path):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except (OSError, SyntaxError, ValueError):
            continue
        lines = text.split("\n")
        ranges = [(c.start, c.end) for c in chunks if c.path == path]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = min([node.lineno] + [d.lineno for d in node.decorator_list])
            b = node.end_lineno or node.lineno
            if b - a >= REWRITE_MAX_LINES or not any(x <= a and b <= y for x, y in ranges):
                continue
            cands.append(localize.Chunk(path, node.name, a, b, "\n".join(lines[a - 1:b]), "function"))
    if not cands:
        return None
    word = re.compile(r"[A-Za-z_]\w{2,}")
    best, best_score = None, 0.0
    for path, search, replace in failed or [("", problem, "")]:
        words = set(word.findall(search))
        if not words:
            continue
        for c in cands:
            sc = len(words & set(word.findall(c.text))) / len(words)
            if c.name in words:
                sc += 0.3
            if path and (c.path == path or c.path.endswith("/" + path)):
                sc += 0.25
            sc -= 0.0005 * (c.end - c.start)              # ties: the smaller definition
            if sc > best_score:
                best, best_score = c, sc
    return best


def apply_rewrite(root: Path, chunk: localize.Chunk, reply: str) -> dict[str, str]:
    """Replace the chunk's lines with the code block of the reply. {} if unusable."""
    blocks = _FENCE.findall(reply or "")
    code = max(blocks, key=len) if blocks else ""
    code = code.strip("\n")
    if not code.strip():
        return {}
    target = root / chunk.path
    lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
    old = lines[chunk.start - 1:chunk.end]
    if not old:
        return {}
    new_lines = code.split("\n")
    # must still be the same definition (decorators allowed above it)
    def head(ls):
        return next((x.strip() for x in ls if x.strip() and not x.strip().startswith("@")), "")
    m_old = re.match(r"(?:async\s+)?(?:def|class)\s+(\w+)", head(old))
    m_new = re.match(r"(?:async\s+)?(?:def|class)\s+(\w+)", head(new_lines))
    if not m_old or not m_new or m_old.group(1) != m_new.group(1):
        return {}
    have = re.match(r"[ \t]*", next(x for x in old if x.strip())).group(0)
    given = re.match(r"[ \t]*", next(x for x in new_lines if x.strip())).group(0)
    if have != given:
        new_lines = _reindent("\n".join(new_lines), f"{len(given)}>{have}").split("\n")
    if [x.rstrip() for x in new_lines] == [x.rstrip() for x in old]:
        return {}
    # a rewrite cut off by the token limit would silently delete the end of the function
    if len(new_lines) < 0.5 * len(old):
        return {}
    return {chunk.path: "\n".join(lines[:chunk.start - 1] + new_lines + lines[chunk.end:])}


REPRO_SYSTEM = ("You write one minimal pytest test that reproduces a reported issue. "
                "You answer only with a single Python code block.")


def repro_prompt(problem: str, context: str) -> str:
    return (f"## Issue\n{problem.strip()[:4000]}\n\n## Relevant code\n{context}\n\n"
            "Write ONE pytest test file that fails today because of this issue and will pass once it is fixed. "
            "Import from the package as its own tests would. No network, no sleeping, under 40 lines. "
            "Reply with a single ```python block only.")


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.S)


# --------------------------------------------------------------------------- pipeline

def _ask(llm, what: str, system: str, text: str, max_new_tokens: int, max_time: float | None = None) -> str:
    t = time.time()
    kw = {} if max_time is None else {"max_time": max_time}
    reply = llm.chat(system, text, max_new_tokens=max_new_tokens, **kw)
    if DEBUG_REPLIES:
        paths.log(f"repo: REPLY {what} >>>\n{(reply or '')[:3000]}\n<<<")
    paths.log(f"repo: llm[{what}] {len(text)} chars in, {len(reply or '')} out, {time.time() - t:.1f}s")
    return reply


class Result:
    def __init__(self):
        self.patch = ""
        self.changed: dict[str, str] = {}
        self.confidence = 0.0
        self.verified = False
        self.label = "no-change"
        self.localized: list[str] = []


def repair_repo(root: Path, problem: str, hints: str = "", llm=None,
                deadline_s: float | None = None) -> Result:
    t0 = time.time()
    deadline = t0 + (REPO_DEADLINE_S if deadline_s is None else deadline_s)
    res = Result()

    ranked = localize.rank_all(root, problem + "\n" + hints,
                               budget_s=min(LOCALIZE_MAX_S, 0.25 * (deadline - t0)))
    res.localized = localize.rank_files(ranked[:14])
    cut = localize.LAST_CUT
    paths.log(f"repo: localized in {time.time() - t0:.1f}s -> {res.localized[:5]}" +
              (f" (out of time: read {cut[0]} of {cut[1]} files)" if cut else ""))
    if llm is None or not ranked:
        return res

    # Let the model name the symbols it wants, from signatures of the best files. Its
    # picks are pinned into the context; BM25 fills the rest, so a bad pick costs little.
    pinned: list[localize.Chunk] = []
    if PICK_SYMBOLS and time.time() < deadline - PICK_MIN_S:
        try:
            sk = {f: localize.skeleton(root, f) for f in res.localized[:3]}
            sk = {f: v for f, v in sk.items() if v}
            if sk:
                pinned = parse_picks(_ask(llm, "pick", PICK_SYSTEM, pick_prompt(problem, sk), 160), ranked)
                paths.log(f"repo: model picked {[c.name for c in pinned]}")
        except Exception as e:  # noqa: BLE001
            paths.log(f"repo: symbol pick failed: {type(e).__name__}: {e}")
    chunks = localize.focus(ranked, CONTEXT_CHARS, pinned=pinned)
    context = build_context(chunks, CONTEXT_CHARS)
    shown: dict[str, list[tuple[int, int]]] = {}
    for c in chunks:
        shown.setdefault(c.path, []).append((c.start, c.end))

    # Reproduction test (best effort): only trusted if it FAILS on the untouched repo.
    def test_time() -> float:
        """Seconds one test run may take now: never past the deadline (2 s kept to write the patch)."""
        return min(TEST_TIMEOUT_S, deadline - time.time() - 2.0)

    repro_rel = None
    if WRITE_REPRO and time.time() < deadline - 60:
        try:
            reply = _ask(llm, "repro", REPRO_SYSTEM, repro_prompt(problem, context[:CONTEXT_CHARS // 2]), 700)
            blocks = _FENCE.findall(reply or "")
            code = max(blocks, key=len) if blocks else ""
            if code.strip() and compiles({"t.py": code})[0] and "def test_" in code:
                rel = "mc5_repro_test.py"
                tree = _overlay(root, {rel: code}, budget_s=min(OVERLAY_MAX_S, test_time() - 2.0))
                try:
                    verdict, out = run_pytest(tree, [rel], test_time())
                finally:
                    shutil.rmtree(tree, ignore_errors=True)
                paths.log(f"repo: reproduction test on base -> {verdict}")
                if verdict == "fail":
                    repro_rel, repro_code = rel, code
        except Exception as e:  # noqa: BLE001
            paths.log(f"repo: reproduction step failed: {type(e).__name__}: {e}")

    baseline: dict[str, str] = {}
    best: tuple[int, dict[str, str], str] = (0, {}, "no-change")
    feedback = ""
    seen_replies: set[str] = set()
    failed_edits: list[tuple[str, str, str]] = []
    unapplied = 0
    for attempt in range(MAX_ATTEMPTS):
        if time.time() > deadline - 8:
            break
        if unapplied and REWRITE and deadline - time.time() < 22:
            break                                         # one more call fits: make it the rewrite
        # Keep part of the clock for the fallback: an edit that cannot be applied is worth
        # nothing, and the rewrite needs no matching at all.
        left = deadline - time.time() - 3.0
        cap = max(6.0, 0.6 * left) if REWRITE and attempt == 0 and left < 60 else None
        try:
            reply = _ask(llm, f"fix{attempt}", SYSTEM, fix_prompt(problem, hints, context, feedback), 1600,
                         max_time=cap)
        except Exception as e:  # noqa: BLE001
            paths.log(f"repo: llm attempt {attempt} failed: {type(e).__name__}: {e}")
            break
        if reply in seen_replies:
            # Greedy decoding: the same answer again means the feedback changed nothing,
            # and a third try would be the same 40-180 s spent on the same text.
            paths.log(f"repo: attempt {attempt}: same reply as before, stopping")
            break
        seen_replies.add(reply)
        edits = parse_edits(reply)
        if not edits:
            feedback = "No SEARCH/REPLACE block could be read from your reply. Use the exact format."
            paths.log(f"repo: attempt {attempt}: no edit blocks ({len(reply or '')} chars)")
            unapplied += 1
            continue
        edits = [e for e in edits if not localize.is_test_path(e[0])] or edits
        changed, problems = apply_edits(root, edits, shown)
        if not changed:
            feedback = ("Edits could not be applied: " + "; ".join(problems[:4]) if problems else
                        "Your edit changed nothing: the REPLACE text was the same as the SEARCH text. "
                        "Write the corrected code in REPLACE.")
            paths.log(f"repo: attempt {attempt}: {feedback}")
            failed_edits = edits
            unapplied += 1
            continue
        ok, why = compiles(changed)
        if not ok:
            feedback = f"The edited file has a syntax error: {why}"
            paths.log(f"repo: attempt {attempt}: syntax error {why}")
            failed_edits = edits
            unapplied += 1
            continue

        feedback = ""                                     # judged afresh below
        score, label = 2, f"llm-{attempt}:compiles"
        if test_time() >= 3:
            tests = nearby_tests(root, list(changed), [c.name for c in chunks[:6] if c.path in changed])
            try:
                tree = _overlay(root, {**changed, **({repro_rel: repro_code} if repro_rel else {})},
                                budget_s=min(OVERLAY_MAX_S, test_time() - 2.0))
            except OverlayTimeout as e:
                paths.log(f"repo: attempt {attempt}: no test verdict, copying the repo took too long ({e})")
                tree = None
            try:
                regress = False
                for tf in (tests if tree is not None else []):
                    if test_time() < 3:
                        break
                    if tf not in baseline:
                        limit = test_time()
                        v0, o0 = run_pytest(root, [tf], limit)
                        if v0 == "error" and o0 == "timeout" and limit < TEST_TIMEOUT_S:
                            break                      # cut short by our own clock: not a baseline
                        baseline[tf] = v0
                    if baseline[tf] != "pass":
                        continue                       # cannot judge with a red or unrunnable baseline
                    if test_time() < 2:
                        break
                    limit = test_time()
                    verdict, out = run_pytest(tree, [tf], limit)
                    if verdict == "error" and out == "timeout" and limit < TEST_TIMEOUT_S:
                        break                          # we cut it short ourselves: no verdict
                    if verdict != "pass":
                        regress = True
                        feedback = f"Your change broke existing tests in {tf}:\n{out[-900:]}"
                        break
                    score, label = 3, f"llm-{attempt}:no-regression"
                if regress:
                    score, label = 1, f"llm-{attempt}:regression"
                elif repro_rel and tree is not None and test_time() >= 3:
                    verdict, out = run_pytest(tree, [repro_rel], test_time())
                    if verdict == "pass":
                        score, label = 5, f"llm-{attempt}:repro-passes"
                    else:
                        feedback = f"The issue still reproduces after your change:\n{out[-900:]}"
            finally:
                if tree is not None:
                    shutil.rmtree(tree, ignore_errors=True)
        paths.log(f"repo: attempt {attempt}: {label} files={sorted(changed)}")
        if score > best[0]:
            best = (score, changed, label)
        if score >= 5 or (score >= 3 and not repro_rel):
            break
        if not feedback:
            # Nothing here can tell a better candidate from this one (no reproduction test,
            # no runnable neighbouring test), and a later attempt can never outrank an
            # equal score. Asking again would only burn the clock.
            break

    # Fallback: no edit could be applied. Have the model write the whole corrected function
    # instead; its lines replace the old ones, so nothing has to match.
    if best[0] == 0 and REWRITE and deadline - time.time() > 7:
        target = rewrite_target(root, chunks, failed_edits, problem)
        if target is not None:
            try:
                n_lines = target.end - target.start + 1
                reply = _ask(llm, "rewrite", REWRITE_SYSTEM, rewrite_prompt(problem, hints, target),
                             min(1600, 120 + 16 * n_lines))
                changed = apply_rewrite(root, target, reply)
                ok, why = compiles(changed) if changed else (False, "no usable code block")
                paths.log(f"repo: rewrite of {target.path}::{target.name} ({n_lines} lines) -> "
                          f"{'ok' if ok else 'rejected: ' + why}")
                if ok:
                    best = (2, changed, "llm-rewrite:compiles")
            except Exception as e:  # noqa: BLE001
                paths.log(f"repo: rewrite failed: {type(e).__name__}: {e}")

    score, changed, label = best
    # A candidate that turns an existing test red ranks below every clean one, but it is
    # still shipped if it is all we have. On 25 practice tasks the REFERENCE fix itself
    # broke an existing test in 5 (the fix changes behaviour and the reference updates
    # that test), so a hard veto threw away correct answers; and an empty patch is a
    # certain fail while this one is only a likely fail.
    if score >= 1:
        res.changed = changed
        res.patch = make_patch(root, changed)
        res.label = label
        res.verified = score >= 5
        res.confidence = {5: 0.8, 3: 0.5, 2: 0.3, 1: 0.15}.get(score, 0.2)
    paths.log(f"repo: done in {time.time() - t0:.1f}s label={res.label}")
    return res
