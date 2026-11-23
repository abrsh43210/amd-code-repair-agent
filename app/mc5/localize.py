"""Where is the bug? Model-free fault localization over a repository.

A small model cannot read a repository, so the quality of a repo-level fix is capped
by what we put in front of it. This module turns "issue text + repo" into a ranked
list of code chunks (functions, methods, classes, module remainders):

  * every source .py file is split along its AST into chunks with qualified names
  * chunks are ranked by BM25 against the issue text
  * strong evidence is boosted: identifiers quoted in the issue that are a chunk's
    own name, file paths and traceback frames named in the issue

No embeddings, no GPU, about a second for a mid-sized repo.
"""
from __future__ import annotations

import ast
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".tox", ".venv", "venv", "env", "node_modules", "__pycache__", "build",
             "dist", ".eggs", "site-packages", "docs", "docs_src", "doc", "examples", "benchmarks",
             ".github", "scripts"}
MAX_FILE_BYTES = 600_000
MAX_FILES = 4000
MAX_LISTED = 60000
LAST_CUT: tuple[int, int] | None = None      # (files read, files found) when rank() ran out of time

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
STOP = set("""a an the and or not of to in is it this that for on with as be are was were by at from if
when then else but so we you i they he she them our your can could should would will do does did has
have had been being which what who how why where there here into out up down over under about after
before again more most some any all no nor only own same than too very just also get got use using used
self cls def return none true false import class try except raise pass""".split())


def is_test_path(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")
    return (any(p in ("tests", "test", "testing") for p in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if len(lw) > 1 and lw not in STOP:
            out.append(lw)
        if "_" in w or _CAMEL.search(w):
            for part in re.split(r"_+", w):
                for sub in _CAMEL.split(part):
                    sl = sub.lower()
                    if len(sl) > 1 and sl not in STOP and sl != lw:
                        out.append(sl)
    return out


class Chunk:
    __slots__ = ("path", "name", "start", "end", "text", "kind", "score", "tf", "length")

    def __init__(self, path: str, name: str, start: int, end: int, text: str, kind: str):
        self.path, self.name, self.start, self.end, self.text, self.kind = path, name, start, end, text, kind
        self.score = 0.0
        toks = tokens(name) * 3 + tokens(path) * 2 + tokens(text)
        self.tf = Counter(toks)
        self.length = len(toks) or 1

    def __repr__(self) -> str:
        return f"<{self.path}:{self.start}-{self.end} {self.name} {self.score:.2f}>"


def source_files(root: Path, include_tests: bool = False, max_files: int | None = None,
                 t_end: float | None = None) -> list[str]:
    """`t_end`: wall-clock time at which the walk stops (checked once per directory)."""
    max_files = MAX_FILES if max_files is None else max_files
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        if t_end is not None and time.time() > t_end:
            break
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            if not include_tests and is_test_path(rel):
                continue
            try:
                if os.path.getsize(os.path.join(dirpath, fn)) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(rel)
            if len(found) >= max_files:
                return sorted(found)
    return sorted(found)


def chunk_file(rel: str, text: str) -> list[Chunk]:
    lines = text.split("\n")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return [Chunk(rel, rel, i + 1, min(i + 60, len(lines)), "\n".join(lines[i:i + 60]), "window")
                for i in range(0, len(lines), 50)]
    chunks: list[Chunk] = []
    covered = [False] * (len(lines) + 2)

    def start_of(node) -> int:
        decos = getattr(node, "decorator_list", None) or []
        return min([node.lineno] + [d.lineno for d in decos])

    def add(node, qual: str, kind: str) -> None:
        s, e = start_of(node), getattr(node, "end_lineno", node.lineno)
        chunks.append(Chunk(rel, qual, s, e, "\n".join(lines[s - 1:e]), kind))
        for i in range(s, e + 1):
            covered[i] = True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            size = getattr(node, "end_lineno", node.lineno) - node.lineno
            if not methods or size <= 40:
                add(node, node.name, "class")
                continue
            # Large class: one chunk per method, plus the class header/attributes.
            first = start_of(methods[0])
            s = start_of(node)
            if first > s:
                chunks.append(Chunk(rel, node.name, s, first - 1, "\n".join(lines[s - 1:first - 1]), "class-head"))
            for i in range(s, getattr(node, "end_lineno", node.lineno) + 1):
                covered[i] = True
            for m in methods:
                ms, me = start_of(m), getattr(m, "end_lineno", m.lineno)
                chunks.append(Chunk(rel, f"{node.name}.{m.name}", ms, me, "\n".join(lines[ms - 1:me]), "method"))
    # Module-level remainder (constants, imports, top-level logic) in runs.
    i = 1
    while i <= len(lines):
        if covered[i] or not lines[i - 1].strip():
            i += 1
            continue
        j = i
        while j <= len(lines) and not covered[j] and j - i < 80:
            j += 1
        body = "\n".join(lines[i - 1:j - 1])
        if body.strip():
            chunks.append(Chunk(rel, f"<module {rel}>", i, j - 1, body, "module"))
        i = j
    return chunks


def _issue_signals(issue: str) -> tuple[set[str], set[str], list[tuple[str, int]]]:
    """(code identifiers, mentioned paths, traceback frames) named by the issue."""
    idents: set[str] = set()
    for m in re.finditer(r"`([^`\n]{1,120})`", issue):
        idents.update(w for w in _WORD.findall(m.group(1)) if len(w) > 2)
    for m in re.finditer(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+)\b", issue):       # dotted.names
        idents.update(p for p in m.group(1).split(".") if len(p) > 2)
    for m in re.finditer(r"\b([A-Za-z_]\w*)\(", issue):                            # calls()
        if len(m.group(1)) > 2:
            idents.add(m.group(1))
    for w in _WORD.findall(issue):
        if ("_" in w.strip("_") or _CAMEL.search(w)) and len(w) > 3:              # snake_case / CamelCase
            idents.add(w)
    paths = set(re.findall(r"[\w./-]+\.py\b", issue))
    frames = [(m.group(1), int(m.group(2)))
              for m in re.finditer(r'File "([^"]+\.py)", line (\d+)', issue)]
    return idents, paths, frames


def rank(root: Path, issue: str, top: int = 12, include_tests: bool = False,
         budget_s: float | None = None) -> list[Chunk]:
    """`budget_s`: seconds this may take in all. On a repo too big for it, the files whose
    path shares words with the issue are read first and the rest are left out (LAST_CUT
    then says how many files were read of how many)."""
    global LAST_CUT
    LAST_CUT = None
    t0 = time.time()
    # With a clock the walk may list far more files than can be read: the order below
    # decides which are, so a file the issue names is never lost to the listing cap.
    files = source_files(root, include_tests, max_files=None if budget_s is None else MAX_LISTED,
                         t_end=None if budget_s is None else t0 + 0.3 * budget_s)
    q = Counter(tokens(issue))
    if budget_s is not None:
        named = set(re.findall(r"[\w./-]+\.py\b", issue))
        def prior(rel: str) -> int:
            base = rel.rsplit("/", 1)[-1]
            hit = sum(1 for t in set(tokens(rel[:-3])) if t in q)
            return -(hit + (10 if any(n.endswith(rel) or n.endswith(base) for n in named) else 0))
        files.sort(key=prior)                              # stable: ties keep path order
    chunks: list[Chunk] = []
    for i, rel in enumerate(files):
        if budget_s is not None and i % 20 == 0 and time.time() - t0 > 0.5 * budget_s:
            LAST_CUT = (i, len(files))
            break
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.extend(chunk_file(rel, text))
    if not chunks:
        return []

    n = len(chunks)
    df: Counter = Counter()
    for c in chunks:
        df.update(c.tf.keys())
    avg = sum(c.length for c in chunks) / n
    k1, b = 1.4, 0.75
    idents, paths, frames = _issue_signals(issue)
    ident_l = {i.lower() for i in idents}

    for c in chunks:
        s = 0.0
        for t, qf in q.items():
            f = c.tf.get(t)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * c.length / avg)) * min(qf, 3) ** 0.5
        leaf = c.name.rsplit(".", 1)[-1].lower()
        if leaf in ident_l:
            s *= 1.9                                   # the issue names this very symbol
        elif c.kind != "module" and any(p.lower() in ident_l for p in c.name.split(".")):
            s *= 1.35
        base = c.path.rsplit("/", 1)[-1]
        if any(p.endswith(c.path) or p.endswith(base) for p in paths):
            s *= 1.5
        for fpath, line in frames:
            if fpath.replace("\\", "/").endswith(c.path) and c.start <= line <= c.end:
                s = s * 2.5 + 5.0                       # a traceback frame points inside
        if c.kind == "module":
            s *= 0.8
        c.score = s
    chunks.sort(key=lambda c: -c.score)
    return chunks[:top]


def rank_files(chunks: list[Chunk]) -> list[str]:
    best: dict[str, float] = {}
    for c in chunks:
        best[c.path] = best.get(c.path, 0.0) + (c.score if c.path not in best else 0.3 * c.score)
    return [p for p, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def rank_all(root: Path, issue: str, include_tests: bool = False,
             budget_s: float | None = None) -> list[Chunk]:
    """Every chunk in the repo, best first."""
    return rank(root, issue, top=10 ** 9, include_tests=include_tests, budget_s=budget_s)


def focus(chunks: list[Chunk], budget_chars: int, n_files: int = 3, seed: int = 14,
          pinned: list[Chunk] | None = None) -> list[Chunk]:
    """Pick the chunks to show a model, within a character budget.

    Measured on practice tasks: taking the best chunks repo-wide puts every line the
    reference fix touches in view for 12 of 25 tasks; concentrating on the three
    best files instead does so for 18-20 of 25. A fix usually touches several places
    in one file, and the second and third of those rank poorly on their own.
    """
    files = rank_files(chunks[:seed])[:n_files]
    if pinned:
        # The model read the signatures of the best files and named what it wants: its
        # first named file leads the context (and gets the larger share), because a
        # model edits the file it sees first. AMD run 4: in 6 of 9 wrong-file misses
        # the right file was ranked second by BM25 and the model edited the first.
        lead = list(dict.fromkeys(c.path for c in pinned))
        files = list(dict.fromkeys(lead + files))[:max(n_files, len(lead[:1]) + n_files - 1)]
    picked: list[Chunk] = []
    seen: set[tuple[str, int]] = set()
    used = 0

    def take(c: Chunk) -> bool:
        nonlocal used
        key = (c.path, c.start)
        if key in seen:
            return True
        cost = len(c.text) + 80
        if used + cost > budget_chars and picked:
            return False
        seen.add(key)
        picked.append(c)
        used += cost
        return True

    for c in pinned or []:
        take(c)
    per_file = {f: [c for c in chunks if c.path == f] for f in files}
    # Round-robin, best file first and taking two from it for each one from the others.
    idx = {f: 0 for f in files}
    progressed = True
    while progressed and used < budget_chars:
        progressed = False
        for rank_i, f in enumerate(files):
            for _ in range(2 if rank_i == 0 else 1):
                lst = per_file[f]
                while idx[f] < len(lst):
                    c = lst[idx[f]]
                    idx[f] += 1
                    if c.score <= 0 and picked:
                        break
                    if len(c.text) > budget_chars // 2:
                        continue                      # one giant chunk must not eat the budget
                    if take(c):
                        progressed = True
                    break
    picked.sort(key=lambda c: (files.index(c.path) if c.path in files else 99, c.start))
    return picked


def skeleton(root: Path, rel: str, max_chars: int = 6000) -> str:
    """Signatures only: what a model needs to name the symbols it wants to read."""
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return ""
    lines = text.split("\n")
    out: list[str] = []

    def sig(node, indent: str, prefix: str) -> None:
        head = lines[node.lineno - 1].strip()
        if not head.endswith(":"):
            head = head + " ..."
        out.append(f"{indent}{head}    # {prefix}{node.name}  L{node.lineno}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig(node, "", "")
        elif isinstance(node, ast.ClassDef):
            sig(node, "", "")
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig(m, "    ", node.name + ".")
        elif isinstance(node, ast.Assign) and node.col_offset == 0:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and names[0].isupper():
                out.append(f"{names[0]} = ...    # L{node.lineno}")
    return "\n".join(out)[:max_chars]
