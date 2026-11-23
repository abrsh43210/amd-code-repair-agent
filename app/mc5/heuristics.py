"""Deterministic repair candidates: no model, no GPU, microseconds each.

Most benchmark code-repair bugs are one of a short list of mutations: a flipped
comparison, an off-by-one bound, a swapped operand, a wrong operator, a missing
return. Enumerating those mutations and letting the verifier judge them is both
faster and more reliable than asking a 4B model to be careful, so we try these
first and only call the model when none of them passes the tests.

Each generator yields (candidate_code, label). Candidates are deduplicated and
capped by the caller.
"""
from __future__ import annotations

import re
from typing import Iterator

# (pattern, replacements) applied one occurrence at a time.
_TOKEN_SWAPS: list[tuple[str, tuple[str, ...]]] = [
    (r"<=", ("<",)),
    (r">=", (">",)),
    (r"(?<![<>=!])<(?![=])", ("<=", ">")),
    (r"(?<![<>=!])>(?![=])", (">=", "<")),
    (r"(?<![=!<>+\-*/%])==(?![=])", ("!=",)),
    (r"!=", ("==",)),
    (r"(?<![/])//(?![/])", ("/",)),       # integer division where float was meant
    (r"(?<![*])\*(?![*=])", ("+",)),
    (r"(?<![+])\+(?![+=])", ("-",)),
    (r"(?<![-])-(?![-=>])", ("+",)),
    (r"\bor\b", ("and",)),
    (r"\band\b", ("or",)),
    (r"\bis not\b", ("is",)),
    (r"\bnot in\b", ("in",)),
    (r"(?<![\w.])append\(", ("extend(",)),
    (r"(?<![\w.])min\(", ("max(",)),
    (r"(?<![\w.])max\(", ("min(",)),
    (r"(?<![\w.])sorted\(", ("reversed(",)),
]

# Numeric nudges on literals that appear in slices, ranges and comparisons.
_OFFSET_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def _one_at_a_time(code: str, pattern: str, repl: str, label: str) -> Iterator[tuple[str, str]]:
    for m in list(re.finditer(pattern, code)):
        yield code[:m.start()] + repl + code[m.end():], f"{label}@{m.start()}"


def token_swaps(code: str, limit: int = 220) -> Iterator[tuple[str, str]]:
    n = 0
    for pat, repls in _TOKEN_SWAPS:
        for repl in repls:
            for cand, label in _one_at_a_time(code, pat, repl, f"{pat}->{repl}"):
                yield cand, label
                n += 1
                if n >= limit:
                    return


def offset_nudges(code: str, limit: int = 120) -> Iterator[tuple[str, str]]:
    """Off-by-one: change one integer literal by +/-1, and fix the common
    `range(len(x) - 1)` / `range(1, len(x))` boundary slips."""
    n = 0
    for m in list(_OFFSET_RE.finditer(code)):
        try:
            val = int(m.group(1))
        except ValueError:
            continue
        if val > 10_000:
            continue
        for new in (val + 1, val - 1):
            if new < 0:
                continue
            yield code[:m.start()] + str(new) + code[m.end():], f"int{val}->{new}@{m.start()}"
            n += 1
            if n >= limit:
                return


def drop_offset_terms(code: str, limit: int = 60):
    """Remove a whole `- 1` / `+ 2` term: `scored[:k - 1]` -> `scored[:k]`.

    This is the clean form of an off-by-one fix. offset_nudges() would reach the
    same behaviour as `k - 0`, which verifies but reads badly, so this runs first.
    """
    n = 0
    for m in list(re.finditer(r"\s*[-+]\s*\d+", code)):
        before = code[:m.start()]
        if not re.search(r"[\w)\]]$", before):          # not a binary operator
            continue
        yield before + code[m.end():], f"drop-offset@{m.start()}"
        n += 1
        if n >= limit:
            return


def range_bounds(code: str) -> Iterator[tuple[str, str]]:
    variants = [
        (r"range\(len\(([^()]+)\)\s*-\s*1\)", r"range(len(\1))", "range-len-minus-1"),
        (r"range\(len\(([^()]+)\)\s*\+\s*1\)", r"range(len(\1))", "range-len-plus-1"),
        (r"range\(1,\s*len\(([^()]+)\)\)", r"range(len(\1))", "range-from-1"),
        (r"range\(0,\s*len\(([^()]+)\)\s*-\s*1\)", r"range(len(\1))", "range-0-len-1"),
        (r"range\(len\(([^()]+)\)\)", r"range(len(\1) - 1)", "range-len-to-len-1"),
    ]
    for pat, repl, label in variants:
        if re.search(pat, code):
            yield re.sub(pat, repl, code, count=1), label


def mutable_default(code: str) -> Iterator[tuple[str, str]]:
    """def f(x, acc=[]) -> the classic shared-default bug."""
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
    while m:
        params = m.group(2)
        for lit, empty in (("[]", "list"), ("{}", "dict"), ("set()", "set")):
            pm = re.search(rf"(\w+)\s*=\s*{re.escape(lit)}", params)
            if not pm:
                continue
            name = pm.group(1)
            new_params = params[:pm.start()] + f"{name}=None" + params[pm.end():]
            head = code[:m.start(2)] + new_params + code[m.end(2):]
            body_at = head.find(":", m.end(2) - len(params) + len(new_params))
            if body_at == -1:
                continue
            nl = head.find("\n", body_at)
            if nl == -1:
                continue
            indent = re.match(r"[ \t]*", head[nl + 1:]).group(0) or "    "
            guard = f"{indent}if {name} is None:\n{indent}    {name} = {lit if lit != 'set()' else 'set()'}\n"
            yield head[:nl + 1] + guard + head[nl + 1:], f"mutable-default-{empty}"
        break
    return


def missing_return(code: str) -> Iterator[tuple[str, str]]:
    """A function whose last statement computes a value but discards it."""
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s+)([A-Za-z_][\w.\[\]]*)\s*$", ln)
        if m and i and re.match(r"^\s*(return|yield)\b", lines[i - 1]) is None:
            out = lines[:i] + [f"{m.group(1)}return {m.group(2)}"] + lines[i + 1:]
            yield "\n".join(out), f"bare-expr-to-return:{i}"


def swap_args(code: str, limit: int = 40) -> Iterator[tuple[str, str]]:
    """f(a, b) -> f(b, a) for two-argument calls, a frequent transposition bug."""
    n = 0
    for m in list(re.finditer(r"(\w+)\(\s*([A-Za-z_][\w.]*)\s*,\s*([A-Za-z_][\w.]*)\s*\)", code)):
        fn, a, b = m.group(1), m.group(2), m.group(3)
        if fn in ("range", "print", "isinstance", "setattr", "getattr"):
            continue
        yield code[:m.start()] + f"{fn}({b}, {a})" + code[m.end():], f"swap-args-{fn}@{m.start()}"
        n += 1
        if n >= limit:
            return


def candidates(code: str, language: str = "python", limit: int = 400) -> list[tuple[str, str]]:
    """Ordered, deduplicated. Cheapest and most likely classes first."""
    gens = [range_bounds(code), drop_offset_terms(code), mutable_default(code),
            missing_return(code), token_swaps(code), offset_nudges(code), swap_args(code)]
    if language != "python":
        gens = [drop_offset_terms(code), token_swaps(code), offset_nudges(code)]
    seen = {code}
    out: list[tuple[str, str]] = []
    for gen in gens:
        for cand, label in gen:
            if cand in seen:
                continue
            seen.add(cand)
            out.append((cand, label))
            if len(out) >= limit:
                return out
    return out
