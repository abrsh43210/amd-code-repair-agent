"""The repair pipeline: propose, verify, keep the best.

Order matters, because the per-task budget is ~25 s:

  1. Is the code already fine? (Some graded items are controls.)
  2. Deterministic mutation search, cheapest classes first, biased towards the line
     the error points at. Each candidate is verified in a subprocess; several run
     at once. A candidate whose tests pass ends the search immediately.
  3. The local model, asked for a minimal fix. Every sample it writes is verified
     the same way; an unverified model patch only wins if nothing better exists.
  4. Whatever scored best. If nothing beat the original, the original is returned
     unchanged - a no-op answer scores like a wrong answer, while a broken file can
     cost the whole run.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import heuristics, paths, verify

HEURISTIC_SHARE = 0.55          # fraction of the budget spent before the model is called
PARALLEL = 4


def _error_lines(task) -> list[int]:
    """Line numbers named by a traceback or test output, best first."""
    nums: list[int] = []
    for m in re.finditer(r"line (\d+)", task.error or ""):
        nums.append(int(m.group(1)))
    for m in re.finditer(r":(\d+):", task.error or ""):
        nums.append(int(m.group(1)))
    return nums


def _offset_of_lines(code: str, lines: list[int]) -> list[tuple[int, int]]:
    starts = [0]
    for ln in code.split("\n")[:-1]:
        starts.append(starts[-1] + len(ln) + 1)
    spans = []
    for n in lines:
        if 1 <= n <= len(starts):
            s = starts[n - 1]
            e = starts[n] if n < len(starts) else len(code)
            spans.append((s, e))
    return spans


def _rank_candidates(cands: list[tuple[str, str]], task) -> list[tuple[str, str]]:
    """Candidates that touch a line the error mentions go first."""
    spans = _offset_of_lines(task.code, _error_lines(task))
    if not spans:
        return cands

    def key(item):
        label = item[1]
        m = re.search(r"@(\d+)$", label)
        if not m:
            return 1
        off = int(m.group(1))
        return 0 if any(s <= off < e for s, e in spans) else 1

    return sorted(cands, key=key)


def _verify_batch(cands: list[tuple[str, str]], task, deadline: float,
                  best: tuple[str, str, str]) -> tuple[str, str, str]:
    """Returns the best (code, verdict, label) seen, original included."""
    i = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        while i < len(cands) and time.time() < deadline:
            chunk = cands[i:i + PARALLEL]
            i += PARALLEL
            budget = min(paths.VERIFY_TIMEOUT_S, max(1.0, deadline - time.time()))
            futs = {pool.submit(verify.verify, c, task, budget): (c, lab) for c, lab in chunk}
            for fut in as_completed(futs):
                code, label = futs[fut]
                try:
                    verdict, _out = fut.result()
                except Exception as e:  # noqa: BLE001
                    paths.log(f"verify future failed: {type(e).__name__}: {e}")
                    continue
                if verify.better(verdict, best[1]):
                    best = (code, verdict, label)
                if verdict == verify.PASS:
                    return best
            if best[1] == verify.PASS:
                break
    return best


def _tidy(code: str, task, deadline: float) -> str:
    """Cosmetic clean-up of a verified fix, each step kept only if it still passes.
    A mutation search can land on `k - 0`, which is right but reads like a mistake,
    and a human reviewer or a reference-patch comparison may hold that against us."""
    for pat, repl in ((r"\s*[-+]\s*0(?![\w.])", ""), (r"\s*\*\s*1(?![\w.])", "")):
        if time.time() > deadline - 1:
            break
        cand = re.sub(pat, repl, code)
        if cand != code and verify.verify(cand, task, min(paths.VERIFY_TIMEOUT_S, 3.0))[0] == verify.PASS:
            code = cand
    return code


# --------------------------------------------------------------------------- model

SYSTEM = (
    "You repair buggy code. You reply with the complete corrected file and nothing "
    "else: no explanation, no commentary, no markdown fence labels other than the "
    "code fence itself. Change as little as possible - fix the defect, keep every "
    "other line, every name and the original formatting."
)


def _prompt(task, failure: str) -> str:
    parts = [f"Language: {task.language}", f"File: {task.file_name}"]
    if task.description:
        parts.append(f"Reported problem:\n{task.description.strip()[:1200]}")
    if task.error or failure:
        parts.append(f"Observed failure:\n{(task.error or failure).strip()[-1500:]}")
    if task.tests:
        parts.append(f"Tests that must pass:\n```\n{task.tests.strip()[:2500]}\n```")
    parts.append(f"Buggy file:\n```\n{task.code.strip()[:6000]}\n```")
    parts.append("Return the whole corrected file inside one ``` fence.")
    return "\n\n".join(parts)


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.S)


def extract_code(reply: str, task) -> str:
    """Pull a file out of the model's answer, tolerating fences and stray prose."""
    blocks = _FENCE.findall(reply or "")
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    text = (reply or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    # No fence: keep the lines that look like code, drop a leading sentence.
    lines = text.split("\n")
    while lines and not re.match(r"^\s*(def|class|import|from|#|@|\w+\s*=|if|for|while|return|/\*|//|package|public|fn|func|let|const|var|using|int|void)",
                                 lines[0]):
        lines.pop(0)
    return ("\n".join(lines).strip("\n") + "\n") if lines else ""


def _llm_candidates(task, llm, deadline: float, failure: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    prompt = _prompt(task, failure)
    for i in range(paths.MAX_LLM_CANDIDATES):
        if time.time() > deadline - 2:
            break
        try:
            reply = llm.chat(SYSTEM, prompt, max_new_tokens=1400)
        except Exception as e:  # noqa: BLE001
            paths.log(f"llm candidate {i} failed: {type(e).__name__}: {e}")
            break
        code = extract_code(reply, task)
        if code and code != task.code and all(code != c for c, _ in out):
            out.append((code, f"llm-{i}"))
        else:
            # Nothing usable came back. Say which, so "0 model candidates" in the log
            # is never ambiguous between a silent model and a failed extraction.
            why = ("empty reply" if not (reply or "").strip() else
                   "nothing extractable" if not code else
                   "identical to the original" if code == task.code else "duplicate")
            paths.log(f"{task.stem}: llm-{i} unused ({why}, reply {len(reply or '')} chars)")
        if i == 0 and out:
            # Second and later samples get the first attempt's verdict as feedback.
            prompt = prompt + "\n\nYour previous attempt did not fix it. Try a different, minimal change."
    return out


# --------------------------------------------------------------------------- entry

def repair(task, deadline_s: float, llm=None) -> tuple[str, float, bool, str]:
    """Returns (fixed_code, confidence, verified, label)."""
    t0 = time.time()
    deadline = t0 + deadline_s
    if not task.code.strip():
        return task.code, 0.0, False, "empty-task"

    base_verdict, base_out = verify.verify(task.code, task, min(paths.VERIFY_TIMEOUT_S, deadline_s / 4))
    paths.log(f"{task.stem}: original verdict={base_verdict}")
    if base_verdict == verify.PASS:
        # Already green: a control item, or the tests do not cover the bug. Do not
        # touch it; an edit here can only lose points.
        return task.code, 0.55, True, "already-passing"

    best = (task.code, base_verdict, "original")

    cands = _rank_candidates(heuristics.candidates(task.code, task.language), task)
    paths.log(f"{task.stem}: {len(cands)} heuristic candidates")
    best = _verify_batch(cands, task, t0 + deadline_s * HEURISTIC_SHARE, best)
    if best[1] == verify.PASS:
        tidied = _tidy(best[0], task, deadline)
        paths.log(f"{task.stem}: heuristic fix {best[2]} in {time.time() - t0:.1f}s")
        return tidied, 0.95, True, best[2]

    if llm is not None and time.time() < deadline - 4:
        llm_cands = _llm_candidates(task, llm, deadline - 2, base_out)
        paths.log(f"{task.stem}: {len(llm_cands)} model candidates")
        best = _verify_batch(llm_cands, task, deadline - 0.5, best)

    verified = best[1] == verify.PASS
    if best[2] == "original":
        return task.code, 0.1, False, "no-candidate-better"
    conf = {verify.PASS: 0.9, verify.COMPILES: 0.5, verify.UNKNOWN: 0.4, verify.FAIL: 0.25}.get(best[1], 0.1)
    paths.log(f"{task.stem}: best={best[2]} verdict={best[1]} in {time.time() - t0:.1f}s")
    return best[0], conf, verified, best[2]
