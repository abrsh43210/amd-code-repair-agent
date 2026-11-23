"""Test double for the LLM (MC5_FAKE_LLM=1), used only by the offline tests.
Never the shipped path.

Default behaviour: make one crude, plausible-looking edit to the buggy file - the
kind of change a weak model produces. That proves an unverified model patch never
displaces a verified fix from the mutation search.

With MC5_FAKE_ORACLE pointing at a JSON map {buggy_code: fixed_code}, it returns
the real fix for the files listed there, which exercises the whole model path
(prompt -> reply -> fence extraction -> verification -> selection) on a case the
model-free search cannot repair.
"""
from __future__ import annotations

import json
import os
import re


class FakeLLM:
    def __init__(self):
        self.oracle: dict[str, str] = {}
        path = os.environ.get("MC5_FAKE_ORACLE", "")
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                self.oracle = json.load(fh)

    def chat(self, system, text, max_new_tokens=120, max_time=None):
        if "SEARCH/REPLACE" in system or "reproduces a reported issue" in system or "locate the code" in system:
            # Repo-level mode: the oracle may carry one scripted answer under "__repo__".
            return self.oracle.get("__repo__", "") if "SEARCH/REPLACE" in system else ""
        m = re.search(r"Buggy file:\n```\n(.*?)\n```", text, re.S)
        code = m.group(1) if m else ""
        for buggy, fixed in self.oracle.items():
            if buggy.strip() == code.strip():
                return "Here is the corrected file:\n```python\n" + fixed.rstrip("\n") + "\n```"
        wrong = re.sub(r"(?<![<>=!])<(?![=])", "<=", code, count=1)
        if wrong == code:
            wrong = code.replace("return ", "return  ", 1) or code + "\n"
        return "```python\n" + wrong.rstrip("\n") + "\n```"
