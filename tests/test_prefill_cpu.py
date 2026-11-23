#!/usr/bin/env python3
"""Sliced prefill must give the same answer as reading the prompt at once.
Needs torch + a small model (CI: Qwen2.5-Coder-0.5B on CPU). MC5_MODEL_DIR, MC5_ALLOW_CPU=1."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
os.environ.setdefault("MC5_ALLOW_CPU", "1")
os.environ.setdefault("MC5_WORK_DIR", "/tmp/mc5-prefill")

from mc5 import model, paths  # noqa: E402

llm = model.LocalLLM()
prompt = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(180)) + "\n\nWhat does f7(1) return? Answer with the number and one short sentence."
model.PREFILL_CHUNK = 0
plain = llm.chat("You are a careful Python reader.", prompt, max_new_tokens=40)
model.PREFILL_CHUNK = 256
sliced = llm.chat("You are a careful Python reader.", prompt, max_new_tokens=40)
log = paths.LOG_FILE.read_text()
print("plain :", repr(plain))
print("sliced:", repr(sliced))
print(log[-600:])
assert "sliced prefill failed" not in log, "sliced path fell back"
assert plain == sliced, "answers differ"
print("ALL PASSED")
