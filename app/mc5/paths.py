import os
from pathlib import Path

WORK_DIR = Path(os.environ.get("MC5_WORK_DIR", "/tmp/mc5"))
LOG_FILE = WORK_DIR / "app.log"
SERVER_LOG = WORK_DIR / "server.log"
SOCKET = Path(os.environ.get("MC5_SOCKET", "/tmp/mc5-model.sock"))
INPUT_DIR = Path(os.environ.get("MC5_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("MC5_OUTPUT_DIR", "/app/output"))
MODEL_DIR = os.environ.get("MC5_MODEL_DIR", "/models/llm")

# Per-task wall clock. Section 2 of the shared rules gives 30 s per item; we stop at 25 s
# so the output file is always written inside the budget.
DEADLINE_S = float(os.environ.get("MC5_DEADLINE_S", "25"))
# Seconds allowed for one candidate's test run inside the sandbox.
VERIFY_TIMEOUT_S = float(os.environ.get("MC5_VERIFY_TIMEOUT_S", "6"))
# How many model-written candidates to try (each one is verified before it is kept).
MAX_LLM_CANDIDATES = int(os.environ.get("MC5_MAX_LLM_CANDIDATES", "3"))


# pip deps shipped in the app layer (the graded image keeps pytest here, not in the base).
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"


def test_pythonpath(*dirs) -> str:
    """PYTHONPATH for a test subprocess: the code under test first, our vendored
    packages (pytest) last so a task's own modules always win."""
    parts = [str(d) for d in dirs]
    if VENDOR_DIR.is_dir():
        parts.append(str(VENDOR_DIR))
    extra = os.environ.get("MC5_TEST_PYTHONPATH")   # practice runs only: test deps installed on the side
    if extra:
        parts.append(extra)
    return os.pathsep.join(parts)


def have_pytest() -> bool:
    """Can a test subprocess import pytest? (base image, or our vendor dir)."""
    import importlib.util
    try:
        if importlib.util.find_spec("pytest") is not None:
            return True
    except (ImportError, ValueError):
        pass
    return (VENDOR_DIR / "pytest").is_dir()


def log(msg: str) -> None:
    """Logs go to a file, never stdout (stdout noise once cost us OUTPUT_MALFORMED)."""
    import time
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S ") + str(msg).rstrip() + "\n")
    except OSError:
        pass
