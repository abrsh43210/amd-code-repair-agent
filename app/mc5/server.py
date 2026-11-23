"""Resident model server: started once (warm-up or first question), keeps the LLM on
the GPU and serves chat requests over a unix socket, so each per-question process
only does web work and one short request.

  {"op": "ping"}                                  -> {"ok": true}
  {"op": "chat", "system": s, "text": t, "max_new_tokens": n} -> {"ok": true, "text": "..."}
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "vendor"))

from mc5 import paths  # noqa: E402


def log(msg: str) -> None:
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.SERVER_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")


def serve(llm) -> None:
    lock = threading.Lock()                      # one generate at a time on the GPU
    sock_path = str(paths.SOCKET)
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(16)
    log("ready")

    def handle(conn):
        with conn:
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf.decode("utf-8") or "{}")
                if req.get("op") == "ping":
                    resp = {"ok": True}
                elif req.get("op") == "chat":
                    with lock:
                        dl = req.get("deadline")
                        if dl is not None and time.time() > float(dl):
                            raise TimeoutError("request expired while queued")   # client gave up
                        # Stop generating in time for the reply to reach a client that is
                        # still listening; a cut-off reply beats no reply.
                        budget = None if dl is None else max(0.3, float(dl) - time.time() - 0.3)
                        text = llm.chat(req.get("system", ""), req.get("text", ""),
                                        max_new_tokens=int(req.get("max_new_tokens", 120)), max_time=budget)
                    resp = {"ok": True, "text": text}
                else:
                    resp = {"ok": False, "error": "unknown op"}
            except Exception as e:  # noqa: BLE001
                log(traceback.format_exc())
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                pass

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main() -> None:
    import fcntl
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh = open(paths.WORK_DIR / "server.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # one server per container
    except OSError:
        return
    (paths.WORK_DIR / "server.started").write_text(str(time.time()))
    if os.environ.get("MC5_FAKE_LLM"):
        from mc5.fake_llm import FakeLLM
        llm = FakeLLM()
    else:
        from mc5.model import LocalLLM
        llm = LocalLLM()
    serve(llm)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
