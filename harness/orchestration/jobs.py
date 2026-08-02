"""
Background-job plumbing for the UI.

A Job is a plain, thread-safe progress object. A worker thread updates it; the
Streamlit UI polls it. Crucially the worker NEVER calls any Streamlit function
(that would fail outside the script-run context) — it only mutates this object
under a lock. The UI reads it and renders.

The JOBS registry is module-level so it survives Streamlit's per-interaction
reruns; the UI keeps only the job_id in session_state and looks the Job up here.
"""

from __future__ import annotations

import threading
import time
import uuid


class Job:
    def __init__(self, kind: str, total: int):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind                 # "ingest" | "eval"
        self.total = max(1, total)       # avoid divide-by-zero
        self.done = 0
        self.status = "running"          # running | done | error
        self.error: str | None = None
        self.messages: list[str] = []
        self.result_rows = 0
        self.started = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()

    # -- worker-side (thread-safe) -------------------------------------- #
    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = max(1, total)

    def advance(self, n: int = 1, message: str | None = None) -> None:
        with self._lock:
            self.done += n
            if message:
                self.messages.append(message)

    def set_done_absolute(self, done: int) -> None:
        with self._lock:
            self.done = done

    def finish(self, rows: int = 0, message: str | None = None) -> None:
        with self._lock:
            self.status = "done"
            self.result_rows = rows
            self.finished = time.time()
            if message:
                self.messages.append(message)

    def fail(self, err: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = err
            self.finished = time.time()

    # -- UI-side (read snapshot) ---------------------------------------- #
    def snapshot(self) -> dict:
        with self._lock:
            frac = min(1.0, self.done / self.total) if self.total else 0.0
            elapsed = (self.finished or time.time()) - self.started
            return {
                "id": self.id, "kind": self.kind, "status": self.status,
                "done": self.done, "total": self.total, "frac": frac,
                "error": self.error, "rows": self.result_rows,
                "elapsed": elapsed,
                "messages": list(self.messages[-8:]),  # last few lines
            }


# module-level registry (persists across Streamlit reruns)
JOBS: dict[str, Job] = {}


def new_job(kind: str, total: int) -> Job:
    job = Job(kind, total)
    JOBS[job.id] = job
    return job


def get_job(job_id: str | None) -> Job | None:
    return JOBS.get(job_id) if job_id else None


def any_running() -> bool:
    return any(j.status == "running" for j in JOBS.values())
