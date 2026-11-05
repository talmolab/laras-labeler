"""Tiny background job runner with an append-only event log (PLAN.md §8).

One worker thread per job; events are appended to a list so SSE and poll both work,
and late/reconnecting subscribers still see the whole history. Single-user, so no
cross-job locking beyond this.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from itertools import count

TERMINAL = ("done", "error", "cancelled")
_local = threading.local()          # the job this worker thread is running, for code deep inside it


class JobCancelled(Exception):
    """Raised inside a job's thread once someone asked it to stop (JobManager.cancel)."""


def current() -> "Job | None":
    return getattr(_local, "job", None)


def raise_if_cancelled() -> None:
    """For long waits and loops that don't report progress: stop here if this job was cancelled."""
    job = current()
    if job is not None and job.cancel_event.is_set():
        raise JobCancelled("stopped")


@contextlib.contextmanager
def track(proc):
    """Run a subprocess as part of the current job: cancelling the job kills it, and leaving the block
    early (an error, or the cancel arriving through a progress callback) never leaves it running."""
    job = current()
    if job is not None:
        with job._lock:
            job._procs.add(proc)
    try:
        yield proc
    finally:
        if job is not None:
            with job._lock:
                job._procs.discard(proc)
        if proc.poll() is None:
            _kill(proc)


def _kill(proc) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


class Job:
    def __init__(self, job_id: str, kind: str, meta: dict) -> None:
        self.id = job_id
        self.kind = kind
        self.meta = meta
        self.events: list[dict] = []
        self.status = "pending"
        self.result: dict | None = None
        self._lock = threading.Lock()
        self.cancel_event = threading.Event()
        self._procs: set = set()          # subprocesses to kill on cancel (see track)

    def _emit(self, **ev) -> None:
        with self._lock:
            self.events.append(ev)

    def snapshot(self) -> dict:
        with self._lock:
            last = self.events[-1] if self.events else {}
        return {"job_id": self.id, "kind": self.kind, "status": self.status,
                "pct": last.get("pct"), "msg": last.get("msg"), "error": last.get("error"),
                "result": self.result, **self.meta}


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._ids = count(1)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop: its next progress report (or raise_if_cancelled) ends it as "cancelled",
        a wait for its turn stops waiting, and any subprocess it runs is killed now."""
        job = self._jobs.get(job_id)
        if job is None or job.status in TERMINAL:
            return job
        job.cancel_event.set()
        with job._lock:
            procs = list(job._procs)
        for p in procs:
            _kill(p)
        return job

    def start(self, kind: str, fn, meta: dict | None = None) -> Job:
        job = Job(f"job-{next(self._ids)}", kind, meta or {})
        self._jobs[job.id] = job

        def run():
            _local.job = job
            job.status = "running"
            job._emit(pct=0, msg="started", status="running")
            try:
                def progress(pct, msg):
                    if job.cancel_event.is_set():
                        raise JobCancelled("stopped")
                    job._emit(pct=pct, msg=msg, status="running")
                job.result = fn(progress) or {}
                if job.cancel_event.is_set():
                    raise JobCancelled("stopped")
                job.status = "done"
                job._emit(pct=100, msg="done", status="done", result=job.result)
            except Exception as e:  # noqa: BLE001
                if job.cancel_event.is_set():     # stopped on request (however it surfaced): not an error
                    job.status = "cancelled"
                    job._emit(status="cancelled", msg="stopped")
                else:
                    job.status = "error"
                    job._emit(status="error", error=str(e), trace=traceback.format_exc())
            finally:
                _local.job = None

        threading.Thread(target=run, daemon=True).start()
        return job

    def stream(self, job_id: str):
        """Yield every event (including history) until a terminal one, for SSE."""
        job = self._jobs[job_id]
        i = 0
        while True:
            if i < len(job.events):
                ev = job.events[i]
                i += 1
                yield ev
                if ev.get("status") in TERMINAL:
                    return
            elif job.status in TERMINAL:
                return
            else:
                time.sleep(0.05)
