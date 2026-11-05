"""■ Stop: cancelling a job ends it as "cancelled" (not an error), kills a subprocess it runs, and pulls a
job out of the queue it is waiting in. python -m pytest tests/test_jobs_cancel.py"""
import subprocess
import sys
import threading
import time

from laras_labeler.jobs import JobManager, raise_if_cancelled, track


def _wait(job, timeout=10.0):
    t0 = time.time()
    while job.status not in ("done", "error", "cancelled") and time.time() - t0 < timeout:
        time.sleep(0.02)
    return time.time() - t0


def test_cancel_ends_a_job_at_its_next_progress():
    jm = JobManager()

    def fn(progress):
        for i in range(500):
            progress(i, "working")
            time.sleep(0.01)
        return {"finished": True}

    job = jm.start("predict", fn)
    time.sleep(0.1)
    jm.cancel(job.id)
    _wait(job)
    assert job.status == "cancelled" and job.result is None
    assert job.snapshot()["status"] == "cancelled" and job.events[-1]["status"] == "cancelled"


def test_cancel_kills_the_subprocess_it_runs():
    jm, procs = JobManager(), []

    def fn(progress):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        procs.append(p)
        with track(p):
            p.wait()
            raise_if_cancelled()
        return {}

    job = jm.start("predict", fn)
    while not procs:
        time.sleep(0.02)
    time.sleep(0.2)
    jm.cancel(job.id)
    took = _wait(job)
    assert job.status == "cancelled" and took < 6 and procs[0].poll() is not None


def test_cancel_while_waiting_for_its_turn_leaves_the_queue():
    jm, busy = JobManager(), threading.Lock()
    busy.acquire()                                       # another HiDRA job holds it

    def fn(progress):
        while not busy.acquire(timeout=0.1):
            raise_if_cancelled()
        busy.release()
        return {}

    job = jm.start("predict", fn)
    time.sleep(0.3)
    jm.cancel(job.id)
    _wait(job)
    assert job.status == "cancelled" and busy.locked()   # it never took the lock


def test_a_real_error_is_still_an_error_and_finished_jobs_ignore_cancel():
    jm = JobManager()
    bad = jm.start("predict", lambda progress: 1 / 0)
    _wait(bad)
    assert bad.status == "error"
    ok = jm.start("predict", lambda progress: {"x": 1})
    _wait(ok)
    jm.cancel(ok.id)
    assert ok.status == "done" and ok.result == {"x": 1}
