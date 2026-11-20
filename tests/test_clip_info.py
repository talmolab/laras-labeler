"""Clip info (clip_info.py): a clip's age, genotype, sex, cage, cohort, camera and recording time, from the
HCM database, the clip planner's clips.summary.json, or the clip's name -- in that order of trust.
No real database and no media files: the one function that runs the query is replaced.

    python -m pytest tests/test_clip_info.py
"""
import json
import tempfile
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from laras_labeler import clip_info, hcm_roi
from laras_labeler.app import create_app
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore

REID = "b21_20260906_h16.f128500-143500"
OLD = "clip_001_2025_02_12_2025_09_02_00_01_05_cam_06.11.predictions.slp_batch_10_f0-15000"
REC = "/mnt/vast/hcm/cam_07/2026-09-06-00-01-05/cam_07.16.mp4"


# ---- the file name ---------------------------------------------------------------------------------------
def test_name_reid_clip():
    assert clip_info.parse_name(REID) == {"cohort": 21, "date": "2026-09-06", "hour": 16, "frames": [128500, 143500]}


def test_name_old_style_hcm_clip():
    # the recording folder is the session start (Pacific) and .11 is hours since then; the leading 2025_02_12
    # is not the recording date, and batch_10 is the cohort
    got = clip_info.parse_name(OLD)
    assert got == {"cohort": 10, "camera": "cam_06", "date": "2025-09-02", "hour": 11, "frames": [0, 15000]}


def test_name_age_token_and_camera_spellings():
    assert clip_info.parse_name("clip_age120_cam07") == {"age_days": 120, "camera": "cam_07"}
    assert clip_info.parse_name("cam_07_2026-09-06_16-00-00") == {"camera": "cam_07", "date": "2026-09-06", "hour": 16}


@pytest.mark.parametrize("name", ["synth", "clip_00", "mouse_video_final", "stage3_page12", "h264_video",
                                  "b21_b22", "x_20261103_20261104", "", "2026-13-45",
                                  # other labs' names: B6 is a strain, h1 a habituation day, age12wk is weeks
                                  "B6_male_day3", "b6-male-cage4", "mouse_b6_2021", "DLC_resnet50_mouse_b2", "A2-B3-C4",
                                  "OF_h1_mouse2", "S1_H1_R1", "block_b1_trial_h3", "m3_age12wk", "age_12_weeks",
                                  "age3mo", "b21_20261345_h16"])
def test_name_with_nothing_clear_sets_nothing(name):
    # no token, or tokens that disagree (two cohorts, two dates), or an impossible date: no field is guessed
    assert clip_info.parse_name(name) == {}


def test_bare_cohort_and_hour_only_in_planner_names():
    assert clip_info.parse_name("b21_20260906_h16") == {"cohort": 21, "date": "2026-09-06", "hour": 16}
    assert clip_info.parse_name("B6_20210305_mouse1") == {"date": "2021-03-05"}          # a date, but no cohort
    assert clip_info.parse_name("batch2_mouse3") == {"cohort": 2}                         # spelled out
    assert clip_info.parse_name("m3_age34d") == {"age_days": 34}


def test_session_chunk_counts_hours_across_daylight_saving():
    # 1 Nov 2026 the clocks go back at 02:00: three hours after 00:01 PDT is 02:01 PST, not 03:01
    t = clip_info.chunk_start(datetime(2026, 11, 1, 0, 1, 5), 3)
    assert (t.date(), t.hour) == (date(2026, 11, 1), 2)


# ---- the clip planner's summary ----------------------------------------------------------------------------
def _reid_folder(age=34):
    d = Path(tempfile.mkdtemp())
    rows = [{"bin": 0, "stem": "b21_20260906_h16", "day": "2026-09-06", "hour": 16, "age_days": age,
             "start": 128500, "end": 143500, "video": REC,
             "clip_video": str(d / f"{REID}.mp4"), "clip_slp": str(d / f"{REID}.reid.slp")}]
    (d / hcm_roi.SUMMARY).write_text(json.dumps(rows))
    return d


def _entry(d, vid=REID):
    return {"video_id": vid, "video_path": str(d / f"{vid}.mp4"), "slp_path": None}


def test_summary_gives_age_day_hour_camera():
    d = _reid_folder()
    fields, video = clip_info.from_summary(_entry(d))
    assert video == REC
    assert fields == {"age_days": 34, "date": "2026-09-06", "hour": 16, "camera": "cam_07", "frames": [128500, 143500]}
    # the ROI lookup reads the same row through the shared helper
    assert hcm_roi.source_recording(_entry(d)) == {"video": REC, "frames": [128500, 143500]}


def test_offline_info_merges_summary_over_name():
    d = _reid_folder()
    info = clip_info.compute(_entry(d), None)
    assert info["fetched_at"] is None and info["db_error"] is None and info["db_found"] is None
    assert info["age_days"] == 34 and info["sources"]["age_days"] == "clip planner summary"
    assert info["cohort"] == 21 and info["sources"]["cohort"] == "file name"
    assert info["recording"] == "2026-09-06-00-01-05/cam_07.16.mp4"


# ---- the database ------------------------------------------------------------------------------------------
def _row(**kw):
    r = {"cam": "cam_07", "rig": "hyena", "rec_date": date(2026, 9, 6), "rec_hour": 16, "batch_id": "u-1",
         "cohort_id": 21, "experiment": "SOD1", "dob": date(2026, 8, 3), "sex": "female", "age_days": 34,
         "animals": [{"identity_type": None, "identity_id": None, "genotype": "SOD1", "treatment": None,
                      "dob": "2026-08-03", "sex": "female"} for _ in range(3)]}
    r.update(kw)
    return r


def test_database_wins_over_summary_and_name(monkeypatch):
    d = _reid_folder(age=33)                       # the summary disagrees on age: the database wins
    asked = []
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: asked.append(pat) or _row())
    info = clip_info.compute(_entry(d), clip_info.DBSession())
    assert asked == ["%/2026-09-06-00-01-05/cam_07.16.mp4"]
    assert info["age_days"] == 34 and info["dob"] == "2026-08-03"
    assert (info["genotype"], info["sex"], info["n_mice"], info["cohort"]) == ("SOD1", "female", 3, 21)
    assert (info["camera"], info["rig"], info["date"], info["hour"]) == ("cam_07", "hyena", "2026-09-06", 16)
    src = info["sources"]
    assert {k: src[k] for k in ("age_days", "dob", "genotype", "sex", "n_mice", "mice", "cohort", "camera", "date", "hour")} \
        == dict.fromkeys(("age_days", "dob", "genotype", "sex", "n_mice", "mice", "cohort", "camera", "date", "hour"), "HCM database")
    assert src["frames"] == "clip planner summary"          # only the planner knows where in the hour the clip is
    assert info["db_found"] is True and info["db_error"] is None and info["fetched_at"]
    assert info["mice"][0] == {"genotype": "SOD1", "sex": "female", "dob": "2026-08-03", "treatment": None,
                               "identity_type": None, "identity_id": None}


def test_age_is_the_pacific_recording_date_minus_the_cohort_dob():
    # the age and the date come from the database's own expressions (hcm_data.queries), so a recording at
    # 23:30 UTC -- still the previous day in Los Angeles -- is dated and aged by its Pacific day
    sql = " ".join(clip_info.INFO_SQL.split())
    assert "(r.start_time AT TIME ZONE 'America/Los_Angeles')::date AS rec_date" in sql
    assert "EXTRACT(HOUR FROM (r.start_time AT TIME ZONE 'America/Los_Angeles'))::int AS rec_hour" in sql
    assert clip_info.AGE_SQL == ("((r.start_time AT TIME ZONE 'America/Los_Angeles')::date - "
                                 "(SELECT MIN((a->>'dob')::date) FROM jsonb_array_elements(b.animals) a))")
    assert f"{clip_info.AGE_SQL} AS age_days" in sql and f"{clip_info.DOB_SQL} AS dob" in sql
    assert "JOIN batches b ON b.batch_id = r.batch_id" in sql and "WHERE r.video_path LIKE :pat" in sql
    assert sql.lstrip().upper().startswith("SELECT")          # read only
    got = clip_info.from_db_row(_row(rec_date=date(2026, 9, 5), rec_hour=16, age_days=33))
    assert (got["date"], got["hour"], got["age_days"]) == ("2026-09-05", 16, 33)


def test_mixed_cage_is_summarised():
    animals = [{"genotype": "SOD1", "sex": "female", "dob": "2026-08-03", "treatment": "vehicle",
                "identity_type": "ear tag", "identity_id": "L1"},
               {"genotype": "WT", "sex": "male", "dob": "2026-08-05", "treatment": "", "identity_type": None, "identity_id": None}]
    got = clip_info.from_db_row(_row(animals=animals, sex=None, experiment="SOD1"))
    assert (got["genotype"], got["sex"], got["n_mice"]) == ("SOD1/WT", "female/male", 2)
    assert got["mice"][0]["identity_id"] == "L1" and got["mice"][1]["treatment"] is None   # blank -> not recorded
    # no genotype on any mouse: the batch's experiment says it
    got = clip_info.from_db_row(_row(animals=[{"sex": "male"}], experiment="WT"))
    assert got["genotype"] == "WT"
    # a recording not yet in a batch still has its camera and time, and nothing about mice
    got = clip_info.from_db_row(_row(batch_id=None, cohort_id=None, experiment=None, animals=None, dob=None, sex=None, age_days=None))
    assert got == {"camera": "cam_07", "rig": "hyena", "date": "2026-09-06", "hour": 16}


def test_database_unreachable_keeps_summary_and_name(monkeypatch):
    d = _reid_folder()

    def down(db, pat):
        raise clip_info.DBUnavailable(clip_info.NO_ANSWER)
    monkeypatch.setattr(clip_info, "query_recording", down)
    info = clip_info.compute(_entry(d), clip_info.DBSession())
    assert info["db_error"] == clip_info.NO_ANSWER and info["db_found"] is None and info["fetched_at"]
    assert info["age_days"] == 34 and info["sources"]["age_days"] == "clip planner summary"
    assert info["cohort"] == 21 and info["sources"]["cohort"] == "file name"


def test_unreadable_row_is_a_plain_reason(monkeypatch):
    def bad(db, pat):
        raise ValueError('invalid input syntax for type date: ""')
    monkeypatch.setattr(clip_info, "query_recording", bad)
    info = clip_info.compute({"video_id": OLD, "video_path": "/x/a.mp4"}, clip_info.DBSession())
    assert info["db_error"].startswith("the HCM database answered, but this clip's recording couldn't be read")
    assert info["cohort"] == 10 and info["sources"]["cohort"] == "file name"


def test_no_database_url_and_no_answer_reasons(monkeypatch):
    pytest.importorskip("sqlalchemy")
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "")
    with clip_info.DBSession() as db:
        info = clip_info.compute({"video_id": OLD, "video_path": "/x/a.mp4"}, db)
    assert "HCM_DATABASE_URL isn't set" in info["db_error"]
    if any(hcm_roi._importable(m) for m in ("psycopg", "psycopg2")):
        monkeypatch.setattr(hcm_roi, "DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")   # nothing listens there
        with clip_info.DBSession(timeout=2) as db:
            a = clip_info.compute({"video_id": OLD, "video_path": "/x/a.mp4"}, db)
            b = clip_info.compute({"video_id": OLD.replace("cam_06.11", "cam_06.12"), "video_path": "/x/b.mp4"}, db)
        assert a["db_error"] == b["db_error"] == clip_info.NO_ANSWER


def test_clip_from_another_experiment_asks_nothing(monkeypatch):
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: pytest.fail("asked the database"))
    info = clip_info.compute({"video_id": "mouse_video_final", "video_path": "/nowhere/mouse_video_final.mp4"},
                             clip_info.DBSession())
    assert {k: v for k, v in info.items() if k != "fetched_at"} == \
        {"sources": {}, "db_error": None, "db_retry": False, "db_found": None, "db_fetched_at": None, "recording": None}


def test_unreachable_refresh_keeps_the_last_database_details(monkeypatch):
    # a clip looked up on the Salk network, then refreshed off it: genotype, sex, the cage and the database's age
    # stay (still "HCM database", with when they were read); the new lookup's reason is shown and retried later
    d = _reid_folder(age=33)
    entry = _entry(d)
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: _row())
    entry["clip_info"] = first = clip_info.compute(entry, clip_info.DBSession())
    assert first["db_fetched_at"] == first["fetched_at"] and clip_info.has_info(entry)

    def down(db, pat):
        raise clip_info.DBUnavailable(clip_info.NO_ANSWER)
    monkeypatch.setattr(clip_info, "query_recording", down)
    info = clip_info.compute(entry, clip_info.DBSession())
    assert (info["genotype"], info["sex"], info["n_mice"], info["age_days"]) == ("SOD1", "female", 3, 34)
    assert {info["sources"][k] for k in ("genotype", "sex", "mice", "age_days", "cohort")} == {"HCM database"}
    assert info["db_fetched_at"] == first["db_fetched_at"] and info["db_found"] is True
    assert info["db_error"] == clip_info.NO_ANSWER and info["db_retry"] is True
    entry["clip_info"] = info
    assert not clip_info.has_info(entry)                    # the next automatic refresh asks again
    # the database answering "no such recording" is an answer: the old details go
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: None)
    info = clip_info.compute(entry, clip_info.DBSession())
    assert info["db_found"] is False and "genotype" not in info and info["sources"]["age_days"] == "clip planner summary"


class _Result(list):
    def mappings(self):
        return self


class _Conn:
    """A stand-in for a SQLAlchemy connection: SELECT number `fail_on` raises `error`; every SELECT returns `rows`."""

    def __init__(self, rows, fail_on=None, error=None):
        self.rows, self.fail_on, self.error = rows, fail_on, error
        self.selects, self.sets, self.rollbacks, self.closed, self.invalidated = 0, [], 0, False, False

    def execute(self, stmt, params=None):
        sql = str(stmt).strip()
        if sql.startswith("SET"):
            self.sets.append(sql)
            return _Result()
        self.selects += 1
        if self.selects == self.fail_on:
            raise self.error
        return _Result(dict(r) for r in self.rows)

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _db_with(conn):
    db = clip_info.DBSession()
    db._conn = conn
    return db


class QueryCanceled(Exception):
    pass


@pytest.mark.parametrize("why, error", [
    ("lost", lambda e: e.OperationalError("SELECT", {}, Exception("consuming input failed"), connection_invalidated=True)),
    ("killed", lambda e: e.OperationalError("SELECT", {}, Exception("terminating connection due to administrator command"))),
    ("slow", lambda e: e.OperationalError("SELECT", {}, QueryCanceled("canceling statement due to statement timeout"))),
])
def test_database_lost_mid_run_ends_the_run_at_once(why, error):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    conn = _Conn([_row()], fail_on=2, error=error(sqlalchemy.exc))
    db = _db_with(conn)
    infos = [clip_info.compute({"video_id": _old(i), "video_path": "/x/a.mp4"}, db) for i in range(1, 5)]
    assert infos[0]["db_found"] is True and infos[0]["db_error"] is None
    reason = clip_info.TOO_SLOW if why == "slow" else clip_info.NO_ANSWER
    assert [x["db_error"] for x in infos[1:]] == [reason] * 3 and all(x["db_retry"] for x in infos[1:])
    assert conn.selects == 2 and conn.closed            # clips 3 and 4 never wait on a dead connection
    # each lookup is its own read-only transaction with a statement timeout, ended right after
    assert conn.sets[:2] == ["SET TRANSACTION READ ONLY", f"SET LOCAL statement_timeout = {clip_info.STATEMENT_TIMEOUT_MS}"]
    assert conn.rollbacks >= 1


def test_unreadable_row_leaves_the_connection_for_the_next_clip():
    sqlalchemy = pytest.importorskip("sqlalchemy")
    conn = _Conn([_row()], fail_on=1, error=sqlalchemy.exc.DataError("SELECT", {}, Exception('invalid input syntax for type date: ""')))
    db = _db_with(conn)
    a = clip_info.compute({"video_id": _old(1), "video_path": "/x/a.mp4"}, db)
    b = clip_info.compute({"video_id": _old(2), "video_path": "/x/b.mp4"}, db)
    assert a["db_error"].startswith("the HCM database answered") and a["db_retry"] is False
    assert b["db_found"] is True and b["db_error"] is None and db.error is None


def test_two_cages_matching_one_clip_are_not_guessed():
    # cam names are only unique within a rig: hyena/cam_07 and rig00/cam_07 can share a session folder name
    rows = [_row(), _row(rig="rig00", batch_id="u-2", cohort_id=40, experiment="WT", sex="male",
                     animals=[{"genotype": "WT", "sex": "male", "dob": "2026-07-01"}])]
    d = _reid_folder()
    info = clip_info.compute(_entry(d), _db_with(_Conn(rows)))       # b21...: the name says cohort 21
    assert info["cohort"] == 21 and info["genotype"] == "SOD1" and info["rig"] == "hyena" and info["db_error"] is None
    entry = {"video_id": _old(1), "video_path": "/x/a.mp4",               # ..._batch_10_...: neither cage
             "clip_info": {"genotype": "SOD1", "sources": {"genotype": "HCM database"}, "fetched_at": "2026-11-01T00:00:00+00:00"}}
    info = clip_info.compute(entry, _db_with(_Conn(rows)))
    assert "2 HCM recordings match" in info["db_error"] and "hyena/cam_07" in info["db_error"]
    assert "genotype" not in info and info["sources"]["cohort"] == "file name" and info["db_retry"] is False


# ---- the app: project GET, GET info, the refresh job --------------------------------------------------------
def _client(vids):
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    c = TestClient(create_app(Settings(projects_root=root), store))
    pid = c.post("/api/projects", json={"name": "info"}).json()["pid"]
    p = store.get(pid)
    for vid in vids:
        p.manifest.setdefault("videos", []).append({"video_id": vid, "video_path": f"/x/{vid}.mp4", "slp_path": None,
                                                    "n_frames": 100, "fps": 50.0, "width": 10, "height": 10,
                                                    "has_poses": False})
    p.save()
    return c, store, pid


def _wait(c, pid, jid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/projects/{pid}/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise TimeoutError


def _old(i):
    return OLD.replace("cam_06.11", f"cam_06.{i:02d}").replace("clip_001", f"clip_{i:03d}")


def test_project_get_carries_clip_info_and_info_works_without_cache():
    c, _, pid = _client([REID, "plain"])
    vids = {v["video_id"]: v for v in c.get(f"/api/projects/{pid}").json()["videos"]}
    assert vids[REID]["clip_info"]["cohort"] == 21 and vids[REID]["clip_info"]["fetched_at"] is None
    assert vids["plain"]["clip_info"]["sources"] == {}
    info = c.get(f"/api/projects/{pid}/videos/{REID}/info").json()
    assert info["hour"] == 16 and info["sources"]["hour"] == "file name" and info["fetched_at"] is None
    assert c.get(f"/api/projects/{pid}/videos/nope/info").status_code == 404


def test_refresh_job_missing_only_and_results(monkeypatch):
    rows = {f"%/2025-09-02-00-01-05/cam_06.{i:02d}.mp4": _row(age_days=30 + i, rec_hour=i) for i in (1, 2)}
    asked = []
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: asked.append(pat) or rows.get(pat))
    c, store, pid = _client([_old(1), _old(2), _old(3), "plain"])
    kept = {"age_days": 99, "sources": {"age_days": "HCM database"}, "fetched_at": "2026-11-01T00:00:00+00:00",
            "db_error": None, "db_found": True, "recording": None}
    store.get(pid).video(_old(2))["clip_info"] = dict(kept)
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"videos": None, "missing_only": True}).json()
    assert r["clips"] == 3 and r["job_id"]
    j = _wait(c, pid, r["job_id"])
    assert j["status"] == "done"
    assert j["result"] == {"clips": 3, "fetched": 3, "asked": 2, "from_db": 1, "no_recording": 1, "db_errors": 0,
                           "db_error": None}
    assert sorted(asked) == sorted(["%/2025-09-02-00-01-05/cam_06.01.mp4", "%/2025-09-02-00-01-05/cam_06.03.mp4"])
    saved = {v["video_id"]: v.get("clip_info") for v in json.loads((store.get(pid).path / "project.json").read_text())["videos"]}
    assert saved[_old(1)]["age_days"] == 31 and saved[_old(1)]["sources"]["age_days"] == "HCM database"
    assert saved[_old(2)] == kept                                      # it had info: skipped
    assert saved[_old(3)]["db_found"] is False and saved[_old(3)]["sources"]["cohort"] == "file name"
    assert saved["plain"]["sources"] == {} and saved["plain"]["fetched_at"]
    got = c.get(f"/api/projects/{pid}/clip-info").json()["videos"]
    assert got[_old(1)]["age_days"] == 31
    # everything has info now: nothing to do
    assert c.post(f"/api/projects/{pid}/clip-info/refresh", json={"missing_only": True}).json()["job_id"] is None
    # one clip, on request, even though it has info
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"videos": [_old(2)]}).json()
    assert r["clips"] == 1 and _wait(c, pid, r["job_id"])["status"] == "done"
    assert store.get(pid).video(_old(2))["clip_info"]["age_days"] == 32
    assert c.post(f"/api/projects/{pid}/clip-info/refresh", json={"videos": ["nope"]}).status_code == 404


def test_refresh_job_unreachable_database(monkeypatch):
    calls = []

    def down(db, pat):
        calls.append(pat)
        db.error = clip_info.NO_ANSWER
        raise clip_info.DBUnavailable(clip_info.NO_ANSWER)
    monkeypatch.setattr(clip_info, "query_recording", down)
    c, store, pid = _client([_old(1), _old(2)])
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={}).json()
    j = _wait(c, pid, r["job_id"])
    assert j["status"] == "done" and j["result"]["db_errors"] == 2 and j["result"]["db_error"] == clip_info.NO_ANSWER
    assert j["result"]["from_db"] == 0
    info = store.get(pid).video(_old(1))["clip_info"]
    assert info["db_error"] == clip_info.NO_ANSWER and info["cohort"] == 10 and info["hour"] == 1
    # the next automatic refresh (missing_only, as on project open) asks again, and fills them once it answers
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: _row())
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"missing_only": True}).json()
    assert r["clips"] == 2 and _wait(c, pid, r["job_id"])["result"]["from_db"] == 2
    assert store.get(pid).video(_old(1))["clip_info"]["genotype"] == "SOD1"
    assert c.post(f"/api/projects/{pid}/clip-info/refresh", json={"missing_only": True}).json()["job_id"] is None


def _gated(after: int):
    """A query_recording that answers, but holds lookup number `after` until released: (fake, reached, release)."""
    reached, release, calls = threading.Event(), threading.Event(), []

    def fake(db, pat):
        calls.append(pat)
        if len(calls) == after:
            reached.set()
            assert release.wait(10)
        return _row()
    return fake, reached, release


def test_refresh_job_one_at_a_time_and_stop_keeps_what_was_fetched(monkeypatch):
    fake, reached, release = _gated(3)
    monkeypatch.setattr(clip_info, "query_recording", fake)
    vids = [_old(i) for i in range(1, 21)]
    c, store, pid = _client(vids)
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"videos": None, "missing_only": False}).json()
    again = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"videos": None}).json()
    assert again["job_id"] == r["job_id"] and again["already_running"] is True
    assert reached.wait(10)                              # the job is on its 3rd clip
    assert c.post(f"/api/projects/{pid}/jobs/{r['job_id']}/cancel").status_code == 200
    release.set()
    j = _wait(c, pid, r["job_id"])
    assert j["status"] == "cancelled"
    saved = [v.get("clip_info") for v in json.loads((store.get(pid).path / "project.json").read_text())["videos"]]
    n = sum(1 for s in saved if s)
    assert n == 3                                        # the 3rd finished; Stop landed before the 4th
    assert all(s["sources"]["age_days"] == "HCM database" for s in saved if s)
    # a new run is allowed once the stopped one has ended, and picks up only the rest
    monkeypatch.setattr(clip_info, "query_recording", lambda db, pat: _row())
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={"missing_only": True}).json()
    assert r["clips"] == len(vids) - n and _wait(c, pid, r["job_id"])["status"] == "done"


def test_clip_added_during_a_refresh_joins_it(monkeypatch):
    # a clip added while a refresh runs is looked up by that refresh (not left for the next project open)
    fake, reached, release = _gated(1)
    monkeypatch.setattr(clip_info, "query_recording", fake)
    c, store, pid = _client([_old(1), _old(2)])
    r = c.post(f"/api/projects/{pid}/clip-info/refresh", json={}).json()
    assert reached.wait(10)
    p = store.get(pid)

    def add_video(video_path, slp_path=None):        # no media here: just the manifest entry the real one writes
        vid = Path(video_path).stem
        entry = {"video_id": vid, "video_path": video_path, "slp_path": None, "n_frames": 100, "fps": 50.0,
                 "width": 10, "height": 10, "has_poses": False}
        p.videos.append(entry)
        return entry
    monkeypatch.setattr(p, "add_video", add_video)
    new = _old(3)
    assert c.post(f"/api/projects/{pid}/videos", json={"video_path": f"/x/{new}.mp4"}).status_code == 200
    assert p.video(new)["clip_info"]["fetched_at"] is None and p.video(new)["clip_info"]["cohort"] == 10
    release.set()
    j = _wait(c, pid, r["job_id"])
    assert j["status"] == "done" and j["result"]["clips"] == 3 and j["result"]["from_db"] == 3
    assert p.video(new)["clip_info"]["genotype"] == "SOD1" and p.video(new)["clip_info"]["fetched_at"]
