"""A clip's mouse metadata: age as a P day, genotype, sex, the mice in the cage, cohort, camera, and when
it was recorded. Shown in the clip picker, the header and Clip ▾ → Clip info…, and searched by the clip
filter. Nothing else reads it (exports and the model don't).

Three sources, most trusted first. Every field records which one it came from, in plain words:

  "HCM database"          the clip's recording and the batch it belongs to (hcm_roi's database, read only)
  "clip planner summary"  the re-ID clip planner's clips.summary.json beside the clip (age, day, hour)
  "file name"             tokens in the clip's name: b21, cam_07, 20260906 / 2026-09-06, h16 / 16-00-00,
                          age34, .f128500-143500, or an old-style HCM clip id naming its recording

A clip from another experiment has none of these and gets no fields, so nothing is shown for it.

Best-effort like the arena outlines (hcm_roi): sqlalchemy is imported lazily, any database problem becomes
a plain-language `db_error` and never an exception, and nothing here writes to the database.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone

from . import hcm_roi

DB, SUMMARY, NAME = "HCM database", "clip planner summary", "file name"

# Every field a clip_info can carry, in display order. `mice` is one dict per mouse in the cage.
FIELDS = ("age_days", "dob", "genotype", "sex", "n_mice", "mice", "cohort", "camera", "rig",
          "date", "hour", "frames")
MOUSE_KEYS = ("genotype", "sex", "dob", "treatment", "identity_type", "identity_id")

NO_ANSWER = "the HCM database didn't answer: are you on the Salk network (on site, VPN or Tailscale)?"

# The HCM repo's own expressions (hcm_data.queries dob_sql / sex_sql / age_sql), copied rather than imported
# so the labeler doesn't depend on hcm-core. Cohort date of birth = the earliest animals[].dob; sex =
# 'female', 'male' or 'female/male'; age = days from that DOB to the Pacific date of the recording.
DOB_SQL = "(SELECT MIN((a->>'dob')::date) FROM jsonb_array_elements(b.animals) a)"
SEX_SQL = "(SELECT string_agg(DISTINCT a->>'sex', '/') FROM jsonb_array_elements(b.animals) a)"
AGE_SQL = f"((r.start_time AT TIME ZONE 'America/Los_Angeles')::date - {DOB_SQL})"

# One recording (the clip's own, by hcm_roi's LIKE pattern) and the batch it was stamped into. LEFT JOIN: a
# recording not yet assigned to a batch still says its camera and when it was recorded.
INFO_SQL = f"""
SELECT r.cam AS cam, r.rig AS rig,
       (r.start_time AT TIME ZONE 'America/Los_Angeles')::date AS rec_date,
       EXTRACT(HOUR FROM (r.start_time AT TIME ZONE 'America/Los_Angeles'))::int AS rec_hour,
       b.batch_id AS batch_id, b.cohort_id AS cohort_id, b.experiment AS experiment, b.animals AS animals,
       {DOB_SQL} AS dob, {SEX_SQL} AS sex, {AGE_SQL} AS age_days
FROM recordings r LEFT JOIN batches b ON b.batch_id = r.batch_id
WHERE r.video_path LIKE :pat
ORDER BY r.start_time
LIMIT 1;
"""


# ---------------------------------------------------------------------------------------------------------
# the file name
# ---------------------------------------------------------------------------------------------------------
_TOK = r"(?:^|[^a-z0-9])"                       # a token starts the name or follows a separator
_COHORT_RE = re.compile(_TOK + r"(?:b|batch[_-]?|cohort[_-]?)(\d{1,4})(?![a-z0-9])")
_CAM_RE = re.compile(_TOK + r"cam[_-]?(\d{1,3})(?!\d)")
_AGE_RE = re.compile(_TOK + r"age[_-]?(\d{1,4})(?!\d)")
_HOUR_RE = re.compile(_TOK + r"h(\d{1,2})(?!\d)")
_FRAMES_RE = re.compile(r"(?:^|[._-])f(\d+)-(\d+)(?!\d)")
# an HCM session folder and chunk: 2025_06_16_00_01_05_cam_07.12 (old-style clip ids) or the same with
# dashes / a slash (a recording's own path). Folder = session start, Pacific; chunk = hours since then.
_REC_RE = re.compile(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_/\\]"
                     r"cam_?(\d{1,3})\.(\d{1,2})(?!\d)")
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})([-_]?)(\d{2})\2(\d{2})(?!\d)")
_TIME_AFTER_DATE_RE = re.compile(r"[-_t ](\d{2})([-_:]?)(\d{2})\2(\d{2})(?!\d)")


def _one(values: list):
    """The value when every match agrees, else None: two different cohorts in one name is not a cohort."""
    vals = {v for v in values if v is not None}
    return vals.pop() if len(vals) == 1 else None


def _valid_date(y, m, d) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except (TypeError, ValueError):
        return None


def chunk_start(folder: datetime, chunk: int) -> datetime:
    """Pacific wall-clock start of hour `chunk` of an HCM session that started at `folder` (Pacific), the way
    hcm_data.paths.parse_relpath counts it (in UTC, so a daylight-saving night is right)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Los_Angeles")
        t = folder.replace(tzinfo=tz).astimezone(timezone.utc) + timedelta(hours=chunk)
        return t.astimezone(tz).replace(tzinfo=None)
    except Exception:  # noqa: BLE001  -- no time-zone data here: the sessions start just after midnight anyway
        return folder + timedelta(hours=chunk)


def _from_session(m) -> dict:
    y, mo, d, hh, mi, ss, cam, chunk = m.groups()
    try:
        folder = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss))
    except ValueError:
        return {}
    t = chunk_start(folder, int(chunk))
    return {"camera": f"cam_{int(cam):02d}", "date": t.date().isoformat(), "hour": t.hour}


def parse_name(name: str) -> dict:
    """The fields a clip's name states outright. Conservative: a field is set only when its token clearly
    matches (and, when it appears more than once, every time with the same value); nothing is guessed."""
    s = str(name or "").lower()
    out: dict = {}
    frames = {(int(a), int(b)) for a, b in _FRAMES_RE.findall(s) if int(a) < int(b)}
    if len(frames) == 1:
        out["frames"] = list(frames.pop())
    rest = _FRAMES_RE.sub("_", s)               # frame numbers are not dates or hours
    cohort = _one([int(x) for x in _COHORT_RE.findall(rest)])
    if cohort is not None:
        out["cohort"] = cohort
    cam = _one([f"cam_{int(x):02d}" for x in _CAM_RE.findall(rest)])
    if cam:
        out["camera"] = cam
    age = _one([int(x) for x in _AGE_RE.findall(rest)])
    if age is not None:
        out["age_days"] = age
    sessions = [_from_session(m) for m in _REC_RE.finditer(rest)]
    sessions = [x for x in sessions if x]
    if sessions:                                # an HCM recording named in the clip id: its date and hour
        if all(x == sessions[0] for x in sessions):
            out.update(sessions[0])
        return out
    dates, hours = [], []
    for m in _DATE_RE.finditer(rest):
        d = _valid_date(m.group(1), m.group(3), m.group(4))
        if d is None:
            continue
        dates.append(d.isoformat())
        t = _TIME_AFTER_DATE_RE.match(rest, m.end())
        # a time right after the date is the hour, unless it is an HCM session folder whose chunk number is
        # missing (then the folder time is the session's start, not this clip's hour)
        if t and int(t.group(1)) < 24 and int(t.group(3)) < 60 and int(t.group(4)) < 60 \
                and not re.match(r"[-_]cam", rest[t.end():]):
            hours.append(int(t.group(1)))
    day = _one(dates)
    if day:
        out["date"] = day
    hours += [int(x) for x in _HOUR_RE.findall(rest) if int(x) < 24]
    hour = _one(hours)
    if hour is not None:
        out["hour"] = hour
    return out


def from_recording_path(path) -> dict:
    """Camera, date and hour from an HCM recording's own path (.../2026-09-06-00-01-05/cam_07.16.mp4)."""
    m = _REC_RE.search(str(path or "").lower())
    return _from_session(m) if m else {}


# ---------------------------------------------------------------------------------------------------------
# the clip planner's clips.summary.json
# ---------------------------------------------------------------------------------------------------------
def _int(v) -> int | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return int(f) if f == int(f) else None
    except (TypeError, ValueError):
        return None


def from_summary(entry: dict, cache: dict | None = None) -> tuple[dict, str | None]:
    """(fields, the recording's path) from the clip planner's summary row for this clip: age_days, the day and
    hour, the camera, and the clip's frames within that hour. A clip already linked to its recording
    (`source_video`, read from the same summary when it was added) still has those after its folder moves."""
    hit = hcm_roi.summary_row(entry, cache)
    if hit:
        row, whole_hour = hit
        video = str(row.get("video") or "") or None
        frames = None if whole_hour else [row.get("start"), row.get("end")]
    else:
        row, video, frames = {}, entry.get("source_video"), entry.get("source_frames")
    if not row and not video:
        return {}, None
    out = dict(from_recording_path(video))
    day = str(row.get("day") or "")[:10]
    m = re.fullmatch(r"(\d{4})-?(\d{2})-?(\d{2})", day)
    if m and _valid_date(*m.groups()):
        out["date"] = _valid_date(*m.groups()).isoformat()
    hour = _int(row.get("hour"))
    if hour is not None and 0 <= hour < 24:
        out["hour"] = hour
    age = _int(row.get("age_days"))
    if age is not None and age >= 0:
        out["age_days"] = age
    if frames and len(frames) == 2 and all(_int(x) is not None for x in frames) and _int(frames[0]) < _int(frames[1]):
        out["frames"] = [_int(frames[0]), _int(frames[1])]
    return out, video


# ---------------------------------------------------------------------------------------------------------
# the HCM database
# ---------------------------------------------------------------------------------------------------------
class DBUnavailable(Exception):
    """The database can't be asked (not set up) or didn't answer; the message is the plain reason."""


class DBSession:
    """One connection to the HCM database for a run of lookups (a refresh job over ~200 clips opens one, not
    200). The connection opens at the first lookup, through hcm_roi's engine (its URL handling, NullPool,
    connect timeout), so a run that has nothing to look up never connects. Once the database fails to
    answer, every later lookup in the run fails at once with the same reason instead of waiting again."""

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout
        self.error: str | None = None
        self._conn = None

    def connection(self):
        if self.error:
            raise DBUnavailable(self.error)
        if self._conn is None:
            reason = hcm_roi.setup_reason()
            if reason:
                self.error = reason
                raise DBUnavailable(reason)
            try:
                from sqlalchemy import create_engine
                self._conn = hcm_roi._engine(create_engine, self.timeout).connect()
            except Exception:  # noqa: BLE001  -- any connection failure is "didn't answer"
                self.error = NO_ANSWER
                raise DBUnavailable(NO_ANSWER) from None
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001, S110  -- closing a dead connection
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def query_recording(db: DBSession, pat: str) -> dict | None:
    """The one place the clip-info SQL runs: the recording `pat` names (hcm_roi's LIKE pattern) and its batch,
    as a dict of INFO_SQL's columns, or None when the database has no such recording. Read only (a SELECT)."""
    from sqlalchemy import text
    conn = db.connection()
    try:
        row = conn.execute(text(INFO_SQL), {"pat": pat}).mappings().first()
    except Exception:                   # re-raised below, after the rollback
        try:
            conn.rollback()                 # this clip's row was unreadable; the connection is still good
        except Exception:  # noqa: BLE001
            db.error = NO_ANSWER            # the connection itself is gone
            raise DBUnavailable(NO_ANSWER) from None
        raise
    return dict(row) if row else None


def _s(v) -> str | None:
    """A stored value as display text, or None for a missing/blank one."""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    t = str(v).strip()
    return t or None


def _joined(values) -> str | None:
    vals = sorted({v for v in values if v})
    return "/".join(vals) if vals else None


def from_db_row(row: dict) -> dict:
    """INFO_SQL's row -> clip_info fields."""
    out: dict = {}
    age = _int(row.get("age_days"))
    if age is not None:
        out["age_days"] = age
    if _s(row.get("dob")):
        out["dob"] = _s(row["dob"])
    animals = row.get("animals")
    if isinstance(animals, str):
        try:
            animals = json.loads(animals)
        except ValueError:
            animals = None
    mice = [{k: _s(a.get(k)) for k in MOUSE_KEYS} for a in animals if isinstance(a, dict)] \
        if isinstance(animals, list) else []
    if mice:
        out["mice"] = mice
        out["n_mice"] = len(mice)
    genotype = _joined(m["genotype"] for m in mice) or _s(row.get("experiment"))
    if genotype:
        out["genotype"] = genotype
    sex = _s(row.get("sex")) or _joined(m["sex"] for m in mice)
    if sex:
        out["sex"] = sex
    if _int(row.get("cohort_id")) is not None:
        out["cohort"] = _int(row["cohort_id"])
    cam = _s(row.get("cam"))
    if cam:
        m = re.fullmatch(r"cam[_-]?0*(\d+)", cam.lower())
        out["camera"] = f"cam_{int(m.group(1)):02d}" if m else cam
    if _s(row.get("rig")):
        out["rig"] = _s(row["rig"])
    if _s(row.get("rec_date")):
        out["date"] = _s(row["rec_date"])
    hour = _int(row.get("rec_hour"))
    if hour is not None:
        out["hour"] = hour
    return out


def _db_problem(e: Exception) -> str:
    detail = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
    return f"the HCM database answered, but this clip's recording couldn't be read ({detail})"


# ---------------------------------------------------------------------------------------------------------
# putting it together
# ---------------------------------------------------------------------------------------------------------
def merge(*layers: tuple[str, dict]) -> tuple[dict, dict]:
    """(fields, sources) from (source name, fields) layers, most trusted first: each field from the first
    layer that has it."""
    fields, sources = {}, {}
    for src, got in layers:
        for k in FIELDS:
            if k not in fields and got.get(k) is not None:
                fields[k], sources[k] = got[k], src
    return fields, sources


def compute(entry: dict, db: DBSession | None = None, cache: dict | None = None) -> dict:
    """The clip_info to cache on a manifest entry: its fields, `sources` (field -> where it came from),
    `fetched_at`, `db_error` (why the database couldn't give this clip's details, or None), `db_found` (True:
    the database has the clip's recording; False: it was asked and has none; None: not asked), and
    `recording` (the recording the clip was matched to). With `db=None` the database is not asked and
    `fetched_at` stays None: a stand-in until a refresh runs."""
    vid = entry.get("video_id") or ""
    name = parse_name(vid)
    summ, video = from_summary(entry, cache)
    pat = hcm_roi.recording_like(vid) or hcm_roi.recording_like_path(entry.get("source_video") or video)
    dbf, db_error, db_found = {}, None, None
    if db is not None and pat:
        try:
            row = query_recording(db, pat)
        except DBUnavailable as e:
            db_error = str(e)
        except Exception as e:  # noqa: BLE001
            db_error = _db_problem(e)
        else:
            db_found = row is not None
            dbf = from_db_row(row) if row else {}
    fields, sources = merge((DB, dbf), (SUMMARY, summ), (NAME, name))
    return {**fields, "sources": sources,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds") if db is not None else None,
            "db_error": db_error, "db_found": db_found, "recording": pat.lstrip("%/") if pat else None}


def name_only(entry: dict) -> dict:
    """What the clip's name alone says, with no file or database read: what a project lists for a clip that
    has no cached clip_info yet (fetched_at None marks it as not fetched)."""
    vid = entry.get("video_id") or ""
    fields, sources = merge((NAME, parse_name(vid)))
    pat = hcm_roi.recording_like(vid)
    return {**fields, "sources": sources, "fetched_at": None, "db_error": None, "db_found": None,
            "recording": pat.lstrip("%/") if pat else None}


def has_info(entry: dict) -> bool:
    """A refresh has run for this clip (whatever it found)."""
    return bool((entry.get("clip_info") or {}).get("fetched_at"))
