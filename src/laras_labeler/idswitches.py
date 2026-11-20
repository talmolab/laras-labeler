"""Tracking identity switches: frames where a track jumps from one animal to another, marked while
watching a clip and listed for whoever fixes the tracks (SLEAP proofreading, a re-ID tool).

One list for the whole projects folder, not per project: a switch is a fact about a clip's tracking
file, and one clip can be in several projects. Each entry keeps a snapshot of what the export needs
(the clip's files, frame rate, track names, and where an HCM clip was cut from), so the list still
exports when the project it was logged in, or the clip, is gone.

    <projects root>/id_switches.json    {"version": 1, "switches": [entry, ...]}

Tracks are 0-based indices into the clip's pose array, the GUI's track 1, 2, ...; `track_names` are
those tracks' names in the .slp, as SLEAP shows them. An empty `tracks` means "not sure which".
"""

from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .project import write_json_atomic

FILE = "id_switches.json"
STATUSES = ("open", "fixed")
MAX_NOTE = 2000

# The clip-info fields copied onto an entry when it is logged (clip_info.py).
INFO_FIELDS = ("recording", "cohort", "camera", "date", "hour")

CSV_COLS = ["clip", "frame", "time_s", "tracks", "sleap_tracks", "note", "status", "logged",
            "video_file", "tracking_file", "source_recording", "source_frame", "cohort", "camera",
            "recording_date", "recording_hour"]


class UnreadableList(Exception):
    """id_switches.json exists but can't be read as a list of switches. It is never overwritten."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def clean_frame(v, n_frames) -> int:
    if not _is_int(v):
        raise ValueError("frame must be a whole number")
    hi = int(n_frames) if n_frames else None
    if v < 0 or (hi is not None and v >= hi):
        raise ValueError(f"frame {v} is outside this clip (0 to {hi - 1})" if hi else f"frame {v} can't be negative")
    return v


def clean_tracks(v, n_tracks) -> list[int]:
    """Sorted, distinct 0-based track indices; [] (or a missing value) means not sure which."""
    if v is None:
        return []
    if not isinstance(v, list) or not all(_is_int(t) for t in v):
        raise ValueError("tracks must be a list of track indices (0 = track 1)")
    for t in v:
        if t < 0 or (n_tracks is not None and t >= n_tracks):
            raise ValueError(f"track index {t} is not in this clip (it has {n_tracks} tracks: 0 to {n_tracks - 1})"
                             if n_tracks is not None else f"track index {t} can't be negative")
    return sorted(set(v))


def clean_note(v) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ValueError("note must be text")
    v = v.strip()
    if len(v) > MAX_NOTE:
        raise ValueError(f"note is longer than {MAX_NOTE} characters")
    return v


def clean_status(v) -> str:
    if v not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    return v


def names_of(tracks: list[int], clip_names: list[str]) -> list[str]:
    return [clip_names[t] if t < len(clip_names) else "" for t in tracks]


def tracks_text(values) -> str:
    """Tracks joined the way the GUI shows them: [0, 2] -> "1 ↔ 3"; names the same way."""
    return " ↔ ".join(str(v) for v in values)


def new_entry(pid: str, clip: dict, info: dict, frame: int, tracks: list[int], note: str,
              n_tracks: int, clip_names: list[str]) -> dict:
    """An entry for a switch at `frame` of the clip `clip` (its manifest entry in project `pid`), with the
    snapshot the export needs. `info` is the clip's clip_info; its `frames` (the clip's span in its source
    recording) give the source frame."""
    span = info.get("frames")
    source_frame = (int(span[0]) + frame if isinstance(span, (list, tuple)) and len(span) == 2
                    and _is_int(span[0]) else None)
    fps = clip.get("fps")
    return {
        "id": uuid.uuid4().hex[:12],
        "video_id": clip["video_id"],
        "frame": frame,
        "tracks": tracks,
        "track_names": names_of(tracks, clip_names),
        "note": note,
        "status": "open",
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": pid,
        "video_path": clip.get("video_path"),
        "slp_path": clip.get("slp_path"),
        "fps": float(fps) if fps else None,
        "n_frames": clip.get("n_frames"),
        "n_tracks": n_tracks,
        "clip_track_names": list(clip_names),
        **{k: info.get(k) for k in INFO_FIELDS},
        "source_frame": source_frame,
    }


def same_clip(s: dict, video_id: str | None, n_frames: int | None = None) -> bool:
    """`s` is on that clip: the same clip name and, when both are known, the same length (another
    experiment's clip can have the same name)."""
    if video_id is not None and s.get("video_id") != video_id:
        return False
    return n_frames is None or not s.get("n_frames") or s.get("n_frames") == n_frames


def _order(s: dict):
    return (str(s.get("video_id") or ""), s.get("frame") if _is_int(s.get("frame")) else -1, str(s.get("logged_at") or ""))


class IdSwitchStore:
    """The list on disk. Every call reads the file afresh, so what another labeler on the same folder wrote
    is kept; writes are atomic (a temp file renamed over) and one at a time."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / FILE
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as e:
            raise UnreadableList(f"{self.path} can't be read: {e}") from None
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise UnreadableList(f"{self.path} isn't valid JSON ({e}). It is left as it is: fix it or move it "
                                 "aside to start a new list.") from None
        switches = data.get("switches") if isinstance(data, dict) else None
        if not isinstance(switches, list):
            raise UnreadableList(f"{self.path} has no list of switches. It is left as it is: fix it or move it "
                                 "aside to start a new list.")
        return [s for s in switches if isinstance(s, dict) and s.get("id")]

    def _write(self, switches: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, {"version": 1, "switches": sorted(switches, key=_order)})

    def list(self, video_id: str | None = None, n_frames: int | None = None) -> list[dict]:
        """Every switch, or one clip's (`same_clip`), sorted by clip, then frame."""
        with self._lock:
            items = self._read()
        return sorted((s for s in items if same_clip(s, video_id, n_frames)), key=_order)

    def add(self, entry: dict) -> dict:
        with self._lock:
            items = self._read()
            items.append(entry)
            self._write(items)
        return entry

    def edit(self, sid: str, changes: dict) -> dict:
        """Change a switch's `tracks`, `note` and/or `status`. KeyError for an unknown id, ValueError for a bad
        value (tracks are checked against the clip as it was when the switch was logged)."""
        if not isinstance(changes, dict) or not any(k in changes for k in ("tracks", "note", "status")):
            raise ValueError("nothing to change: send tracks, note or status")
        with self._lock:
            items = self._read()
            s = next((x for x in items if x.get("id") == sid), None)
            if s is None:
                raise KeyError(sid)
            patch = {}
            if "tracks" in changes:
                clip_names = list(s.get("clip_track_names") or [])
                n = s.get("n_tracks") if _is_int(s.get("n_tracks")) else (len(clip_names) or None)
                patch["tracks"] = clean_tracks(changes["tracks"], n)
                patch["track_names"] = names_of(patch["tracks"], clip_names)
            if "note" in changes:
                patch["note"] = clean_note(changes["note"])
            if "status" in changes:
                patch["status"] = clean_status(changes["status"])
            s.update(patch)
            self._write(items)
            return dict(s)

    def delete(self, sid: str) -> None:
        with self._lock:
            items = self._read()
            keep = [x for x in items if x.get("id") != sid]
            if len(keep) == len(items):
                raise KeyError(sid)
            self._write(keep)


def _local_time(iso) -> str:
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return ""


def _cell(v):
    return "" if v is None else v


def csv_row(s: dict) -> dict:
    fps, frame, names = s.get("fps"), s.get("frame"), s.get("track_names") or []
    return {
        "clip": s.get("video_id") or "",
        "frame": _cell(frame),
        "time_s": f"{frame / fps:.2f}" if fps and _is_int(frame) else "",
        "tracks": tracks_text(t + 1 for t in s.get("tracks") or []),
        "sleap_tracks": tracks_text(n or "?" for n in names) if any(names) else "",
        "note": s.get("note") or "",
        "status": s.get("status") or "open",
        "logged": _local_time(s.get("logged_at")),
        "video_file": s.get("video_path") or "",
        "tracking_file": s.get("slp_path") or "",
        "source_recording": s.get("recording") or "",
        "source_frame": _cell(s.get("source_frame")),
        "cohort": _cell(s.get("cohort")),
        "camera": s.get("camera") or "",
        "recording_date": s.get("date") or "",
        "recording_hour": _cell(s.get("hour")),
    }


def to_csv(switches: list[dict]) -> str:
    """The list as a CSV for the person fixing the tracks: one row per switch, sorted by clip, then frame;
    a blank cell where something isn't known."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(csv_row(s) for s in sorted(switches, key=_order))
    return buf.getvalue()
