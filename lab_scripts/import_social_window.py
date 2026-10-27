# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl"]
# ///
"""Set a laras-labeler clip's social window (the frame range the second rat is actually in the
arena) from Hannah's QC spreadsheet, via the running labeler's HTTP API.

Only the main `asd_session_qc_social` sheet is used. Its entry_time and exit_time are
MINUTES:SECONDS written as a number, so 9.41 means 9 min 41 s (and 10.4 means 10 min 40 s). They
are not decimal minutes. Frames = (minutes x 60 + seconds) x fps. At 50 fps this lands exactly on
the `partner_present` record in ugne_clusters for every session checked (2026-10-23).
`Sheet1` is ignored. Its start_frame and end_frame are empty for most sessions.

Video ids in this project are just the source filename stem (e.g. "0" for .../Camera1/0.mp4), not
descriptive, so this script does not try to auto-match a project's clips to a spreadsheet row --
you look up the row with --list, then apply it to a specific --pid/--vid.

This has nothing to import for alone-condition sessions (single rat, no entry/exit event) -- they
are not in this spreadsheet at all. Just never call this for those clips; a clip with no window set
behaves exactly as before (whole video usable).

Usage:
    # see every row (day, cage, animal_id, entry/exit minutes, start/end frame)
    uv run scripts/import_social_window.py --list

    # find cage1RB on day1 and print its row
    uv run scripts/import_social_window.py --day day1 --cage cage1RB

    # apply it to a clip in a running labeler project
    uv run scripts/import_social_window.py --day day1 --cage cage1RB --pid cage1rb_day1 --vid 0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

DEFAULT_XLSX = ("/Volumes/kennedy/DATA/GLOBUS_SYNC/Hannah_data/ASD_Social_Photometry/"
                "asd_session_qc_social.xlsx")


def mmss_seconds(t) -> float | None:
    """9.41 -> 581 s. The sheet writes minutes:seconds as minutes.seconds."""
    if pd.isna(t):
        return None
    minutes = int(t)
    seconds = round((float(t) - minutes) * 100)
    if not 0 <= seconds < 60:
        sys.exit(f"time {t!r} is not minutes.seconds (seconds part {seconds})")
    return minutes * 60 + seconds


def load_sessions(xlsx_path: str, fps: float) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="asd_session_qc_social")
    for col, out in (("entry_time", "start_frame"), ("exit_time", "end_frame")):
        df[out] = [None if s is None else int(round(s * fps)) for s in map(mmss_seconds, df[col])]
    return df


def _put(base: str, timeout: int = 15):
    def call(path, body=None):
        url = path if path.startswith("http") else base + path
        req = urllib.request.Request(url, data=body,
                                     method="PUT" if body is not None else "GET",
                                     headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"Labeler error: HTTP {e.code} — {e.read().decode(errors='replace')}")
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach the labeler at {base} ({e.reason}). Is it running?")
    return call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="path to asd_session_qc_social.xlsx")
    ap.add_argument("--list", action="store_true", help="print every session row, then exit")
    ap.add_argument("--day", help="e.g. day1")
    ap.add_argument("--cage", help="e.g. cage1RB")
    ap.add_argument("--pid", help="labeler project id to set the window on (omit to just print the row)")
    ap.add_argument("--vid", help="labeler video id within --pid")
    ap.add_argument("--url", default="http://127.0.0.1:8760", help="labeler base URL (default %(default)s)")
    ap.add_argument("--fps", type=float, default=50.0, help="video frame rate (default %(default)s)")
    args = ap.parse_args()

    df = load_sessions(args.xlsx, args.fps)

    if args.list:
        cols = ["day", "cage", "animal_id", "entry_time", "exit_time", "start_frame", "end_frame", "flags"]
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(df[cols].to_string(index=False))
        return

    if not args.day or not args.cage:
        ap.error("give --day and --cage (see --list), or use --list alone")

    rows = df[(df["day"] == args.day) & (df["cage"] == args.cage)]
    if len(rows) == 0:
        sys.exit(f"no row for day={args.day!r} cage={args.cage!r} in {args.xlsx}")
    if len(rows) > 1:
        sys.exit(f"{len(rows)} rows match day={args.day!r} cage={args.cage!r} — expected one")
    row = rows.iloc[0]

    if pd.isna(row["start_frame"]) or pd.isna(row["end_frame"]):
        sys.exit(f"day={args.day} cage={args.cage}: no entry_time/exit_time in the sheet "
                 f"(entry_time={row['entry_time']!r}, exit_time={row['exit_time']!r}). "
                 f"This looks like a session with no social window, not something to import")

    s, e = int(row["start_frame"]), int(row["end_frame"])
    em, es = divmod(int(mmss_seconds(row["entry_time"])), 60)
    xm, xs = divmod(int(mmss_seconds(row["exit_time"])), 60)
    print(f"day={args.day} cage={args.cage} animal_id={row.get('animal_id')}: "
          f"entry {em}:{es:02d} exit {xm}:{xs:02d} -> frames [{s}, {e})")
    if row.get("flags") and not pd.isna(row.get("flags")):
        print(f"  NOTE flags={row['flags']!r} — check this session before trusting it")

    if not args.pid:
        return
    if not args.vid:
        ap.error("--pid requires --vid (which clip in that project this row applies to)")

    call = _put(args.url.rstrip("/"))
    q_pid, q_vid = urllib.parse.quote(args.pid, safe=""), urllib.parse.quote(args.vid, safe="")
    got = call(f"/api/projects/{q_pid}/videos/{q_vid}/social-window",
              json.dumps({"start_frame": s, "end_frame": e}).encode())
    print(f"set on {args.pid}/{args.vid}: social_start={got['social_start']} social_end={got['social_end']}")


if __name__ == "__main__":
    main()
