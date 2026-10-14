"""Build a tailreid inputs CSV (video, slp, start_time) for one HCM camera.

Stdlib only, so it runs with any python3 on the analysis box. Expects the HCM layout:

    <videos>/<YYYY-MM-DD-HH-MM-SS>/<cam>.NN.mp4
    <preds>/<same session>/<cam>.NN.predictions.tracked.slp

An hour's start is the session start + NN hours (the same rule tailreid applies to HCM paths); the
start_time column writes it out so the CSV can be checked by eye. Only hours starting at or after
--since (and before --until, if given) are kept. tailreid needs the TRACKED predictions; hours that
only have an untracked .predictions.slp (or nothing) are skipped and listed. Files modified in the last
--min-age-min minutes are skipped too: the hour still being recorded or tracked would fail tailreid's
video probe and abort the whole run.

    python3 scripts/make_tailreid_inputs.py \
        --videos <VAST>/carlos/HCM_Sync/talmolab/2025-02-12-AzimALS/cam_05 \
        --preds  <VAST>/than/preds_v4/talmolab/2025-02-12-AzimALS/cam_05 \
        --since "2026-10-09 00:00" --out cam05.csv
"""
import argparse, csv, re, sys, time
from datetime import datetime, timedelta
from pathlib import Path

SESSION_RE = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')


def parse_when(s):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f'unrecognised time {s!r} (want "YYYY-MM-DD HH:MM")')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', required=True, type=Path, help='camera video folder (holds the session folders)')
    ap.add_argument('--preds', required=True, type=Path, help='camera predictions folder (same session folders)')
    ap.add_argument('--since', required=True, type=parse_when, help='keep hours starting at or after this (local wall clock)')
    ap.add_argument('--until', type=parse_when, default=None, help='keep hours starting before this')
    ap.add_argument('--min-age-min', type=float, default=10, help='skip files modified less than this many minutes ago (default 10)')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)

    for d in (a.videos, a.preds):
        if not d.is_dir():
            sys.exit(f'not a directory: {d}')
    cam = a.videos.name
    vid_re = re.compile(rf'^{re.escape(cam)}\.(\d{{2}})\.mp4$')
    cutoff = time.time() - a.min_age_min * 60

    rows, untracked, missing, fresh = [], [], [], []
    for sess in sorted(p for p in a.videos.iterdir() if p.is_dir() and SESSION_RE.match(p.name)):
        t0 = datetime.strptime(sess.name, '%Y-%m-%d-%H-%M-%S')
        for v in sorted(sess.iterdir()):
            m = vid_re.match(v.name)
            if not m:
                continue
            start = t0 + timedelta(hours=int(m.group(1)))
            if start < a.since or (a.until and start >= a.until):
                continue
            slp = a.preds / sess.name / f'{cam}.{m.group(1)}.predictions.tracked.slp'
            label = f'{sess.name}/{v.name} ({start:%Y-%m-%d %H:%M})'
            if not slp.exists():
                (untracked if slp.with_name(f'{cam}.{m.group(1)}.predictions.slp').exists() else missing).append(label)
            elif max(v.stat().st_mtime, slp.stat().st_mtime) > cutoff:
                fresh.append(label)
            else:
                rows.append(dict(video=str(v), slp=str(slp), start_time=f'{start:%Y-%m-%d %H:%M:%S}'))

    rows.sort(key=lambda r: r['start_time'])
    if not rows:
        sys.exit(f'no usable hours under {a.videos} since {a.since}')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video', 'slp', 'start_time'])
        w.writeheader(); w.writerows(rows)

    print(f'{cam}: wrote {len(rows)} hour(s) to {a.out}: {rows[0]["start_time"]} -> {rows[-1]["start_time"]}')
    starts = [datetime.strptime(r['start_time'], '%Y-%m-%d %H:%M:%S') for r in rows]
    gaps = [(p, q) for p, q in zip(starts, starts[1:]) if q - p > timedelta(hours=1, minutes=5)]
    for p, q in gaps:
        print(f'  gap: no hour between {p:%Y-%m-%d %H:%M} and {q:%Y-%m-%d %H:%M}')
    for name, lst in (('only untracked .predictions.slp (wait for tracking)', untracked),
                      ('no predictions at all', missing),
                      (f'modified < {a.min_age_min:g} min ago (still being written)', fresh)):
        if lst:
            print(f'  skipped {len(lst)} hour(s), {name}:')
            for x in lst:
                print(f'    {x}')


if __name__ == '__main__':
    main()
