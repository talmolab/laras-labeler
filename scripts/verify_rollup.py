"""Assert the annotation-timing rollup's arithmetic, over hand-built event streams.

`verify_event_log.py` checks that the BROWSER emits the right events by driving the real UI; this
checks what `events.py` does with them, which needs no browser, no server and no data — so it runs
anywhere in a second and can be pointed at a regression the moment one is suspected:

    python3 scripts/verify_rollup.py

Stdlib only. Each case is a stream of events with a known answer, so a failure names the invariant
that broke rather than a number that moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from laras_labeler.events import ROUND_CSV_COLS, rounds_csv, summarize  # noqa: E402

FAILED: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(name)


def ev(ms, typ, bid=0, src="client", **k):
    """One event. `ms` is the client clock in milliseconds; the rollup reads t_ms and t."""
    secs = ms / 1000.0
    return {"src": src, "t_ms": ms, "t": f"2026-01-01T00:{int(secs // 60):02d}:{secs % 60:06.3f}+00:00",
            "type": typ, "behavior_id": bid, **k}


def trained(ms, bid=0, ap=0.9, **k):
    """The server's record of a finished Train — this is what CLOSES a round."""
    return ev(ms, "job_done", bid=bid, src="server", kind="train", status="done",
              seconds=k.pop("seconds", 1.0), ap=ap, **k)


def paint(ms, start, end, value=1, bid=0):
    return ev(ms, "paint_commit", bid=bid, start=start, end=end, n_frames=end - start, value=value)


# --------------------------------------------------------------- rounds and what they contain

def test_round_grouping():
    print("\nrounds are the stretch between two Trains of a behavior")
    recs = [
        ev(0, "paint_start"), paint(1000, 0, 50), paint(2000, 60, 110),
        trained(2100, ap=0.5),
        ev(3000, "paint_start"), paint(4000, 200, 250),
        trained(4100, ap=0.8),
    ]
    s = summarize(recs)
    check(s["n_rounds"] == 2, "one round per Train", f"{s['n_rounds']}")
    check([r["manual_bouts"] for r in s["rounds"]] == [2, 1], "bouts land in the round that painted them")
    check([r["ap"] for r in s["rounds"]] == [0.5, 0.8], "each round carries the model it produced")
    check(all(r["closed"] for r in s["rounds"]), "a Train closes its round")


def test_two_behaviors_are_separate_loops():
    print("\ntwo behaviors labeled in one sitting are two loops, not one")
    recs = [paint(0, 0, 50, bid=0), paint(1000, 0, 50, bid=1),
            trained(2000, bid=0), trained(3000, bid=1)]
    s = summarize(recs)
    check({r["behavior_id"] for r in s["rounds"]} == {0, 1}, "rounds are per behavior")


def test_open_round_has_no_model():
    print("\na round no Train has closed yet")
    s = summarize([paint(0, 0, 50), ev(1000, "paint_start")])
    r = s["rounds"][0]
    check(not r["closed"] and r["ap"] is None, "stays open, no model attached")
    check(r["manual_bouts"] == 1, "still credits what it produced")


# --------------------------------------------------------------- time attribution

def test_active_never_exceeds_wall():
    """Regression: a gap is charged to the round open when it STARTED, but that round's `end` was
    stamped at its own last event — so a short round followed by a long gap reported more active
    time than it had clock time."""
    print("\nactive time never exceeds wall time")
    recs = [ev(0, "paint_start"), paint(1000, 0, 50), trained(1100),
            ev(15000, "paint_start"), paint(15500, 60, 70)]
    s = summarize(recs, gap_cap_s=15.0)
    for r in s["rounds"]:
        check(r["active_s"] <= r["wall_s"] + 0.05,
              f"round {r['round']}: active {r['active_s']}s <= wall {r['wall_s']}s")


def test_setup_and_navigation_are_not_labeling():
    """Regression: 'label' was the starting phase and the catch-all, so opening the app, picking a
    project and browsing all accrued to hand-labeling — inflating s_per_manual_bout, which is the
    numerator of the headline ratio."""
    print("\nsetup and navigation are not charged to hand-labeling")
    recs = [ev(0, "session_start"), ev(3000, "project_open"), ev(6000, "video_open"),
            ev(9000, "behavior_select"), ev(12000, "paint_start"), paint(13000, 0, 50),
            trained(13100)]
    s = summarize(recs, gap_cap_s=15.0)
    r = s["rounds"][0]
    check(r["other_s"] >= 11.9, "the 12 s of setup lands in other_s", f"{r['other_s']}s")
    check(r["label_s"] <= 2.0, "label_s is only the painting", f"{r['label_s']}s")
    check(abs(r["active_s"] - (r["label_s"] + r["review_s"] + r["wait_s"] + r["other_s"])) < 0.05,
          "the four buckets sum to active_s")
    check(r["work_s"] == round(r["label_s"] + r["review_s"], 1), "work_s = label + review")


def test_wait_is_its_own_phase():
    print("\nwatching a progress bar is neither labeling nor reviewing")
    recs = [ev(0, "paint_start"), paint(500, 0, 50), ev(1000, "train_click"),
            ev(9000, "train_result", ap=0.9), trained(9100, seconds=8.0)]
    s = summarize(recs, gap_cap_s=15.0)
    r = s["rounds"][0]
    check(r["wait_s"] >= 7.9, "the wait after Train is wait_s", f"{r['wait_s']}s")
    check(r["train_s"] == 8.0, "the server's own job duration is recorded", f"{r['train_s']}s")
    check(r["work_s"] < 2.0, "and none of it counts as annotation work", f"work {r['work_s']}s")


def test_wait_charged_to_the_round_that_clicked():
    print("\nthe wait after Train belongs to the round that clicked it")
    recs = [paint(0, 0, 50), ev(1000, "train_click"), ev(9000, "train_result"),
            trained(9100), ev(10000, "paint_start")]
    s = summarize(recs, gap_cap_s=15.0)
    check(s["rounds"][0]["wait_s"] >= 7.9, "round 1 carries the wait",
          f"r1 {s['rounds'][0]['wait_s']}s")
    check(len(s["rounds"]) < 2 or s["rounds"][1]["wait_s"] == 0.0, "round 2 does not")


def test_break_is_not_billed():
    print("\na break stops accruing time instead of being billed to the annotation")
    recs = [ev(0, "paint_start"), paint(1000, 0, 50),
            ev(20 * 60 * 1000, "paint_start"), paint(20 * 60 * 1000 + 1000, 60, 110),
            trained(20 * 60 * 1000 + 1100)]
    s = summarize(recs, gap_cap_s=15.0)
    r = s["rounds"][0]
    check(r["active_s"] < 30, "a 20-minute gap is not counted as work", f"active {r['active_s']}s")
    check(r["wall_s"] > 1000, "wall time still spans it", f"wall {r['wall_s']}s")


def test_idle_heartbeat_breaks_the_chain():
    print("\nan idle heartbeat ends the active stretch even with the tab open")
    recs = [ev(0, "paint_start"),
            ev(10000, "heartbeat", visible=True, idle_ms=200_000),   # idle past the break
            ev(20000, "paint_start"), paint(21000, 0, 50), trained(21100)]
    s = summarize(recs, gap_cap_s=15.0, idle_break_s=120.0)
    r = s["rounds"][0]
    check(r["active_s"] < 25, "the stretch after an idle beat does not accrue",
          f"active {r['active_s']}s")


def test_hidden_tab_breaks_the_chain():
    print("\na hidden tab stops the clock")
    recs = [ev(0, "paint_start"), ev(1000, "tab_hidden", visible=False),
            ev(30000, "tab_visible", visible=True), paint(31000, 0, 50), trained(31100)]
    s = summarize(recs, gap_cap_s=15.0)
    r = s["rounds"][0]
    check(r["active_s"] < 20, "the hidden stretch is not work", f"active {r['active_s']}s")


# --------------------------------------------------------------- the comparison itself

def test_only_positive_paints_are_bouts():
    """A Not-happening paint is work but not a product: counting it would divide the same labeling
    seconds by a bigger number and flatter the by-hand arm."""
    print("\nonly Happening paints count as hand-labeled bouts")
    recs = [paint(0, 0, 50, value=1), paint(1000, 60, 110, value=2), paint(2000, 200, 250, value=3),
            trained(2100)]
    r = summarize(recs)["rounds"][0]
    check(r["manual_bouts"] == 1, "positives are the count", f"{r['manual_bouts']}")
    check(r["manual_neg_bouts"] == 1, "negatives reported separately", f"{r['manual_neg_bouts']}")
    check(r["manual_unknown_bouts"] == 1, "and so are Unknowns")
    check(r["manual_frames"] == 50, "frames follow the same rule", f"{r['manual_frames']}")


def test_paint_mode_falls_back_to_positive():
    print("\na paint_commit with neither value nor mode reads as Happening (old logs)")
    recs = [ev(0, "paint_commit", start=0, end=50, n_frames=50), trained(100)]
    check(summarize(recs)["rounds"][0]["manual_bouts"] == 1, "counted as a positive bout")


def test_review_decisions_and_fixing_cost():
    print("\nreview decisions, dwell, and how much the proposals needed fixing")
    recs = [
        ev(0, "review_start"),
        ev(1000, "candidate_show", start=100, end=200, n_frames=100),
        ev(3000, "candidate_replay"),
        ev(4000, "candidate_trim", edge="start"),
        ev(5000, "candidate_accept", start=100, end=200, dwell_ms=4000, trimmed=True),
        ev(6000, "candidate_show", start=300, end=340, n_frames=40),
        ev(8000, "candidate_reject", start=300, end=340, dwell_ms=2000, trimmed=False),
        ev(9000, "review_end"), trained(9100),
    ]
    r = summarize(recs, gap_cap_s=15.0)["rounds"][0]
    check(r["decisions"]["accept"] == 1 and r["decisions"]["reject"] == 1, "decisions counted")
    check(r["accepted"] == 1, "only bout-producing calls count as accepted")
    check(r["candidates_shown"] == 2, "candidates shown counted")
    check(r["candidate_trims"] == 1, "bound edits counted", f"{r['candidate_trims']}")
    check(r["decisions_trimmed"] == 1, "decisions that edited the bounds counted")
    check(r["replays"] == 1, "replays counted")
    check(r["median_decision_s"] == 3.0, "dwell read from the log", f"{r['median_decision_s']}")
    check(r["review_s"] > r["label_s"], "the time is review time",
          f"review {r['review_s']}s vs label {r['label_s']}s")


def test_headline_direction_and_units():
    print("\nthe manual-vs-review headline")
    # 4 hand bouts over ~20s of labeling; 2 accepted of 4 decisions over ~8s of review
    recs = [ev(0, "paint_start")]
    for i in range(4):
        recs.append(paint(1000 + i * 5000, i * 100, i * 100 + 50))
    recs += [
        ev(21000, "review_start"),
        ev(22000, "candidate_show", start=900, end=950),
        ev(23000, "candidate_accept", start=900, end=950, dwell_ms=1000),
        ev(24000, "candidate_show", start=960, end=990),
        ev(25000, "candidate_reject", start=960, end=990, dwell_ms=1000),
        ev(26000, "review_end"), trained(26100),
    ]
    t = summarize(recs, gap_cap_s=15.0)["totals"]
    check(t["manual"]["bouts"] == 4, "hand bouts")
    check(t["review"]["decisions"] == 2 and t["review"]["accepted"] == 1, "decisions vs accepted")
    check(t["manual"]["s_per_bout"] is not None and t["review"]["s_per_accepted_bout"] is not None,
          "both arms priced", f"{t['manual']['s_per_bout']}s vs {t['review']['s_per_accepted_bout']}s")
    expect = round(t["manual"]["s_per_bout"] / t["review"]["s_per_accepted_bout"], 2)
    check(t["speedup_per_bout"] == expect, "speedup is hand-cost / review-cost (>1 = review cheaper)",
          f"{t['speedup_per_bout']}")


def test_headline_absent_when_an_arm_is_empty():
    print("\nno headline until both arms have produced something")
    t = summarize([paint(0, 0, 50), trained(100)])["totals"]
    check(t["speedup_per_bout"] is None, "hand-only session has no ratio")
    check(t["review"]["s_per_accepted_bout"] is None, "and no review cost")


# --------------------------------------------------------------- the curve, the filter, the CSV

def test_curve_axis_excludes_waiting():
    """Regression: the learning-curve x-axis summed active_s, which includes wait_s — so a slower
    machine read as more annotation effort."""
    print("\nthe accuracy-per-minute axis is annotation only")
    recs = [paint(0, 0, 50), ev(1000, "train_click"), ev(13000, "train_result"),
            trained(13100, seconds=12.0)]
    s = summarize(recs, gap_cap_s=15.0)
    r, c = s["rounds"][0], s["curve"][0]
    check(r["wait_s"] >= 11.9, "the wait is recorded", f"{r['wait_s']}s")
    check(c["cum_work_min"] * 60 < 2.0, "but is not on the work axis", f"{c['cum_work_min']} min")
    check(c["cum_active_min"] > c["cum_work_min"], "active still reports the whole sitting")
    check(all("cum_work_min" in m for m in s["milestones"]), "milestones use the work axis")


def test_behavior_filter():
    """Regression: events carrying no behavior_id survived the filter, so boot and pre-selection
    time was billed to whichever behavior was asked for."""
    print("\n?behavior= means only that behavior")
    recs = [ev(0, "session_start", bid=None), ev(5000, "project_open", bid=None),
            paint(6000, 0, 50, bid=7), trained(6100, bid=7),
            paint(7000, 0, 50, bid=9), trained(7100, bid=9)]
    s = summarize(recs, gap_cap_s=15.0, behavior_id=7)
    check(all(r["behavior_id"] == 7 for r in s["rounds"]), "only that behavior's rounds")
    check(s["rounds"][0]["label_s"] < 1.0, "no pre-selection time billed to it",
          f"{s['rounds'][0]['label_s']}s")
    check(s["totals"]["manual"]["bouts"] == 1, "only its bouts")


def test_csv_round_trips():
    print("\nthe CSV is the per-round table")
    recs = [paint(0, 0, 50), ev(1000, "candidate_show", start=9, end=9),
            ev(2000, "candidate_accept", start=9, end=9, dwell_ms=500), trained(2100)]
    s = summarize(recs)
    lines = [ln for ln in rounds_csv(s).splitlines() if ln.strip()]
    check(lines[0] == ",".join(ROUND_CSV_COLS), "header is every column")
    check(len(lines) == 1 + len(s["rounds"]), "a row per round", f"{len(lines) - 1} rows")
    check(len(lines[1].split(",")) == len(ROUND_CSV_COLS), "no ragged row")
    for col in ("work_s", "other_s", "cum_work_min", "manual_neg_bouts", "decisions_trimmed"):
        check(col in ROUND_CSV_COLS, f"CSV carries {col}")


def test_empty_and_degenerate():
    print("\nnothing to report is not a crash")
    s = summarize([])
    check(s["n_rounds"] == 0 and s["n_events"] == 0, "empty log")
    check(s["totals"]["speedup_per_bout"] is None, "no headline")
    check(rounds_csv(s).strip() == ",".join(ROUND_CSV_COLS), "CSV is just a header")
    s = summarize([ev(0, "heartbeat", visible=True, idle_ms=0)])
    check(s["n_rounds"] == 1 and s["rounds"][0]["manual_bouts"] == 0, "a lone heartbeat")


def main() -> int:
    for fn in (test_round_grouping, test_two_behaviors_are_separate_loops, test_open_round_has_no_model,
               test_active_never_exceeds_wall, test_setup_and_navigation_are_not_labeling,
               test_wait_is_its_own_phase, test_wait_charged_to_the_round_that_clicked,
               test_break_is_not_billed, test_idle_heartbeat_breaks_the_chain,
               test_hidden_tab_breaks_the_chain, test_only_positive_paints_are_bouts,
               test_paint_mode_falls_back_to_positive, test_review_decisions_and_fixing_cost,
               test_headline_direction_and_units, test_headline_absent_when_an_arm_is_empty,
               test_curve_axis_excludes_waiting, test_behavior_filter, test_csv_round_trips,
               test_empty_and_degenerate):
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}):\n  - " + "\n  - ".join(FAILED))
        return 1
    print("OK: rollup arithmetic verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
