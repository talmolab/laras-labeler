#!/usr/bin/env python3
"""Repair a behavior whose HiDRA binding + prediction lanes got clobbered by predicting the WRONG
head over it.

The labeler's "bind a head" endpoint REPLACES the whole `hidra` dict (lab/action/collapse/rate), so
switching the head dropdown on a fine-tuned behavior silently drops its `weights`/`finetune_mode`
pointer to the fine-tuned checkpoints — and the mistaken Predict overwrites that behavior's
`predictions/<clip>/<bid>.npy` (plus a stray `<bid>.target.npy` if the wrong head was directed).

This restores the intended binding (re-deriving the `{config}`-templated weights path from the
fine-tune checkpoints still on disk) and deletes the contaminated prediction lanes for that behavior,
so a fresh Predict regenerates the correct ones. LABELS are never touched — Predict never writes them.

    python scripts/repair_hidra_binding.py \
        --project C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl \
        --behavior 0 --lab NiftyGoldfinch --action climb --rate 0.15 --mode tail

Stop the labeler first (it holds the project in memory and would re-save over this on its next write).
Defaults match the DooM jump-down fine-tune. Pass --no-restore to only clear prediction lanes, or
--keep-preds to only fix the binding. Needs no third-party packages — stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_collapse(action: str) -> str:
    """Mirror hidra.default_collapse so the restored collapse matches what the GUI first set."""
    a = action.lower()
    if a.endswith("object") or "cage" in a or "wall" in a:
        return "self"
    if any(k in a for k in ("allo", "sniff", "attack", "mount", "chase", "approach", "follow",
                            "dominan", "escape", "avoid", "flee", "defend", "intromi", "ejacul",
                            "anogenital", "partner", "social", "huddle", "nose", "bite")):
        return "directed"
    if any(k in a for k in ("self", "groom", "rear", "jump", "climb", "dig", "freeze", "immobil",
                            "run", "walk", "rest", "sleep", "explor", "eat", "drink", "scratch")):
        return "self"
    return "scene"


def _tag(pid: str, bid: int, action: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in f"{pid}_{bid}_{action}")[:48]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Restore a clobbered HiDRA head binding + clear its prediction lanes.")
    ap.add_argument("--project", required=True, help="project directory (contains project.json)")
    ap.add_argument("--behavior", type=int, default=0, help="behavior id to repair (default 0 = jump down)")
    ap.add_argument("--lab", default="NiftyGoldfinch", help="HiDRA lab of the intended head")
    ap.add_argument("--action", default="climb", help="HiDRA action of the intended head")
    ap.add_argument("--collapse", help="collapse mode (default: derived from the action, like the GUI)")
    ap.add_argument("--rate", type=float, default=0.15, help="recalibration rate (default 0.15)")
    ap.add_argument("--mode", default="tail", help="finetune_mode recorded on the behavior (default tail)")
    ap.add_argument("--no-restore", action="store_true", help="do not touch the binding, only clear prediction lanes")
    ap.add_argument("--keep-preds", action="store_true", help="do not delete prediction lanes, only fix the binding")
    ap.add_argument("--zero-shot", action="store_true", help="bind the SHIPPED head (drop weights) so Predict runs "
                    "zero-shot — for a zero-shot-vs-fine-tuned baseline; re-run without this flag to restore the "
                    "fine-tuned head afterwards")
    ap.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    pjson = proj / "project.json"
    if not pjson.exists():
        sys.exit(f"no project.json under {proj}")
    man = json.loads(pjson.read_text(encoding="utf-8"))
    pid = man.get("pid") or proj.name
    behs = man.get("behaviors", [])
    beh = next((b for b in behs if int(b.get("id", -1)) == args.behavior), None)
    if beh is None:
        sys.exit(f"behavior id {args.behavior} not found; have {[b.get('id') for b in behs]}")
    bname = beh.get("name", "?")

    # ---- 1. restore the binding (re-derive the fine-tuned weights path) --------------------------
    if not args.no_restore:
        collapse = args.collapse or _default_collapse(args.action)
        models = proj / "hidra" / "_finetune" / f"{args.lab}__{args.action}" / "models"
        tag = _tag(pid, args.behavior, args.action)
        weights = None
        if args.zero_shot:
            print("  --zero-shot: binding the SHIPPED head (no weights) so Predict runs zero-shot")
        elif models.is_dir():
            pkls = sorted(models.glob(f"*__{tag}.pkl"))
            if pkls:
                weights = str(models / ("{config}__" + tag + ".pkl"))
                print(f"  found {len(pkls)} fine-tune checkpoint(s) for tag {tag!r}")
            else:
                print(f"  WARNING: no *__{tag}.pkl in {models} — restoring binding WITHOUT weights "
                      f"(it will predict zero-shot). Re-run the fine-tune to get an adapted head.")
        else:
            print(f"  WARNING: no fine-tune models dir at {models} — restoring binding WITHOUT weights.")

        new_hidra = {"lab": args.lab, "action": args.action, "collapse": collapse, "rate": args.rate}
        if weights:
            new_hidra["weights"] = weights
            new_hidra["finetune_mode"] = args.mode
        print(f"  behavior {args.behavior} ({bname!r}) hidra:")
        print(f"    was: {json.dumps(beh.get('hidra', {}))}")
        print(f"    now: {json.dumps(new_hidra)}")
        beh["hidra"] = new_hidra

    # ---- 2. delete the contaminated prediction lanes for this behavior ---------------------------
    removed = []
    if not args.keep_preds:
        preds = proj / "predictions"
        for pat in (f"*/{args.behavior}.npy", f"*/{args.behavior}.target.npy"):
            for f in preds.glob(pat):
                removed.append(f)
        for f in removed:
            print(f"  delete stale prediction lane: {f.relative_to(proj)}")
            if not args.dry_run:
                f.unlink()
        if not removed:
            print("  no prediction lanes to delete for this behavior")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    if not args.no_restore:
        pjson.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"\nwrote {pjson}")

    print(f"\nDone. Restart the labeler, select '{bname}', confirm the head reads "
          f"'{args.action} · {args.lab}' with the fine-tuned note, then ▶ Predict to regenerate "
          f"the correct lanes. Your reviewed labels were not touched.")


if __name__ == "__main__":
    main()
