"""A library of fine-tuned HiDRA classifiers, reusable across projects.

A fine-tune is otherwise private to one behavior in one project: its checkpoints live under
`<project>/hidra/_finetune/<lab>__<action>/models`, and the next fine-tune of that behavior wipes
that directory before it starts. That is right for the HITL loop and wrong for anything you want to
keep. Publishing copies the checkpoints out into the library, beside a manifest saying what the
classifier is and what it was trained on, so any behavior in any project can bind to it later and
Predict with it straight away.

The first library is home-cage monitoring (HCM): one fine-tuned head per HCM behavior (drinking,
rearing, jump-down, ...), trained on HCM recordings. Nothing below is HCM-specific beyond the
default `domain`; a second domain is a second value of that field, not a second code path.

Layout, one directory per entry, self-contained so a directory can be copied between machines:

    <library>/
      <entry id>/
        manifest.json
        checkpoints/<config>.pkl      one per HiDRA config; Predict averages them

Where the library lives: `HIDRA_LIBRARY`, else `<projects root>/hidra_library`. Point the env var at
a shared drive to share one library across a lab.

Entries are immutable once published. Publishing the same name again makes a new version
(`hcm-drinking`, `hcm-drinking-v2`, ...) rather than overwriting, because a project bound to v1 must
keep predicting with v1 -- silently swapping a classifier under a finished analysis is the failure
this guards against.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "manifest.json"
CHECKPOINTS = "checkpoints"
DEFAULT_DOMAIN = "hcm"
SCHEMA = 1

_ROOT: Path | None = None


def configure(projects_root: Path) -> None:
    """Default the library to sit beside the projects (overridden by HIDRA_LIBRARY)."""
    global _ROOT
    _ROOT = Path(projects_root) / "hidra_library"


def root() -> Path:
    env = os.environ.get("HIDRA_LIBRARY")
    if env:
        return Path(env).expanduser()
    if _ROOT is None:
        raise RuntimeError("library root not configured")
    return _ROOT


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:64] or "classifier"


def weights_template(entry_dir: Path) -> str:
    """The `{config}`-templated path predict.py --weights takes."""
    return str(Path(entry_dir) / CHECKPOINTS / "{config}.pkl")


def _read(d: Path) -> dict | None:
    try:
        m = json.loads((d / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or not m.get("id"):
        return None
    m["weights"] = weights_template(d)
    m["n_checkpoints"] = len(list((d / CHECKPOINTS).glob("*.pkl")))
    return m


def entries(domain: str | None = None) -> list[dict]:
    """Every published classifier, newest version of each name last. Unreadable entries are skipped
    rather than raised: one bad directory on a shared drive must not hide the rest."""
    r = root()
    if not r.is_dir():
        return []
    out = [m for d in sorted(r.iterdir())
           if d.is_dir() and not d.name.startswith(".") and (m := _read(d))]
    if domain:
        out = [m for m in out if m.get("domain") == domain]
    return sorted(out, key=lambda m: (m.get("domain", ""), m.get("behavior", ""),
                                      m.get("version", 0)))


def get(entry_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", entry_id or ""):     # no path tricks
        raise KeyError(entry_id)
    d = root() / entry_id
    m = _read(d) if d.is_dir() else None
    if m is None:
        raise KeyError(entry_id)
    return m


def _next_id(base: str) -> tuple[str, int]:
    r = root()
    if not (r / base).exists():
        return base, 1
    v = 2
    while (r / f"{base}-v{v}").exists():
        v += 1
    return f"{base}-v{v}", v


def publish(*, name: str, behavior: str, head: dict, weights: str, trained: dict | None = None,
            source: dict | None = None, domain: str = DEFAULT_DOMAIN, notes: str = "") -> dict:
    """Copy a fine-tune's checkpoints into the library and write its manifest.

    `weights` is the `{config}`-templated path a fine-tune recorded on the behavior; every file it
    matches is copied, keyed by config. `head` is the behavior's binding (lab, action, collapse,
    rate, finetune_mode) -- the collapse and rate are part of what makes the classifier answer the
    right question, so they travel with it."""
    if "{config}" not in str(weights):
        raise ValueError("weights must be a {config}-templated path")
    src_dir, fname = Path(weights).parent, Path(weights).name
    if "{config}" not in fname:
        raise ValueError("the {config} placeholder must be in the file name")
    prefix, _, suffix = fname.partition("{config}")
    found = {}
    for p in sorted(src_dir.glob(f"{prefix}*{suffix}")):
        cfg = p.name[len(prefix): len(p.name) - len(suffix)] if suffix else p.name[len(prefix):]
        if cfg:
            found[cfg] = p
    if not found:
        raise FileNotFoundError(f"no checkpoints match {weights} — fine-tune this behavior first")

    base = slug(f"{domain}-{name}")
    entry_id, version = _next_id(base)
    final = root() / entry_id
    tmp = root() / f".{entry_id}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / CHECKPOINTS).mkdir(parents=True)
    for cfg, p in found.items():
        shutil.copy2(p, tmp / CHECKPOINTS / f"{cfg}.pkl")
    manifest = {
        "schema": SCHEMA, "id": entry_id, "name": name, "version": version, "domain": domain,
        "behavior": behavior,
        "base": {"lab": head["lab"], "action": head["action"]},
        "collapse": head.get("collapse"), "rate": head.get("rate"),
        "finetune_mode": head.get("finetune_mode") or "tail",
        "configs": sorted(found),
        "trained": trained or {}, "source": source or {}, "notes": notes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))
    tmp.replace(final)                  # a half-copied entry never shows up in the listing
    return get(entry_id)


def binding(entry: dict) -> dict:
    """The `behavior["hidra"]` dict that makes Predict use this library classifier."""
    return {"lab": entry["base"]["lab"], "action": entry["base"]["action"],
            "collapse": entry.get("collapse") or "scene", "rate": float(entry.get("rate") or 0.15),
            "weights": entry["weights"], "finetune_mode": entry.get("finetune_mode") or "tail",
            "library": entry["id"]}
