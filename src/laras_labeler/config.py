"""Runtime settings (PLAN.md §10), plus the small persisted app settings.

`Settings` is what the CLI passes in for this process. `app_settings` is what the GUI can change
and expects to survive a restart -- currently only where the HiDRA checkout lives. It sits beside
the projects rather than in a home-directory dotfile so that a projects folder carries its own
setup: copy the folder to another machine and the configuration goes with the data it belongs to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SETTINGS_FILE = "settings.json"


@dataclass
class Settings:
    projects_root: Path
    host: str = "127.0.0.1"
    port: int = 8760
    frame_cache_size: int = 256
    jpeg_quality: int = 85


def settings_path(projects_root: Path) -> Path:
    return Path(projects_root) / SETTINGS_FILE


def load_app_settings(projects_root: Path) -> dict:
    """Persisted settings, or {} if there are none. Never raises: a corrupt or unreadable file must
    not stop the app from starting -- the GUI can simply set the values again."""
    p = settings_path(projects_root)
    try:
        v = json.loads(p.read_text())
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def save_app_settings(projects_root: Path, patch: dict) -> dict:
    """Merge `patch` in and write it back. Returns the merged settings."""
    merged = {**load_app_settings(projects_root), **patch}
    p = settings_path(projects_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(p)          # atomic: a crash mid-write leaves the old file, not a truncated one
    return merged
