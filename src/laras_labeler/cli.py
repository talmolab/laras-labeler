"""`laras-labeler` entry point (PLAN.md §10): launch uvicorn, open the browser when ready.

v0a bootstraps an in-memory "dev" project pointing at the mice sample so there is
something to label immediately. On-disk projects (§9) land in a later step.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from .config import Settings
from .project import ProjectStore

# Optional dev sample, off by default. Hard-coding a path here made a fresh install crash in
# ensure_dev on any machine that did not happen to have that file, and it is not needed for normal
# use — a project folder is passed on the command line.
_sample = os.environ.get("LARAS_SAMPLE_SLP", "").strip()
SAMPLE_SLP = Path(_sample).expanduser() if _sample else None


def _free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]


def _open_when_ready(host: str, port: int) -> None:
    url = f"http://{host}:{port}/"
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.1)


def main(argv: list[str] | None = None) -> None:
    # the invoked name, so `hidra-in-the-loop --help` does not report itself as laras-labeler
    ap = argparse.ArgumentParser(prog=Path(sys.argv[0]).name or "laras-labeler")
    ap.add_argument("projects_root", nargs="?", default="~/laras-projects",
                    help="directory of on-disk projects (a 'dev' project is seeded on first run)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = auto-pick (prefers 8760)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    projects_root = Path(args.projects_root).expanduser()
    port = args.port or _free_port(args.host, 8760)
    settings = Settings(projects_root=projects_root, host=args.host, port=port)
    store = ProjectStore(projects_root)
    if SAMPLE_SLP and SAMPLE_SLP.exists():
        store.ensure_dev(SAMPLE_SLP)          # first-run demo project only if configured

    from .app import create_app

    app = create_app(settings, store)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(args.host, port), daemon=True).start()
    print(f"laras-labeler -> http://{args.host}:{port}/")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


def hidra_main(argv: list[str] | None = None) -> None:
    """`hidra-in-the-loop`: the same app, named for the HiDRA workflow.

    A second console script rather than a fork or a rename. The labeler is a general pose-based
    labeler that trains its own model and works with no HiDRA at all -- naming the whole tool after
    one classifier family would misdescribe it -- but "install the labeler, then discover HiDRA is
    in there" is a discoverability problem. This is the name to install, cite and point people at.

    The only behavioural difference: it prints the HiDRA runtime before serving, so an unconfigured
    checkout is stated on the terminal too, not only in the GUI's setup row.
    """
    from . import hidra

    rt = hidra.runtime()
    n = len(hidra.catalog())
    print(f"hidra-in-the-loop -> HiDRA checkout: {rt['home']}  ({rt['home_source']})")
    if n and rt["can_infer"]:
        print(f"  {n} classifiers available · backend: {rt['backend']}")
    elif n:
        print(f"  {n} classifiers listed, but inference will not run: {rt['why']}")
    else:
        print(f"  no classifiers found: {rt['why']}")
        print("  set it in the browser (the HiDRA setup box), or with HIDRA_HOME / HIDRA_PYTHON")
    main(argv)


if __name__ == "__main__":
    main()
