"""PyInstaller entry point for the packaged Windows desktop app.

Double-clicking the frozen `.exe` should open the window a non-technical
user expects — the same thing `swap gui` launches — not a CLI prompt.

Kept deliberately tiny: all real logic lives in ``swap_cli.gui``. This file
exists only so PyInstaller has a script to freeze, and so the frozen app has
a clean ``__main__`` guard with ``freeze_support`` (a frozen process can be
re-spawned by libraries that use multiprocessing; without this the child
would re-run the GUI instead of the worker).
"""

from __future__ import annotations

import multiprocessing


def main() -> None:
    from swap_cli.gui import launch

    launch()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
