"""Render an aiortc remote video track in a cv2.imshow window.

Also handles snapshot-on-keypress and optional MP4 recording.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from aiortc.mediastreams import MediaStreamTrack


WINDOW_TITLE = "swap — Lucy 2 live"


class Display:
    """Pulls frames from a remote MediaStreamTrack and renders them.

    Press Q in the window or call `stop()` to terminate the loop.
    """

    def __init__(
        self,
        track: MediaStreamTrack,
        *,
        record_path: Path | None = None,
        on_quit: callable = lambda: None,  # type: ignore[assignment]
    ) -> None:
        self._track = track
        self._record_path = record_path
        self._on_quit = on_quit
        self._writer: cv2.VideoWriter | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._latest_bgr: np.ndarray | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        if self._writer is not None:
            self._writer.release()
        cv2.destroyAllWindows()

    def snapshot(self, dest: Path) -> bool:
        """Save the most recent rendered frame as JPEG. Returns success."""
        if self._latest_bgr is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(dest), self._latest_bgr)

    async def _loop(self) -> None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        try:
            while not self._stopped.is_set():
                frame = await self._track.recv()
                bgr = frame.to_ndarray(format="bgr24")
                self._latest_bgr = bgr
                self._maybe_init_writer(bgr.shape, fps_guess=20)
                if self._writer is not None:
                    self._writer.write(bgr)
                cv2.imshow(WINDOW_TITLE, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # q or ESC
                    self._on_quit()
                    self._stopped.set()
                    break
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 — show + exit cleanly
            print(f"[display] error: {err}")
        finally:
            cv2.destroyAllWindows()

    def _maybe_init_writer(self, shape: tuple[int, ...], fps_guess: int) -> None:
        if self._record_path is None or self._writer is not None:
            return
        h, w = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self._record_path), fourcc, fps_guess, (w, h))


def default_snapshot_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "snapshots" / f"swap-{ts}.jpg"


def default_recording_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "recordings" / f"swap-{ts}.mp4"
