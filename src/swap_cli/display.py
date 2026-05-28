"""Render an aiortc remote video track in a cv2.imshow window.

Also handles snapshot-on-keypress, optional MP4 recording, and
(Sprint 14k) optional output to a virtual camera device so apps like
Zoom/Meet/Discord see the deepfake stream directly.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        virtual_camera: bool = False,
    ) -> None:
        self._track = track
        self._record_path = record_path
        self._on_quit = on_quit
        self._virtual_camera = virtual_camera
        self._writer: cv2.VideoWriter | None = None
        # pyvirtualcam.Camera — lazy-init on first frame so we know the
        # actual width/height from Decart's stream rather than guessing.
        self._vcam: Any = None
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
        if self._vcam is not None:
            with suppress(Exception):
                self._vcam.close()
            self._vcam = None
        cv2.destroyAllWindows()

    def snapshot(self, dest: Path) -> bool:
        """Save the most recent rendered frame as JPEG. Returns success."""
        if self._latest_bgr is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(dest), self._latest_bgr)

    async def _loop(self) -> None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 960, 540)
        first_frame = True
        try:
            while not self._stopped.is_set():
                frame = await self._track.recv()
                bgr = frame.to_ndarray(format="bgr24")
                self._latest_bgr = bgr
                self._maybe_init_writer(bgr.shape, fps_guess=20)
                if self._writer is not None:
                    self._writer.write(bgr)
                cv2.imshow(WINDOW_TITLE, bgr)
                # Sprint 14k: also push the frame to the OBS Virtual Camera
                # driver so Zoom/Meet/Discord pick it up as a real camera.
                # pyvirtualcam expects RGB.
                if self._virtual_camera:
                    self._maybe_init_vcam(bgr.shape, fps_guess=20)
                    if self._vcam is not None:
                        try:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            self._vcam.send(rgb)
                            self._vcam.sleep_until_next_frame()
                        except Exception as err:  # noqa: BLE001
                            print(f"[display] vcam send error: {err}", flush=True)
                            # Don't tear the driver down on a single bad frame.
                if first_frame:
                    # Flash topmost so the cv2 window pops above the tk GUI on
                    # Windows. We don't want it pinned forever — just one beat.
                    with suppress(Exception):
                        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
                        cv2.waitKey(1)
                        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 0)
                    first_frame = False
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

    def _maybe_init_vcam(self, shape: tuple[int, ...], fps_guess: int) -> None:
        """Open pyvirtualcam.Camera on the first frame so we use the
        actual stream resolution. Silently no-ops if pyvirtualcam isn't
        installed or no virtual camera driver is registered — preview
        window keeps working in that case."""
        if self._vcam is not None:
            return
        try:
            import pyvirtualcam  # type: ignore[import-not-found]
        except ImportError:
            # Pure-Python wrapper missing; ship hint via the doctor row.
            self._virtual_camera = False
            print(
                "[display] vcam: pyvirtualcam not installed — `pip install pyvirtualcam`",
                flush=True,
            )
            return
        h, w = shape[:2]
        try:
            # backend=None lets pyvirtualcam pick the right one per OS:
            # Windows → OBS Virtual Camera, macOS → OBS, Linux → v4l2loopback.
            self._vcam = pyvirtualcam.Camera(width=w, height=h, fps=fps_guess)
            print(
                f"[display] vcam ready: '{self._vcam.device}' "
                f"{w}x{h}@{fps_guess}fps — Zoom/Meet/Discord can pick it now.",
                flush=True,
            )
        except Exception as err:  # noqa: BLE001
            # Driver not installed, busy, or fps unsupported. Disable vcam
            # for this session and keep the preview window healthy.
            self._virtual_camera = False
            self._vcam = None
            print(
                f"[display] vcam unavailable: {err}\n"
                "[display] Install OBS Studio for the OBS Virtual Camera driver: "
                "https://obsproject.com/download",
                flush=True,
            )


def default_snapshot_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "snapshots" / f"swap-{ts}.jpg"


def default_recording_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "recordings" / f"swap-{ts}.mp4"
