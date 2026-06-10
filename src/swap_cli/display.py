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

    from .watermark import WatermarkRemover


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
        watermark: WatermarkRemover | None = None,
        show_window: bool = True,
    ) -> None:
        self._track = track
        self._record_path = record_path
        self._on_quit = on_quit
        self._virtual_camera = virtual_camera
        # Whether THIS loop owns the cv2 preview window. False on macOS GUI: the
        # session runs on a worker thread and macOS forbids cv2 HighGUI off the
        # main thread, so the GUI pumps the SAME cv2 window from the main thread
        # (via latest_frame()) instead. The window is identical either way.
        self._show_window = show_window
        # Sprint 15: optional per-frame watermark remover. None = no-op.
        # Injected as a configured instance so _loop stays thin and the
        # remover is unit-testable in isolation.
        self._watermark = watermark
        self._writer: cv2.VideoWriter | None = None
        # pyvirtualcam.Camera — lazy-init on first frame so we know the
        # actual width/height from Decart's stream rather than guessing.
        self._vcam: Any = None
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._latest_bgr: np.ndarray | None = None
        # Raw (pre-removal) frame, kept so the W-key capture grabs the badge
        # even while watermark removal is on.
        self._latest_raw_bgr: np.ndarray | None = None

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
        if self._show_window:
            cv2.destroyAllWindows()

    def snapshot(self, dest: Path) -> bool:
        """Save the most recent rendered frame as JPEG. Returns success."""
        if self._latest_bgr is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(dest), self._latest_bgr)

    def latest_frame(self) -> np.ndarray | None:
        """Most recent processed (cleaned) frame — so the macOS GUI can pump the
        cv2 window from the main thread. Rebound atomically each frame."""
        return self._latest_bgr

    def latest_raw_frame(self) -> np.ndarray | None:
        """Most recent RAW (pre-removal) frame — for the W-key watermark capture."""
        return self._latest_raw_bgr

    async def _loop(self) -> None:
        if self._show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 960, 540)
        first_frame = True
        try:
            while not self._stopped.is_set():
                frame = await self._track.recv()
                bgr = frame.to_ndarray(format="bgr24")
                self._latest_raw_bgr = bgr  # keep the un-cleaned frame for W capture
                # Sprint 15: strip the Decart watermark before anything
                # downstream sees the frame. process() never raises — it
                # returns the frame unchanged on any failure.
                if self._watermark is not None:
                    bgr = self._watermark.process(bgr)
                self._latest_bgr = bgr
                self._maybe_init_writer(bgr.shape, fps_guess=20)
                if self._writer is not None:
                    self._writer.write(bgr)
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
                if not self._show_window:
                    # macOS GUI: the main thread owns the cv2 window (pumped via
                    # latest_frame()) and the W/Q keys. Yield if we're not pacing
                    # via the vcam so the loop doesn't spin hot.
                    if not self._virtual_camera:
                        await asyncio.sleep(0)
                    first_frame = False
                    continue
                cv2.imshow(WINDOW_TITLE, bgr)
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
                if key in (ord("w"), ord("W")):  # Sprint 15: capture watermark
                    self._capture_watermark_template()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 — show + exit cleanly
            # The Decart remote track raises (often with an empty message) when
            # the connection drops. End the session cleanly so it doesn't hang
            # in "reconnecting" — the user can click Live again.
            detail = str(err) or err.__class__.__name__
            print(f"[display] stream ended: {detail} — stopping session.", flush=True)
            with suppress(Exception):
                self._on_quit()
            self._stopped.set()
        finally:
            if self._show_window:
                cv2.destroyAllWindows()

    def capture_watermark(self, roi: tuple[int, int, int, int] | None = None) -> None:
        """Public entry for the macOS GUI's main-thread W key. With roi=None it
        runs cv2.selectROI (which the GUI calls on the main thread, so it works
        on macOS); the rest of the logic is shared with the in-window W key."""
        self._capture_watermark_template(roi=roi)

    def _capture_watermark_template(
        self, roi: tuple[int, int, int, int] | None = None
    ) -> None:
        """Drag-select the watermark on the current frame and save it as the
        template PNG (Sprint 15). Bound to the `w` key in the preview window.

        Captures from the RAW (pre-removal) frame so it works even while
        removal is on but not matching. Persists the path AND the frame width
        so the multi-scale match centers exactly, then HOT-RELOADS the live
        remover so removal starts immediately — no restart needed.
        """
        # Prefer the raw frame; fall back to the displayed one.
        source = self._latest_raw_bgr
        if source is None:
            source = self._latest_bgr
        if source is None:
            print("[display] no frame yet — can't capture watermark.", flush=True)
            return
        try:
            from . import config as _config

            if roi is None:
                roi = cv2.selectROI(
                    "select watermark — ENTER to save, C to cancel",
                    source,
                    showCrosshair=False,
                )
                cv2.destroyWindow("select watermark — ENTER to save, C to cancel")
            x, y, w, h = (int(v) for v in roi)
            if w <= 0 or h <= 0:
                print("[display] watermark capture cancelled.", flush=True)
                return
            # Guard against a stray click saving a junk template that then
            # matches nothing — the badge is ~150–260px wide.
            if w < 20 or h < 10:
                print(
                    f"[display] selection too small ({w}x{h}) — press W again "
                    "and drag a box around the whole badge.",
                    flush=True,
                )
                return
            crop = source[y : y + h, x : x + w]
            crop = _tighten_to_badge(crop)  # shrink a loose box to the strokes
            frame_width = int(source.shape[1])
            dest = default_watermark_template_path()
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), crop):
                print(f"[display] failed to write template → {dest}", flush=True)
                return
            _config.update(
                watermark_template=str(dest),
                watermark_template_width=frame_width,
            )
            print(
                f"[display] watermark template saved → {dest} "
                f"(from {frame_width}px-wide frame)",
                flush=True,
            )
            # Hot-reload: rebuild the live remover from the just-saved config so
            # removal starts THIS frame — no restart. Pressing W is a clear
            # intent to remove, so we enable even if the toggle was off.
            from .watermark import WatermarkRemover

            new = WatermarkRemover.from_config(_config.load(), enabled=True)
            if new is not None:
                self._watermark = new
                print(
                    "[display] watermark removal now using the captured template "
                    "(live) — if the badge is still visible, press W again and "
                    "box it tighter.",
                    flush=True,
                )
        except Exception as err:  # noqa: BLE001 — capture is best-effort
            print(f"[display] watermark capture error: {err}", flush=True)

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


def _tighten_to_badge(crop: np.ndarray) -> np.ndarray:
    """Shrink a (possibly loose) selection to the badge's bright strokes, so a
    sloppy drag still yields a tight, background-free template that matches
    with high confidence. Uses the same white top-hat as the matcher; returns
    the original crop if no clear strokes are found."""
    try:
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _ret, strokes = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ys, xs = np.where(strokes > 0)
        if xs.size < 20:  # not enough signal — keep the user's box
            return crop
        m = 6  # small margin around the strokes
        x0 = max(0, int(xs.min()) - m)
        y0 = max(0, int(ys.min()) - m)
        x1 = min(crop.shape[1], int(xs.max()) + m + 1)
        y1 = min(crop.shape[0], int(ys.max()) + m + 1)
        tight = crop[y0:y1, x0:x1]
        if tight.shape[0] < 8 or tight.shape[1] < 20:
            return crop
        print(
            f"[display] tightened selection {crop.shape[1]}x{crop.shape[0]} "
            f"-> {tight.shape[1]}x{tight.shape[0]} (badge strokes)",
            flush=True,
        )
        return tight
    except Exception:  # noqa: BLE001 — tightening is best-effort
        return crop


def default_watermark_template_path() -> Path:
    """Canonical location for the captured watermark template PNG."""
    from platformdirs import user_config_dir

    from .config import APP_NAME

    return Path(user_config_dir(APP_NAME)) / "watermarks" / "watermark.png"


def default_snapshot_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "snapshots" / f"swap-{ts}.jpg"


def default_recording_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "recordings" / f"swap-{ts}.mp4"
