"""DroidCam virtual camera sink (Windows) — the one that works in WhatsApp.

Why this exists alongside the OBS/pyvirtualcam path:

OBS's virtual camera is a **DirectShow filter**, not a real camera device. Apps
that enumerate cameras through Media Foundation (Store/UWP apps, the Windows
Camera app, WhatsApp Desktop) never see it — and WhatsApp additionally refuses
devices it can identify as virtual. DroidCam instead installs a **real camera
driver** ("Droidcam Video" in Device Manager), so apps treat it as an ordinary
webcam and it shows up where OBS's camera can't.

DroidCam's drivers read frames from a named shared-memory buffer. We write into
that buffer directly, so neither OBS nor the DroidCam OBS plugin is needed —
only DroidCam's drivers, which the user installs from dev47apps.

Interface contract (reimplemented here; no third-party code is used):

  file mapping  DroidCamOBS_VideoOut1    1024-byte header, then the frame buffer
  events        DroidCamOBS_VideoWr1     writer holds it low while copying
                DroidCamOBS_VideoRd1     reader signals it's safe to write
  header        int32 version, control, checksum, width, height, interval, format
                valid iff control == 0x02020101
                and checksum == interval ^ width ^ height
  frames        YUYV 4:2:2 (Y0 U0 Y1 V0), stride = width * 2

The DRIVER publishes the mode it wants — resolution and frame interval are set
by whichever app opened the camera, so we resize each frame to match rather than
imposing our own geometry. When the header is invalid, no app has the camera
open and there's nothing to send.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from typing import Any

# --- protocol constants -------------------------------------------------------

CONTROL = 0x02020101
VIDEO_MAP_NAME = "DroidCamOBS_VideoOut1"
VIDEO_WR_LOCK_NAME = "DroidCamOBS_VideoWr1"
VIDEO_RD_LOCK_NAME = "DroidCamOBS_VideoRd1"

HEADER_SIZE = 1024  # the header struct is padded to 1 KiB
MAX_WIDTH = 3860
MAX_HEIGHT = 2160
# The driver sizes the buffer for the largest supported frame at 3 bytes/px,
# which comfortably covers YUYV's 2 bytes/px.
VIDEO_MAP_SIZE = HEADER_SIZE + MAX_WIDTH * MAX_HEIGHT * 3

# Frame intervals are in 100 ns units (10^7 per second), matching DirectShow.
UNITS_PER_SECOND = 10_000_000

# Win32 constants used by the sink (module-level: they are constants, and
# defining them in-function trips the "locals are lowercase" convention).
_PAGE_READWRITE = 0x04
_FILE_MAP_WRITE = 0x0002
_WAIT_OBJECT_0 = 0x0

# How long to wait for the reader to release the buffer before dropping a frame.
# Matches the driver's own cadence: better to skip a frame than stall the
# pipeline and add latency to the live preview.
WRITE_TIMEOUT_MS = 5


@dataclass(frozen=True)
class VideoMode:
    """The resolution/rate the driver is currently asking for."""

    width: int
    height: int
    interval: int  # 100 ns units per frame

    @property
    def fps(self) -> float:
        if self.interval <= 0:
            return 0.0
        return UNITS_PER_SECOND / self.interval


def parse_header(raw: bytes) -> VideoMode | None:
    """Header bytes → the mode the driver wants, or None if no app is consuming.

    The checksum guards against reading a half-written or stale header: the
    driver only leaves `control`/`checksum` consistent while a client actually
    holds the camera open.
    """
    if len(raw) < 28:  # 7 x int32
        return None
    _version, control, checksum, width, height, interval, _format = struct.unpack(
        "<7i", raw[:28]
    )
    if control != CONTROL:
        return None
    if checksum != (interval ^ width ^ height):
        return None
    if not (0 < width <= MAX_WIDTH and 0 < height <= MAX_HEIGHT):
        return None
    return VideoMode(width=width, height=height, interval=interval)


# --- pixel conversion ---------------------------------------------------------

# BT.601 studio swing. Webcam YUYV is limited-range (Y 16-235, C 16-240), but
# OpenCV's BGR2YUV emits full-range, so we squeeze it with a 256-entry LUT —
# far cheaper per frame than redoing the colour maths in floating point.
def _luts() -> tuple[Any, Any]:
    import numpy as np

    y_lut = (np.arange(256) * 219 // 255 + 16).astype(np.uint8)
    c_lut = (((np.arange(256).astype(np.int32) - 128) * 224 // 255) + 128).astype(
        np.uint8
    )
    return y_lut, c_lut


_LUT_CACHE: tuple[Any, Any] | None = None


def bgr_to_yuyv(bgr: Any, *, full_range: bool = False) -> bytes:
    """BGR ndarray → packed YUYV bytes (Y0 U0 Y1 V0 …).

    Chroma is averaged across each horizontal pair rather than point-sampled,
    which avoids the shimmer you get from dropping every other column.

    `full_range` is an escape hatch: if colours come out washed out or crushed
    on a given driver build, flipping it skips the studio-swing squeeze.
    """
    global _LUT_CACHE
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    if w % 2:  # YUYV encodes pixels in pairs
        bgr = bgr[:, : w - 1]
        w -= 1

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    if not full_range:
        if _LUT_CACHE is None:
            _LUT_CACHE = _luts()
        y_lut, c_lut = _LUT_CACHE
        yuv = cv2.merge(
            [
                cv2.LUT(yuv[:, :, 0], y_lut),
                cv2.LUT(yuv[:, :, 1], c_lut),
                cv2.LUT(yuv[:, :, 2], c_lut),
            ]
        )

    u = yuv[:, :, 1].astype(np.uint16)
    v = yuv[:, :, 2].astype(np.uint16)
    u_pairs = ((u[:, 0::2] + u[:, 1::2]) // 2).astype(np.uint8)
    v_pairs = ((v[:, 0::2] + v[:, 1::2]) // 2).astype(np.uint8)

    out = np.empty((h, w, 2), np.uint8)
    out[:, :, 0] = yuv[:, :, 0]
    out[:, 0::2, 1] = u_pairs
    out[:, 1::2, 1] = v_pairs
    return out.tobytes()


def fit_to_mode(bgr: Any, mode: VideoMode) -> Any:
    """Resize a frame to exactly what the driver asked for.

    The driver supports letterboxing a smaller frame, but matching its geometry
    outright keeps the sink simple and avoids stale pixels around the edges.
    """
    import cv2

    h, w = bgr.shape[:2]
    if (w, h) == (mode.width, mode.height):
        return bgr
    # INTER_AREA downscales without aliasing; LINEAR is fine going up.
    interp = cv2.INTER_AREA if (mode.width < w or mode.height < h) else cv2.INTER_LINEAR
    return cv2.resize(bgr, (mode.width, mode.height), interpolation=interp)


# --- the Windows sink ---------------------------------------------------------


class DroidCamUnavailableError(RuntimeError):
    """Raised when the DroidCam shared-memory interface can't be opened."""


class DroidCamSink:
    """Writes frames into DroidCam's shared-memory buffer.

    Usage mirrors the pyvirtualcam path: construct, `open()`, then `send()` each
    frame. `send()` returns False when no app currently holds the camera, which
    is the normal idle state — not an error.
    """

    def __init__(self, *, full_range: bool = False) -> None:
        self._full_range = full_range
        self._map: Any = None
        self._view: Any = None
        self._wr: Any = None
        self._rd: Any = None
        self._k32: Any = None
        self._warned_inactive = False

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> None:
        if sys.platform != "win32":
            raise DroidCamUnavailableError(
                "The DroidCam camera output is Windows-only. On macOS/Linux use "
                "the OBS virtual camera instead."
            )
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32 = k32

        k32.CreateFileMappingW.restype = wintypes.HANDLE
        k32.CreateFileMappingW.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPCWSTR,
        ]
        k32.MapViewOfFile.restype = ctypes.c_void_p
        k32.MapViewOfFile.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_size_t,
        ]
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]

        self._map = k32.CreateFileMappingW(
            wintypes.HANDLE(-1),  # INVALID_HANDLE_VALUE: backed by the paging file
            None,
            _PAGE_READWRITE,
            0,
            VIDEO_MAP_SIZE,
            VIDEO_MAP_NAME,
        )
        if not self._map:
            raise DroidCamUnavailableError(
                f"couldn't open DroidCam's shared buffer "
                f"(error {ctypes.get_last_error()}) — are the DroidCam drivers installed?"
            )

        self._view = k32.MapViewOfFile(self._map, _FILE_MAP_WRITE, 0, 0, 0)
        if not self._view:
            err = ctypes.get_last_error()
            self.close()
            raise DroidCamUnavailableError(f"couldn't map DroidCam's buffer (error {err})")

        # Manual-reset, initially signalled — same handshake the drivers expect.
        self._wr = k32.CreateEventW(None, True, True, VIDEO_WR_LOCK_NAME)
        self._rd = k32.CreateEventW(None, True, True, VIDEO_RD_LOCK_NAME)
        if not self._wr or not self._rd:
            err = ctypes.get_last_error()
            self.close()
            raise DroidCamUnavailableError(f"couldn't open DroidCam's locks (error {err})")

    def close(self) -> None:
        """Release everything. Never raises — callers are in teardown paths."""
        import contextlib

        k32 = self._k32
        if k32 is None:
            return
        import ctypes

        with contextlib.suppress(Exception):
            if self._view:
                k32.UnmapViewOfFile(ctypes.c_void_p(self._view))
        for handle in ("_wr", "_rd", "_map"):
            with contextlib.suppress(Exception):
                h = getattr(self, handle)
                if h:
                    k32.CloseHandle(h)
            setattr(self, handle, None)
        self._view = None

    # -- frame path -----------------------------------------------------------

    def mode(self) -> VideoMode | None:
        """What the driver is asking for, or None if no app has the camera."""
        if not self._view:
            return None
        import ctypes

        raw = ctypes.string_at(self._view, 28)
        return parse_header(raw)

    def send(self, bgr: Any) -> bool:
        """Publish one frame. Returns False if no app is consuming the camera.

        Follows the driver's handshake: drop the write flag, wait briefly for the
        reader to release the buffer, copy, then raise the flag again. A missed
        wait drops the frame rather than blocking the pipeline.
        """
        if not self._view:
            return False
        mode = self.mode()
        if mode is None:
            if not self._warned_inactive:
                self._warned_inactive = True
                print(
                    "[droidcam] no app has the DroidCam camera open yet — "
                    "pick 'Droidcam Video' in WhatsApp/Zoom to start the stream.",
                    flush=True,
                )
            return False
        self._warned_inactive = False

        import ctypes

        payload = bgr_to_yuyv(fit_to_mode(bgr, mode), full_range=self._full_range)
        expected = mode.width * mode.height * 2
        if len(payload) != expected:
            # Odd-width frames lose a column in conversion; skip rather than
            # write a short buffer the driver would render as torn.
            return False

        k32 = self._k32
        k32.ResetEvent(self._wr)
        try:
            if k32.WaitForSingleObject(self._rd, WRITE_TIMEOUT_MS) == _WAIT_OBJECT_0:
                ctypes.memmove(
                    ctypes.c_void_p(self._view + HEADER_SIZE), payload, len(payload)
                )
                return True
            return False  # reader busy — drop this frame
        finally:
            k32.SetEvent(self._wr)

    def __enter__(self) -> DroidCamSink:
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def is_driver_installed() -> bool:
    """True when DroidCam's camera driver looks present.

    Checks the registry key the drivers create rather than probing the shared
    memory, so `swap doctor` can report it without opening any handles.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return False
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\DroidCam"):
                return True
        except OSError:
            continue
    return False
