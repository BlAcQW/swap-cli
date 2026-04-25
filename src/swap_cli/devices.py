"""Cross-platform camera enumeration.

We probe up to MAX_CAMERAS by opening each index with cv2.VideoCapture.
Indexes that open AND can read at least one frame are considered usable.
On macOS this triggers the AppKit camera permission prompt the first time.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import cv2

MAX_CAMERAS = 6


@dataclass(frozen=True)
class CameraDevice:
    index: int
    label: str


def _probe_one(index: int, backend: int) -> bool:
    """Open + read one frame from a single camera index. Crash-safe."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return False
        ok, _ = cap.read()
        return bool(ok)
    except Exception as err:  # noqa: BLE001 — DirectShow can throw on IR cams
        print(f"[devices] index {index} probe error: {err}", flush=True)
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


def enumerate_cameras() -> list[CameraDevice]:
    """Return cameras that successfully opened.

    Note: this is *slow* (~200ms per probe on cold cameras). Call once at
    GUI startup and cache the result.
    """
    # On Windows force the DirectShow backend explicitly. The default backend
    # (MSMF / obsensor) hard-crashes the process when probing some Alienware
    # laptops' IR cameras and depth sensors.
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_V4L2

    devices: list[CameraDevice] = []
    for i in range(MAX_CAMERAS):
        print(f"[devices] probing index {i} with backend={backend}", flush=True)
        if _probe_one(i, backend):
            devices.append(CameraDevice(index=i, label=_label_for(i)))
    print(f"[devices] {len(devices)} camera(s) found", flush=True)
    return devices


def _label_for(index: int) -> str:
    system = platform.system()
    suffix = {
        "Darwin": "FaceTime / iSight",
        "Windows": "DirectShow",
        "Linux": "/dev/video",
    }.get(system, "")
    if index == 0:
        return f"Camera 0 (default){' · ' + suffix if suffix else ''}"
    return f"Camera {index}"
