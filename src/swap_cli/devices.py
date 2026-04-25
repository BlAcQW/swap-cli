"""Cross-platform camera enumeration.

We probe up to MAX_CAMERAS by opening each index with cv2.VideoCapture.
Indexes that open AND can read at least one frame are considered usable.
On macOS this triggers the AppKit camera permission prompt the first time.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2

MAX_CAMERAS = 6


@dataclass(frozen=True)
class CameraDevice:
    index: int
    label: str


def enumerate_cameras() -> list[CameraDevice]:
    """Return cameras that successfully opened.

    Note: this is *slow* (~200ms per probe on cold cameras). Call once at
    GUI startup and cache the result.
    """
    devices: list[CameraDevice] = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if not ok:
            continue
        devices.append(CameraDevice(index=i, label=_label_for(i)))
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
