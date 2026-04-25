"""Cross-platform camera enumeration.

We probe up to MAX_CAMERAS by opening each index in a SUBPROCESS so a
native crash (common on Alienware/IR cameras with OpenCV) only kills
the probe — not the GUI.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass

MAX_CAMERAS = 6
PROBE_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class CameraDevice:
    index: int
    label: str


# This is the script run inside each probe subprocess. Kept inline so we don't
# need to ship an extra file. Exits 0 if the camera works, non-zero otherwise.
# A native crash (access violation) results in a non-zero exit code too — and
# crucially, it doesn't take down the parent.
_PROBE_SCRIPT = """
import sys
import cv2
idx = int(sys.argv[1])
backend = int(sys.argv[2])
cap = cv2.VideoCapture(idx, backend)
try:
    if not cap.isOpened():
        sys.exit(2)
    ok, _ = cap.read()
    sys.exit(0 if ok else 3)
finally:
    cap.release()
"""


def _backend_for_platform() -> int:
    """Pick the most stable cv2 capture backend for the current OS."""
    import cv2

    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _probe_one(index: int, backend: int) -> bool:
    """Probe a single camera index in an isolated subprocess.

    A native access violation in the OpenCV DirectShow plugin on Windows
    (typical on Alienware IR cameras) shows up here as a non-zero exit
    code instead of taking down the GUI.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, str(index), str(backend)],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[devices] index {index} probe timed out", flush=True)
        return False
    except Exception as err:  # noqa: BLE001
        print(f"[devices] index {index} probe error: {err}", flush=True)
        return False


def enumerate_cameras() -> list[CameraDevice]:
    """Return cameras that successfully opened.

    Each index is probed in a subprocess so a crash on one device (e.g. an
    Alienware IR camera) doesn't take down the GUI. Slow (~300ms per index
    via subprocess + cold camera open) — call once at GUI start-up.
    """
    backend = _backend_for_platform()
    devices: list[CameraDevice] = []
    for i in range(MAX_CAMERAS):
        print(f"[devices] probing index {i} (subprocess, backend={backend})", flush=True)
        if _probe_one(i, backend):
            print(f"[devices] index {i} ok", flush=True)
            devices.append(CameraDevice(index=i, label=_label_for(i)))
        else:
            print(f"[devices] index {i} unavailable", flush=True)
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
