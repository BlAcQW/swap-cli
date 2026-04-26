"""Cross-platform camera enumeration.

We probe up to MAX_CAMERAS by opening each index in a SUBPROCESS so a
native crash (common on Alienware/IR cameras with OpenCV) only kills
the probe — not the GUI.

Friendly names:
- Windows: pygrabber's DirectShow filter list ("Logitech BRIO", etc.)
- Linux: /sys/class/video4linux/videoN/name
- macOS: AVFoundation gives us no easy name list, fall back to generic
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_CAMERAS = 6
PROBE_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class CameraDevice:
    index: int
    label: str


# Inline probe script — runs in a child Python process so an OpenCV
# native crash (access violation on e.g. Alienware IR cameras) only kills
# the child, not the GUI.
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
    """Probe a single camera index in an isolated subprocess."""
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


def _windows_friendly_names() -> dict[int, str]:
    """Return {index: friendly name} from DirectShow's filter list.

    pygrabber wraps the same DirectShow API OBS / Discord / Zoom use, so
    the names match what users see in those apps. Returns {} if the
    library is missing or anything goes sideways.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore[import-not-found]

        graph = FilterGraph()
        names = graph.get_input_devices()
        return {i: name for i, name in enumerate(names)}
    except Exception as err:  # noqa: BLE001
        print(f"[devices] friendly-name lookup unavailable: {err}", flush=True)
        return {}


def _linux_friendly_names() -> dict[int, str]:
    """Read /sys/class/video4linux/videoN/name on Linux. Empty dict on failure."""
    out: dict[int, str] = {}
    base = Path("/sys/class/video4linux")
    if not base.exists():
        return out
    try:
        for entry in base.glob("video*"):
            stem = entry.name.removeprefix("video")
            if not stem.isdigit():
                continue
            idx = int(stem)
            name_file = entry / "name"
            if name_file.exists():
                out[idx] = name_file.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as err:  # noqa: BLE001
        print(f"[devices] /sys readout failed: {err}", flush=True)
    return out


def _friendly_names() -> dict[int, str]:
    if sys.platform == "win32":
        return _windows_friendly_names()
    if sys.platform.startswith("linux"):
        return _linux_friendly_names()
    return {}


def enumerate_cameras() -> list[CameraDevice]:
    """Return working cameras with their friendly names where possible."""
    backend = _backend_for_platform()
    names = _friendly_names()
    if names:
        print(f"[devices] friendly names: {names}", flush=True)

    devices: list[CameraDevice] = []
    for i in range(MAX_CAMERAS):
        print(f"[devices] probing index {i} (subprocess, backend={backend})", flush=True)
        if _probe_one(i, backend):
            label = _label_for(i, names.get(i))
            print(f"[devices] index {i} ok — {label}", flush=True)
            devices.append(CameraDevice(index=i, label=label))
        else:
            print(f"[devices] index {i} unavailable", flush=True)
    print(f"[devices] {len(devices)} camera(s) found", flush=True)
    return devices


def _label_for(index: int, friendly: str | None = None) -> str:
    if friendly:
        return f"{friendly} (#{index})"
    system = platform.system()
    suffix = {
        "Darwin": "FaceTime / iSight",
        "Windows": "DirectShow",
        "Linux": "/dev/video",
    }.get(system, "")
    if index == 0:
        return f"Camera 0 (default){' · ' + suffix if suffix else ''}"
    return f"Camera {index}"
