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
    # Sprint 14o: True when the device's name matches a virtual-camera
    # driver (OBS / Snap / ManyCam / DroidCam / XSplit / generic). The
    # GUI deprioritises these in auto-pick to avoid the feedback loop:
    # swap reading from OBS Virtual Camera while also writing to it.
    virtual: bool = False


# Virtual-camera deny list. Sprint 14o: user hit a feedback loop where
# swap auto-picked OBS Virtual Camera (because the real webcam probe
# timed out) and then also wrote its output to it — Lucy was consuming
# its own previous frames. Mirror voice_router's Sound Mapper pattern.
_VIRTUAL_CAMERA_NEEDLES = (
    "obs virtual",
    "virtual camera",
    "virtual webcam",
    "snap camera",
    "manycam",
    "xsplit",
    "droidcam",
    "e2esoft",
    "iriun",
)


def is_virtual_camera(name: str) -> bool:
    """True iff the device name matches a known virtual-camera driver."""
    n = (name or "").lower()
    return any(needle in n for needle in _VIRTUAL_CAMERA_NEEDLES)


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


def _probe_one(index: int, backend: int) -> tuple[bool, str]:
    """Probe a single camera index in an isolated subprocess.

    Returns (ok, reason). reason is one of:
      ""             - probe succeeded
      "timeout"      - cv2.VideoCapture blocked > PROBE_TIMEOUT_S; most
                       common cause is another app holding the camera
                       (Zoom/Teams/Discord/browsers)
      "not_present"  - probe returned non-zero exit; the index doesn't
                       map to a real device
      "error:<msg>"  - subprocess raised something unexpected
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, str(index), str(backend)],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
        )
        if proc.returncode == 0:
            return True, ""
        return False, "not_present"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as err:  # noqa: BLE001
        return False, f"error:{err}"


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
    """Return working cameras with their friendly names where possible.

    Sprint 14o: each device now carries a `virtual` flag so the GUI can
    deprioritise OBS/Snap/etc. when picking the default input. Probe
    failures distinguish timeout (camera held by another app) from
    not_present (no device at that index).
    """
    backend = _backend_for_platform()
    names = _friendly_names()
    if names:
        print(f"[devices] friendly names: {names}", flush=True)

    devices: list[CameraDevice] = []
    for i in range(MAX_CAMERAS):
        print(f"[devices] probing index {i} (subprocess, backend={backend})", flush=True)
        ok, reason = _probe_one(i, backend)
        if ok:
            friendly = names.get(i)
            label = _label_for(i, friendly)
            virtual = is_virtual_camera(friendly or "")
            tag = " [virtual]" if virtual else ""
            print(f"[devices] index {i} ok — {label}{tag}", flush=True)
            devices.append(CameraDevice(index=i, label=label, virtual=virtual))
        elif reason == "timeout":
            print(
                f"[devices] index {i} probe timed out — another app may be holding "
                "this camera (Zoom/Teams/Discord/browsers). Close it and re-launch.",
                flush=True,
            )
        elif reason.startswith("error:"):
            print(f"[devices] index {i} probe error: {reason[6:]}", flush=True)
        else:
            print(f"[devices] index {i} unavailable", flush=True)
    real = sum(1 for d in devices if not d.virtual)
    virtual = sum(1 for d in devices if d.virtual)
    print(
        f"[devices] {len(devices)} camera(s) found ({real} real, {virtual} virtual)",
        flush=True,
    )
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
