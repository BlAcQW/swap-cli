"""Tests for camera enumeration + virtual-camera deny list (Sprint 14o).

The user hit a feedback loop where swap-cli auto-picked OBS Virtual
Camera as input (because the real webcam probe timed out) and also
wrote its output to OBS Virtual Camera — Lucy consumed its own
previous frames. These tests pin the deny-list invariant + verify the
CameraDevice.virtual flag is set correctly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# devices.py imports cv2 lazily inside helpers; preempt with a stub so
# the unit-test process doesn't need OpenCV installed.
if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.CAP_DSHOW = 700  # type: ignore[attr-defined]
    cv2_stub.CAP_AVFOUNDATION = 1200  # type: ignore[attr-defined]
    cv2_stub.CAP_V4L2 = 200  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2_stub


from swap_cli import devices  # noqa: E402


# ── is_virtual_camera ─────────────────────────────────────────────────


def test_is_virtual_camera_recognises_obs() -> None:
    assert devices.is_virtual_camera("OBS Virtual Camera")
    # Also tolerate truncated names (DirectShow truncates at ~32 chars).
    assert devices.is_virtual_camera("OBS Virtual Camera (#1)")
    assert devices.is_virtual_camera("OBS Virtual Camera Output")


def test_is_virtual_camera_recognises_other_virtuals() -> None:
    assert devices.is_virtual_camera("Snap Camera")
    assert devices.is_virtual_camera("ManyCam Virtual Webcam")
    assert devices.is_virtual_camera("XSplit VCam")
    assert devices.is_virtual_camera("DroidCam Source 1")


def test_is_virtual_camera_excludes_real_webcams() -> None:
    """Must NOT match real webcam names — false positives here would
    hide the user's actual camera from auto-pick."""
    assert not devices.is_virtual_camera("HD Webcam")
    assert not devices.is_virtual_camera("Logitech BRIO")
    assert not devices.is_virtual_camera("Microsoft LifeCam HD-3000")
    assert not devices.is_virtual_camera("FaceTime HD Camera")
    assert not devices.is_virtual_camera("Integrated Webcam")
    assert not devices.is_virtual_camera("")
    assert not devices.is_virtual_camera("Camera 0")


def test_is_virtual_camera_handles_none() -> None:
    """Defensive: name might be missing from friendly-names lookup."""
    assert not devices.is_virtual_camera(None)  # type: ignore[arg-type]


# ── _probe_one timeout vs not_present ─────────────────────────────────


def test_probe_one_timeout_returns_timeout_reason(monkeypatch) -> None:
    """The user's HD Webcam probe timed out (camera held by another app).
    _probe_one must distinguish that from not_present so the caller can
    print a helpful 'close other apps' hint instead of just 'unavailable'."""
    import subprocess as real_subprocess

    def fake_run(*_a, **_kw):
        raise real_subprocess.TimeoutExpired(cmd="probe", timeout=3.0)

    monkeypatch.setattr(devices.subprocess, "run", fake_run)
    ok, reason = devices._probe_one(0, backend=700)
    assert ok is False
    assert reason == "timeout"


def test_probe_one_returns_not_present_on_nonzero_exit(monkeypatch) -> None:
    """When the subprocess exits non-zero, the device just doesn't
    exist at that index — different signal from 'app holding it'."""
    fake_proc = MagicMock(returncode=2)
    monkeypatch.setattr(devices.subprocess, "run", lambda *a, **kw: fake_proc)
    ok, reason = devices._probe_one(5, backend=700)
    assert ok is False
    assert reason == "not_present"


def test_probe_one_success(monkeypatch) -> None:
    fake_proc = MagicMock(returncode=0)
    monkeypatch.setattr(devices.subprocess, "run", lambda *a, **kw: fake_proc)
    ok, reason = devices._probe_one(0, backend=700)
    assert ok is True
    assert reason == ""


# ── enumerate_cameras flags virtuals ──────────────────────────────────


def test_enumerate_flags_obs_as_virtual(monkeypatch) -> None:
    """When the friendly-name lookup returns OBS Virtual Camera at
    index 1, enumerate_cameras() must mark that CameraDevice with
    virtual=True — the GUI uses this flag for auto-pick + label prefix."""
    monkeypatch.setattr(
        devices, "_friendly_names", lambda: {0: "HD Webcam", 1: "OBS Virtual Camera"}
    )
    # Both probes succeed in this test.
    monkeypatch.setattr(devices, "_probe_one", lambda i, b: (True, ""))
    monkeypatch.setattr(devices, "MAX_CAMERAS", 2)

    cams = devices.enumerate_cameras()
    assert len(cams) == 2
    by_idx = {c.index: c for c in cams}
    assert by_idx[0].virtual is False
    assert "HD Webcam" in by_idx[0].label
    assert by_idx[1].virtual is True
    assert "OBS" in by_idx[1].label


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
