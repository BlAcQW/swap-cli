"""Tests for the virtual camera output (Sprint 14k).

Verifies the wiring: Display accepts the flag, RunOptions defaults
to off for back-compat, and the prereq doctor check returns a
platform-correct shape.

Doesn't exercise the actual driver — that requires OBS Virtual Camera
or v4l2loopback present in CI, which we don't have.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# CI sandbox doesn't have cv2 / av installed — they're real runtime deps
# but the bits of Display / RunOptions we test here don't touch them.
# Insert stub modules so the imports below resolve.
for _mod in ("cv2", "av", "aiortc.mediastreams", "aiortc", "decart", "decart.realtime", "decart.types"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["cv2"].VideoWriter_fourcc = MagicMock(return_value=0)  # type: ignore[attr-defined]
sys.modules["cv2"].VideoWriter = MagicMock()  # type: ignore[attr-defined]


def test_display_accepts_virtual_camera_flag() -> None:
    """Display constructor must accept virtual_camera without raising."""
    from swap_cli.display import Display

    fake_track = MagicMock(name="MediaStreamTrack")
    disp = Display(track=fake_track, virtual_camera=True)
    assert disp._virtual_camera is True

    # Default should remain False to avoid surprising legacy users.
    disp2 = Display(track=fake_track)
    assert disp2._virtual_camera is False


def test_runtime_options_default_no_vcam() -> None:
    """RunOptions.virtual_camera defaults False so existing callers
    without the flag keep their behavior."""
    import dataclasses

    # Stub heavier transitive imports from runtime.py.
    if "swap_cli.camera" not in sys.modules:
        sys.modules["swap_cli.camera"] = types.ModuleType("swap_cli.camera")
        sys.modules["swap_cli.camera"].CameraTrack = MagicMock()  # type: ignore[attr-defined]

    from swap_cli.runtime import RunOptions

    fields = {f.name: f for f in dataclasses.fields(RunOptions)}
    assert "virtual_camera" in fields
    assert fields["virtual_camera"].default is False


class _FakePath:
    """Always-missing Path replacement for the win32 driver-detection tests."""

    def __init__(self, *_a, **_kw) -> None:
        pass

    def exists(self) -> bool:
        return False


def test_obs_vcam_check_handles_missing(monkeypatch) -> None:
    """When NEITHER the driver nor pyvirtualcam are present, the check
    fails with a hint that mentions both fixes."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "win32")
    monkeypatch.setattr(voice_prereq, "Path", _FakePath)
    monkeypatch.setattr(
        voice_prereq.importlib.util, "find_spec", lambda name: None
    )

    check = voice_prereq._check_obs_vcam()
    assert check.ok is False
    assert check.hint is not None
    h = check.hint.lower()
    assert "obs" in h or "obsproject" in h
    assert "pip install" in h


def test_obs_vcam_driver_present_but_pyvcam_missing(monkeypatch) -> None:
    """Sprint 14m branch: user has OBS installed but didn't reinstall
    swap-cli after the 14k dep bump. Hint must point at `pip install`."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "win32")

    class _ExistingPath(_FakePath):
        def exists(self) -> bool:
            return True

    monkeypatch.setattr(voice_prereq, "Path", _ExistingPath)
    monkeypatch.setattr(
        voice_prereq.importlib.util, "find_spec", lambda name: None
    )

    check = voice_prereq._check_obs_vcam()
    assert check.ok is False
    assert "pyvirtualcam" in check.label.lower()
    assert "pip install" in (check.hint or "").lower()


def test_obs_vcam_pyvcam_present_but_driver_missing(monkeypatch) -> None:
    """Sprint 14m branch: dev environment with pip dep but no OBS install.
    Hint must point at the OBS Studio installer."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "win32")
    monkeypatch.setattr(voice_prereq, "Path", _FakePath)
    monkeypatch.setattr(
        voice_prereq.importlib.util, "find_spec", lambda name: object()
    )

    check = voice_prereq._check_obs_vcam()
    assert check.ok is False
    assert "driver missing" in check.label.lower() or "obs" in check.label.lower()
    assert "obsproject" in (check.hint or "").lower()


def test_obs_vcam_both_present(monkeypatch) -> None:
    """Happy path: driver + pyvirtualcam both there → ok=True."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "win32")

    class _ExistingPath(_FakePath):
        def exists(self) -> bool:
            return True

    monkeypatch.setattr(voice_prereq, "Path", _ExistingPath)
    monkeypatch.setattr(
        voice_prereq.importlib.util, "find_spec", lambda name: object()
    )

    check = voice_prereq._check_obs_vcam()
    assert check.ok is True
    assert "ready" in check.label.lower()


def test_obs_vcam_check_linux_soft_passes(monkeypatch) -> None:
    """On Linux we soft-pass the driver (v4l2loopback can be modprobed
    on demand) BUT still require pyvirtualcam. With both, ok=True."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "linux")
    monkeypatch.setattr(
        voice_prereq.importlib.util, "find_spec", lambda name: object()
    )
    check = voice_prereq._check_obs_vcam()
    assert check.ok is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
