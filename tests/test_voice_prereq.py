"""Tests for voice_prereq platform-specific GPU detection.

Sprint 14c: verifies the honest Apple Silicon check — we used to claim
"MPS" was supported, but rvc-python's MPS path is broken upstream
(PyTorch 2.6 + fairseq + MPS), so we now return ok=False with a clear
hint that RVC streaming will be CPU-bound on M1/M2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_apple_silicon_returns_honest_cpu_only(monkeypatch) -> None:
    """On arm64 Mac the GPU check is *not* a green MPS pass — it's a
    soft-fail with an explanatory hint that RVC needs NVIDIA."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "darwin")
    monkeypatch.setattr(voice_prereq.platform, "machine", lambda: "arm64")

    check = voice_prereq._check_gpu()
    assert check.ok is False
    assert "Apple Silicon" in check.label
    assert "CPU" in check.label
    assert check.hint is not None
    assert "NVIDIA" in check.hint


def test_intel_mac_blocked(monkeypatch) -> None:
    """Intel Mac has no GPU path at all."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "darwin")
    monkeypatch.setattr(voice_prereq.platform, "machine", lambda: "x86_64")

    check = voice_prereq._check_gpu()
    assert check.ok is False
    assert "Intel" in check.label


def test_check_deps_required_modules() -> None:
    """Sprint 14e: voice path is RVC-only. _check_deps must include the
    RVC runtime modules (rvc_python + fairseq) plus the base audio stack
    (torch, torchaudio, sounddevice, librosa). Critically must NOT
    include 'openvoice' anymore.
    """
    from swap_cli import voice_prereq
    import inspect

    src = inspect.getsource(voice_prereq._check_deps)
    for dep in ("torch", "torchaudio", "sounddevice", "librosa", "rvc_python", "fairseq"):
        assert f'"{dep}"' in src, f"{dep} must be in the required tuple"
    assert '"openvoice"' not in src, "openvoice was removed in 14e"


def test_check_ffmpeg_present_when_on_path(monkeypatch) -> None:
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    check = voice_prereq._check_ffmpeg()
    assert check.ok is True
    assert "ffmpeg" in check.label


def test_check_ffmpeg_missing_with_platform_hint(monkeypatch) -> None:
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.shutil, "which", lambda name: None)
    monkeypatch.setattr(voice_prereq.sys, "platform", "win32")
    check = voice_prereq._check_ffmpeg()
    assert check.ok is False
    assert check.hint is not None and "winget" in check.hint


def test_check_build_tools_skipped_off_windows(monkeypatch) -> None:
    """Visual C++ Build Tools is a Windows-only concern."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "linux")
    check = voice_prereq._check_build_tools()
    assert check.ok is True


def test_prereq_result_has_no_weights_field() -> None:
    """Sprint 14e: PrereqResult.weights field was removed alongside
    OpenVoice. Replaced with ffmpeg + build_tools."""
    from swap_cli import voice_prereq

    fields = voice_prereq.PrereqResult.__dataclass_fields__
    assert "weights" not in fields
    assert "ffmpeg" in fields
    assert "build_tools" in fields


def test_check_audio_cable_darwin_no_blackhole(monkeypatch, tmp_path) -> None:
    """When BlackHole isn't installed the hint points at brew."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "darwin")
    # Make Path("/Library/Audio/...") .exists() always False by replacing
    # the Path class in the module to one rooted at tmp_path.
    monkeypatch.setattr(
        voice_prereq, "Path", lambda *a, **kw: type(Path(""))(tmp_path / "nope")
    )

    check = voice_prereq._check_audio_cable()
    assert check.ok is False
    assert "BlackHole" in check.label
    assert check.hint is not None and "brew" in check.hint


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
