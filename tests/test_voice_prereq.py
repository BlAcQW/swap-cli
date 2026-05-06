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
    soft-fail with an explanatory hint about CPU performance."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "darwin")
    monkeypatch.setattr(voice_prereq.platform, "machine", lambda: "arm64")

    check = voice_prereq._check_gpu()
    assert check.ok is False
    assert "Apple Silicon" in check.label
    assert "CPU" in check.label
    assert check.hint is not None
    assert "OpenVoice" in check.hint  # tells users one path still works


def test_intel_mac_blocked(monkeypatch) -> None:
    """Intel Mac has no GPU path at all."""
    from swap_cli import voice_prereq

    monkeypatch.setattr(voice_prereq.sys, "platform", "darwin")
    monkeypatch.setattr(voice_prereq.platform, "machine", lambda: "x86_64")

    check = voice_prereq._check_gpu()
    assert check.ok is False
    assert "Intel" in check.label


def test_check_deps_includes_rvc_python_and_fairseq() -> None:
    """Sprint 14d: `swap voices install` short-circuits on this check, so
    omitting rvc_python or fairseq makes the install command lie about
    success when one of the engines is half-installed.
    """
    from swap_cli import voice_prereq
    import inspect

    src = inspect.getsource(voice_prereq._check_deps)
    assert '"rvc_python"' in src, "rvc_python must be in the required tuple"
    assert '"fairseq"' in src, "fairseq must be in the required tuple"
    # And the existing OpenVoice deps must still be there.
    for dep in ("torch", "torchaudio", "sounddevice", "librosa", "openvoice"):
        assert f'"{dep}"' in src, f"{dep} must remain in the required tuple"


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
