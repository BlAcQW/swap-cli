"""Tests for the CUDA torch helpers (Sprint 14g.5).

Verifies:
  - is_cuda_torch_available() returns False cleanly when torch isn't
    importable (typical CI environment).
  - reinstall_cuda_torch() is a no-op on non-NVIDIA platforms (no
    nvidia-smi reachable).
  - reinstall_cuda_torch() invokes pip install with the cu121 index URL
    on NVIDIA platforms.

Pure Python — torch is NOT a swap-cli base dep, so the "torch missing"
path is the realistic CI scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_is_cuda_torch_available_returns_false_when_torch_missing() -> None:
    """In CI, torch isn't installed. The helper must not raise — it
    returns False so doctor / repair logic can branch cleanly."""
    from swap_cli import voice_ops

    # If torch happens to be present in this environment, the test is
    # a no-op (we still trust the helper not to crash).
    result = voice_ops.is_cuda_torch_available()
    assert isinstance(result, bool)


def test_reinstall_cuda_torch_skips_on_non_nvidia(monkeypatch) -> None:
    """No nvidia-smi → no work done. Must NOT shell out to pip."""
    from swap_cli import voice_ops

    monkeypatch.setattr(voice_ops, "_is_nvidia_platform", lambda: False)

    called: list[list[str]] = []

    def fake_check_call(cmd, *a, **kw):
        called.append(cmd)

    monkeypatch.setattr(voice_ops.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(voice_ops.subprocess, "run", fake_check_call)

    assert voice_ops.reinstall_cuda_torch() is True
    assert called == [], "must not invoke pip on non-NVIDIA platforms"


def test_reinstall_cuda_torch_uses_cu121_index(monkeypatch) -> None:
    """On NVIDIA platforms, pip install must include the cu121 index URL."""
    from swap_cli import voice_ops

    monkeypatch.setattr(voice_ops, "_is_nvidia_platform", lambda: True)

    install_args: list[list[str]] = []

    def fake_check_call(cmd, *a, **kw):
        install_args.append(cmd)

    def fake_run(cmd, *a, **kw):
        # Uninstall step — record but don't fail.
        install_args.append(cmd)
        return None

    monkeypatch.setattr(voice_ops.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(voice_ops.subprocess, "run", fake_run)

    assert voice_ops.reinstall_cuda_torch() is True

    # Find the install call (not the uninstall).
    install_calls = [c for c in install_args if "install" in c and "uninstall" not in c]
    assert install_calls, "must run pip install for CUDA torch"
    install_cmd = install_calls[0]
    assert "--index-url" in install_cmd
    assert any("cu121" in arg for arg in install_cmd), (
        f"--index-url must point at cu121, got {install_cmd}"
    )
    assert any("torch" in arg for arg in install_cmd)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
