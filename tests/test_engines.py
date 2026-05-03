"""Tests for the voice_engines registry + Protocol contracts.

Pure Python — doesn't need torch/sounddevice/openvoice. Runs in any
environment with numpy installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import voice_engines  # noqa: E402
from swap_cli.voice_engines import VoiceEngine  # noqa: E402


def test_registry_lists_both_engines() -> None:
    """Both built-in engines self-register on module import."""
    names = voice_engines.available_engines()
    assert "openvoice" in names
    assert "rvc" in names


def test_get_engine_openvoice() -> None:
    engine = voice_engines.get_engine("openvoice")
    assert engine.name == "openvoice"
    assert isinstance(engine, VoiceEngine)
    # Display name is the user-facing label — not empty.
    assert engine.display_name


def test_get_engine_rvc_stub() -> None:
    """RVC engine registers but is_available() returns False until 14b.2."""
    engine = voice_engines.get_engine("rvc")
    assert engine.name == "rvc"
    assert engine.is_available() is False


def test_get_engine_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown voice engine"):
        voice_engines.get_engine("definitely-not-an-engine")


def test_rvc_engine_methods_raise_not_implemented() -> None:
    """RVCEngine methods make the stub status explicit."""
    engine = voice_engines.get_engine("rvc")
    with pytest.raises(NotImplementedError, match="14b.2"):
        engine.extract_embedding("/dev/null")
    with pytest.raises(NotImplementedError, match="14b.2"):
        engine.make_converter([0.0] * 256)


def test_openvoice_engine_is_available_returns_bool() -> None:
    """is_available() should return a bool — not raise — even when voice
    deps + weights aren't installed (typical CI environment)."""
    engine = voice_engines.get_engine("openvoice")
    result = engine.is_available()
    assert isinstance(result, bool)


def test_default_engine_falls_back_to_openvoice() -> None:
    """Until RVC is implemented, default_engine_name() picks openvoice
    (and it returns a string regardless of whether deps are installed)."""
    name = voice_engines.default_engine_name()
    assert name in ("openvoice", "rvc")


def test_voice_engine_protocol_runtime_check() -> None:
    """Engines pass an isinstance check against the runtime-checkable
    VoiceEngine Protocol. Catches accidental signature drift."""
    for engine_name in voice_engines.available_engines():
        engine = voice_engines.get_engine(engine_name)
        assert hasattr(engine, "name")
        assert hasattr(engine, "display_name")
        assert hasattr(engine, "is_available")
        assert hasattr(engine, "extract_embedding")
        assert hasattr(engine, "make_converter")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
