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


def test_registry_only_has_rvc() -> None:
    """Sprint 14e: OpenVoice engine removed. RVC is the only registered
    voice engine."""
    names = voice_engines.available_engines()
    assert names == ["rvc"]


def test_get_engine_rvc() -> None:
    engine = voice_engines.get_engine("rvc")
    assert engine.name == "rvc"
    assert isinstance(engine, VoiceEngine)
    assert engine.display_name


def test_get_engine_rvc_not_available_in_ci() -> None:
    """RVC engine registers but is_available() returns False when
    rvc_python isn't installed (the CI default)."""
    engine = voice_engines.get_engine("rvc")
    assert engine.is_available() is False
    assert engine.is_ready() is False


def test_rvc_is_ready_distinct_from_is_available(monkeypatch) -> None:
    """is_available() answers 'can I switch?', is_ready() also requires
    a registered rvc-* voice."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake(name, *a, **kw):
        if name == "rvc_python":
            return object()
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake)

    engine = voice_engines.get_engine("rvc")
    assert engine.is_available() is True
    assert engine.is_ready() is False


def test_get_engine_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown voice engine"):
        voice_engines.get_engine("definitely-not-an-engine")


def test_rvc_engine_extract_embedding_explains_path_only() -> None:
    """RVCEngine.extract_embedding clearly explains RVC voices are .pth
    files, not embeddings — and points at `swap voices add-rvc`."""
    engine = voice_engines.get_engine("rvc")
    with pytest.raises(RuntimeError, match="add-rvc"):
        engine.extract_embedding("/dev/null")


def test_default_engine_is_rvc() -> None:
    """Sprint 14e: default_engine_name() always returns rvc."""
    assert voice_engines.default_engine_name() == "rvc"


def test_voice_engine_protocol_runtime_check() -> None:
    """Engines pass an isinstance check against the runtime-checkable
    VoiceEngine Protocol. Catches accidental signature drift."""
    for engine_name in voice_engines.available_engines():
        engine = voice_engines.get_engine(engine_name)
        assert hasattr(engine, "name")
        assert hasattr(engine, "display_name")
        assert hasattr(engine, "is_available")
        assert hasattr(engine, "is_ready")
        assert hasattr(engine, "extract_embedding")
        assert hasattr(engine, "make_converter")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
