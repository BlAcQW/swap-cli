"""Tests for sprint 14b.2.a engine wiring.

Verifies:
  - Config round-trips voice_engine field.
  - VoiceTrack rejects an unknown engine name with a clear error.
  - VoiceTrack rejects an unavailable engine (deps not installed) with a
    clear error pointing at `swap voices install`.

Pure-Python — no torch / sounddevice / GPU required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── Config round-trip ──────────────────────────────────────────────────────


def test_config_voice_engine_default(tmp_path, monkeypatch) -> None:
    """Sprint 14e: fresh config has voice_engine='rvc' by default."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Reload the module after env change so config_path() is fresh.
    import importlib

    from swap_cli import config as _config

    importlib.reload(_config)
    cfg = _config.load()
    assert cfg.voice_engine == "rvc"


def test_config_voice_engine_persists(tmp_path, monkeypatch) -> None:
    """update(voice_engine='rvc') survives a reload."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import importlib

    from swap_cli import config as _config

    importlib.reload(_config)
    _config.update(voice_engine="rvc")
    reloaded = _config.load()
    assert reloaded.voice_engine == "rvc"


def test_config_voice_engine_does_not_break_old_config(tmp_path, monkeypatch) -> None:
    """An old config.toml without voice_engine still loads — defaults applied."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import importlib

    from swap_cli import config as _config

    importlib.reload(_config)
    cfg_path = _config.config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        # An "old" config — no voice_engine key.
        'license_key = "SWAP-CLI-TEST-1234"\n'
        "voice_enabled = true\n",
        encoding="utf-8",
    )
    cfg = _config.load()
    assert cfg.voice_engine == "rvc"  # default applied
    assert cfg.voice_enabled is True


# ── VoiceTrack engine resolution ───────────────────────────────────────────


def test_voicetrack_rejects_unknown_engine() -> None:
    """Unknown engine name raises a clear RuntimeError, not KeyError."""
    from swap_cli.voice_library import Voice
    from swap_cli.voice_track import VoiceTrack, VoiceTrackOptions

    fake_voice = Voice(
        id="test",
        name="Test",
        description="",
        source="library",
        embedding=[0.0] * 256,
        sample_rate=16_000,
        created_at=0,
    )
    opts = VoiceTrackOptions(
        voice=fake_voice,
        microphone_device=0,
        output_device=None,
        engine_name="not-an-engine",
    )
    with pytest.raises(RuntimeError, match="Unknown voice engine"):
        VoiceTrack(opts)


def test_voicetrack_rejects_unavailable_engine() -> None:
    """If the engine is registered but is_available() is False, VoiceTrack
    raises a RuntimeError pointing at `swap voices install`."""
    from swap_cli.voice_library import Voice
    from swap_cli.voice_track import VoiceTrack, VoiceTrackOptions

    fake_voice = Voice(
        id="test",
        name="Test",
        description="",
        source="library",
        embedding=[0.0] * 256,
        sample_rate=16_000,
        created_at=0,
    )
    # RVCEngine is registered but is_available() is False (stub).
    opts = VoiceTrackOptions(
        voice=fake_voice,
        microphone_device=0,
        output_device=None,
        engine_name="rvc",
    )
    with pytest.raises(RuntimeError, match="isn't installed/available"):
        VoiceTrack(opts)


def test_voicetrack_options_default_engine() -> None:
    """VoiceTrackOptions defaults to 'rvc' when engine_name omitted."""
    from swap_cli.voice_library import Voice
    from swap_cli.voice_track import VoiceTrackOptions

    fake_voice = Voice(
        id="test",
        name="Test",
        description="",
        source="library",
        embedding=[0.0] * 256,
        sample_rate=16_000,
        created_at=0,
    )
    opts = VoiceTrackOptions(
        voice=fake_voice,
        microphone_device=0,
        output_device=None,
    )
    assert opts.engine_name == "rvc"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
