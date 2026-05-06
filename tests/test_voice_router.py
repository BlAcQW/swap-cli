"""Tests for voice_router device picking (Sprint 14i).

Sprint 14i added a deny list for Windows compatibility shim devices
('Microsoft Sound Mapper - Input', 'Primary Sound', 'Stereo Mix') that
return silent audio. pick_input_device should skip those when
auto-picking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_is_shim_recognises_sound_mapper() -> None:
    from swap_cli import voice_router

    assert voice_router.is_shim_input_device("Microsoft Sound Mapper - Input")
    assert voice_router.is_shim_input_device("Primary Sound Capture Driver")
    assert voice_router.is_shim_input_device("Stereo Mix (Realtek)")
    assert not voice_router.is_shim_input_device("Microphone Array (Intel® Smart Sound)")
    assert not voice_router.is_shim_input_device("Microphone (Realtek HD Audio)")


def test_pick_input_skips_shim(monkeypatch) -> None:
    """When the first device is a shim, auto-pick must skip it."""
    from swap_cli import voice_router

    fake = (
        [
            {"index": 0, "name": "Microsoft Sound Mapper - Input"},
            {"index": 2, "name": "Microphone Array (Intel)"},
        ],
        [],
    )
    monkeypatch.setattr(voice_router, "list_audio_devices", lambda: fake)

    pick = voice_router.pick_input_device()
    assert pick is not None
    assert pick["index"] == 2
    assert "Microphone Array" in pick["name"]


def test_pick_input_honors_explicit_preferred_even_if_shim(monkeypatch) -> None:
    """If the user explicitly asks for the shim, give it to them — they
    might be debugging or have a weird setup."""
    from swap_cli import voice_router

    fake = (
        [
            {"index": 0, "name": "Microsoft Sound Mapper - Input"},
            {"index": 2, "name": "Microphone Array (Intel)"},
        ],
        [],
    )
    monkeypatch.setattr(voice_router, "list_audio_devices", lambda: fake)

    pick = voice_router.pick_input_device(preferred_index=0)
    assert pick is not None
    assert pick["index"] == 0  # explicit pick respected


def test_pick_input_falls_back_to_shim_when_only_choice(monkeypatch) -> None:
    """If every device is a shim, return the first one — better silent
    audio than crashing with no input device."""
    from swap_cli import voice_router

    fake = (
        [
            {"index": 0, "name": "Microsoft Sound Mapper - Input"},
            {"index": 1, "name": "Primary Sound Capture Driver"},
        ],
        [],
    )
    monkeypatch.setattr(voice_router, "list_audio_devices", lambda: fake)

    pick = voice_router.pick_input_device()
    assert pick is not None
    assert pick["index"] == 0  # last-resort fallback


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
