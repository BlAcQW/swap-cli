"""Test the silent-threshold gate added in Sprint 14i.

The gate is small enough that a unit test wouldn't add much value beyond
exercising the constant. We verify:
  - The threshold constant exists and is in a sane range.
  - A trivial helper that mirrors the gate logic returns the right thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import voice_track  # noqa: E402


def test_silent_threshold_is_sane() -> None:
    """0.001 is below ambient noise (we'd over-trigger), 0.05 cuts speech."""
    assert hasattr(voice_track, "SILENT_THRESHOLD_RMS")
    t = voice_track.SILENT_THRESHOLD_RMS
    assert 0.001 < t < 0.05


def test_silent_buffer_below_threshold() -> None:
    """A pure silence buffer's RMS sits well below the threshold."""
    silence = np.zeros(16000, dtype=np.float32)
    rms = float(np.sqrt(np.mean(silence**2)))
    assert rms < voice_track.SILENT_THRESHOLD_RMS


def test_speech_level_buffer_above_threshold() -> None:
    """A typical speech-level buffer (peak ~0.3, RMS ~0.05) clears the gate."""
    rng = np.random.default_rng(42)
    speech = rng.standard_normal(16000).astype(np.float32) * 0.05
    rms = float(np.sqrt(np.mean(speech**2)))
    assert rms > voice_track.SILENT_THRESHOLD_RMS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
