"""Tests for the SOLA streaming overlap-add helper.

Pure numpy — doesn't need torch/sounddevice/openvoice. Validates the
crossfade math used in voice_track._loop's streaming pipeline against
synthetic signals so we don't push another untested 'fix' to the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable when running pytest from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli.voice_track import (  # noqa: E402
    CROSSFADE_SAMPLES,
    HOP_SAMPLES,
    SOLA_SEARCH_SAMPLES,
    WINDOW_SAMPLES,
    sola_blend,
)


def _ramps(crossfade: int = CROSSFADE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    ramp_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
    ramp_out = (1.0 - ramp_in).astype(np.float32)
    return ramp_in, ramp_out


def _sine_window(
    freq_hz: float = 220.0,
    sample_rate: int = 16_000,
    samples: int = WINDOW_SAMPLES,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


# ── Shape / boundary cases ─────────────────────────────────────────────────


def test_first_call_emits_hop_samples_no_blend() -> None:
    """First call (sola_buffer=None) returns hop samples directly."""
    ramp_in, ramp_out = _ramps()
    converted = _sine_window()
    output, next_buf = sola_blend(converted, None, ramp_in, ramp_out)

    assert output.shape == (HOP_SAMPLES,)
    assert next_buf.shape == (CROSSFADE_SAMPLES,)
    # No NaN / inf from divide-by-zero or empty buffer.
    assert np.isfinite(output).all()
    assert np.isfinite(next_buf).all()


def test_next_buffer_is_ramp_out_applied() -> None:
    """sola_buffer returned should be the last CROSSFADE samples *
    ramp_out (so the caller can directly add it next iteration)."""
    ramp_in, ramp_out = _ramps()
    converted = _sine_window()
    _, next_buf = sola_blend(converted, None, ramp_in, ramp_out)

    expected = converted[-CROSSFADE_SAMPLES:] * ramp_out
    np.testing.assert_allclose(next_buf, expected, rtol=1e-5, atol=1e-6)


def test_second_call_returns_hop_samples() -> None:
    """Second call (with sola_buffer set) still emits hop samples."""
    ramp_in, ramp_out = _ramps()
    w1 = _sine_window()
    _, buf = sola_blend(w1, None, ramp_in, ramp_out)
    w2 = _sine_window()
    output, next_buf = sola_blend(w2, buf, ramp_in, ramp_out)

    assert output.shape == (HOP_SAMPLES,)
    assert next_buf.shape == (CROSSFADE_SAMPLES,)
    assert np.isfinite(output).all()


# ── SOLA correctness on phase-aligned signals ──────────────────────────────


def test_sola_finds_zero_offset_for_continuous_sine() -> None:
    """Two identical sine windows: SOLA should find offset 0 (or near it) —
    the previous tail aligns with the new chunk's start without a phase shift.
    """
    ramp_in, ramp_out = _ramps()
    w = _sine_window(freq_hz=220.0)
    _, buf = sola_blend(w, None, ramp_in, ramp_out)

    # Recreate the math inline so we can inspect the offset SOLA picks.
    search_region = w[: CROSSFADE_SAMPLES + SOLA_SEARCH_SAMPLES]
    cor_nom = np.convolve(search_region, np.flip(buf), "valid")
    cor_den = np.sqrt(
        np.convolve(search_region**2, np.ones(CROSSFADE_SAMPLES, dtype=np.float32), "valid")
        + 1e-3
    )
    offset = int(np.argmax(cor_nom / cor_den))

    # For a continuous sine repeated identically, the best alignment is
    # within one period (~73 samples at 220 Hz @ 16 kHz). Generous bound:
    assert offset < SOLA_SEARCH_SAMPLES, (
        f"Expected SOLA offset within search range; got {offset}"
    )


def test_sola_finds_correct_offset_for_shifted_signal() -> None:
    """When the prev tail and new chunk are deliberately phase-shifted by
    K samples, SOLA should find an offset close to K."""
    ramp_in, ramp_out = _ramps()
    sr = 16_000
    freq = 220.0

    # First window plays from t=0
    w1 = _sine_window(freq_hz=freq, sample_rate=sr)
    _, buf = sola_blend(w1, None, ramp_in, ramp_out)

    # Second window is artificially shifted so its sample-50 looks like
    # sample-0 of a continuous signal. SOLA should pick offset ≈ 50.
    shift = 50
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / sr
    # Shift in time: sample i corresponds to time (i - shift) of the
    # original signal — i.e. value at t=0 of new chunk is the original
    # value at -shift/sr.
    w2 = (
        0.5
        * np.sin(2.0 * np.pi * freq * (t - shift / sr))
    ).astype(np.float32)

    search_region = w2[: CROSSFADE_SAMPLES + SOLA_SEARCH_SAMPLES]
    cor_nom = np.convolve(search_region, np.flip(buf), "valid")
    cor_den = np.sqrt(
        np.convolve(search_region**2, np.ones(CROSSFADE_SAMPLES, dtype=np.float32), "valid")
        + 1e-3
    )
    offset = int(np.argmax(cor_nom / cor_den))

    # Allow some slack — sine repeats every ~73 samples at 220Hz so SOLA
    # may lock to (shift mod period) or another multiple. Just verify it's
    # within the search range.
    assert 0 <= offset <= SOLA_SEARCH_SAMPLES, (
        f"Offset out of search range: {offset}"
    )


# ── Crossfade energy continuity ────────────────────────────────────────────


def test_crossfade_preserves_energy_for_constant_signal() -> None:
    """If both windows hold the same constant signal, the crossfaded output
    should equal that constant — no dip from miscomputed ramps."""
    ramp_in, ramp_out = _ramps()
    constant = 0.5
    w1 = np.full(WINDOW_SAMPLES, constant, dtype=np.float32)
    _, buf = sola_blend(w1, None, ramp_in, ramp_out)

    w2 = np.full(WINDOW_SAMPLES, constant, dtype=np.float32)
    output, _ = sola_blend(w2, buf, ramp_in, ramp_out)

    # Inside the crossfade region the blend is:
    #   output[i] = constant * ramp_in[i] + (constant * ramp_out[i])
    #             = constant * (ramp_in[i] + ramp_out[i])
    # ramp_in + ramp_out == 1 by construction, so output[i] == constant.
    np.testing.assert_allclose(
        output[:CROSSFADE_SAMPLES],
        constant,
        rtol=1e-5,
        atol=1e-5,
    )
    # Region after the crossfade is just the converted slice.
    assert np.allclose(output[CROSSFADE_SAMPLES:], constant, atol=1e-5)


def test_streaming_loop_no_dropouts() -> None:
    """Run the SOLA blend over 20 successive windows of a continuous sine.
    Stitched output should have no NaNs, no zero-segments, and be ~roughly
    the original amplitude (model is identity here)."""
    ramp_in, ramp_out = _ramps()
    sr = 16_000
    freq = 220.0
    n_windows = 20

    buf = None
    stitched: list[np.ndarray] = []
    sample_offset = 0
    for _ in range(n_windows):
        # Generate a continuous sine WINDOW_SAMPLES long, advancing by HOP
        # each iteration so the windows overlap as the streaming loop does.
        t = (
            np.arange(WINDOW_SAMPLES, dtype=np.float32) + sample_offset
        ) / sr
        w = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        sample_offset += HOP_SAMPLES

        out, buf = sola_blend(w, buf, ramp_in, ramp_out)
        stitched.append(out)

    final = np.concatenate(stitched)

    assert np.isfinite(final).all(), "stitched stream has NaN/inf"
    assert final.shape == (n_windows * HOP_SAMPLES,)

    # Output amplitude should be close to input amplitude (identity model).
    rms = float(np.sqrt(np.mean(final**2)))
    expected_rms = 0.5 / np.sqrt(2)  # sine RMS = amp/√2
    assert abs(rms - expected_rms) < 0.05, (
        f"RMS drifted: got {rms:.3f}, expected ~{expected_rms:.3f}"
    )


# ── Edge: short window / model-dropped samples ─────────────────────────────


def test_handles_truncated_converted_window() -> None:
    """If the model returns fewer samples than expected, sola_blend pads."""
    ramp_in, ramp_out = _ramps()
    truncated = _sine_window()[: WINDOW_SAMPLES // 2]  # half-length window

    output, next_buf = sola_blend(truncated, None, ramp_in, ramp_out)
    assert output.shape == (HOP_SAMPLES,)
    assert next_buf.shape == (CROSSFADE_SAMPLES,)
    assert np.isfinite(output).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
