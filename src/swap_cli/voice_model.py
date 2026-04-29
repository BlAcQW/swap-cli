"""OpenVoice v2 wrapper — lazy-loaded, GPU-only.

Sprint 13b.1 ships a working CPU/skeleton path so the rest of the surface
(library loading, voices subcommand, GUI modal) is testable without a GPU.
The real-time tone-color converter lands in 13b.2 once we can iterate on a
GPU machine.

Design:
- Heavy imports (torch, openvoice, numpy) are deferred until the model is
  actually loaded. Calling code can `import voice_model` cheaply.
- `extract_embedding()` runs on CPU in seconds — used by `swap voices add`.
- `convert_chunk()` runs on GPU in real time — used by the live session.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .voice_prereq import openvoice_weights_dir

if TYPE_CHECKING:
    import numpy as np


def voice_deps_present() -> bool:
    """Cheap check — are torch + sounddevice + librosa importable?"""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "torchaudio", "sounddevice", "librosa")
    )


def select_device() -> str:
    """Pick CUDA / MPS / CPU. Lazy-imports torch."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Embedding extraction (used at session-config time, runs once) ─────────


def extract_embedding(wav_path: Path | str, device: str | None = None) -> list[float]:
    """Extract a tone-color embedding from a reference WAV/MP3.

    Used by `swap voices add ./alice.wav --name Alice`. Runs on CPU in a
    few seconds — no GPU needed. Returns a 256-dim list of floats safe to
    JSON-serialize.

    13b.1: stub that returns a deterministic placeholder so the rest of
    the pipeline is testable. 13b.2 swaps in the real OpenVoice extractor.
    """
    if not voice_deps_present():
        raise RuntimeError(
            "Voice deps not installed. Run `swap voices install` first."
        )

    import hashlib

    # Placeholder: hash the file path into a 256-d "embedding" so different
    # files get different placeholders. Real extractor is the TODO below.
    raw = Path(wav_path).resolve().as_posix().encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    # 256 dims → repeat the 32-byte digest 8 times, normalize to [-1, 1].
    placeholder = [
        ((b - 128) / 128.0) for _ in range(8) for b in digest
    ]
    return placeholder

    # TODO (13b.2): replace placeholder with real OpenVoice tone-color extraction.
    #
    # from openvoice.api import ToneColorConverter
    # weights = openvoice_weights_dir() / "checkpoints/converter"
    # converter = ToneColorConverter(weights / "config.json", device=device or select_device())
    # converter.load_ckpt(weights / "checkpoint.pth")
    # embedding_tensor = converter.extract_se(str(wav_path))  # (1, 256, 1)
    # return embedding_tensor.squeeze().cpu().numpy().tolist()


# ── Realtime conversion (used per audio chunk) ────────────────────────────


class VoiceConverter:
    """Stateful tone-color converter. Loads weights once, then converts
    streamed audio chunks in real time.

    13b.1: skeleton that no-ops (passes audio through unchanged) so the
    runtime pipeline can be wired up without a GPU. 13b.2 swaps in the
    real model.
    """

    def __init__(self, target_embedding: list[float], device: str | None = None) -> None:
        if not voice_deps_present():
            raise RuntimeError(
                "Voice deps not installed. Run `swap voices install` first."
            )
        self.target_embedding = target_embedding
        self.device = device or select_device()
        # 13b.2: load OpenVoice ToneColorConverter, move target_se to device.
        self._model: Any = None
        self._source_se: Any = None  # source tone color, computed from first ~3s of mic
        self._warmed_up = False

    def warm_up(self, source_audio: "np.ndarray", sample_rate: int) -> None:
        """Compute the source tone-color embedding from the first few
        seconds of microphone input. Called once at session start.

        13b.1: no-op.
        13b.2: extract source SE so the converter has source→target mapping.
        """
        self._warmed_up = True

    def convert(self, audio_chunk: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Convert one audio chunk. Input and output are float32 numpy arrays
        in [-1, 1] at the given sample rate.

        13b.1: pass-through (returns the input unchanged) — useful for
        testing the audio routing pipeline without the model.
        13b.2: real tone-color conversion.
        """
        if not self._warmed_up:
            # In real use, runtime calls warm_up() once before the first
            # convert(). For 13b.1 this just early-returns.
            return audio_chunk
        return audio_chunk

    def close(self) -> None:
        """Release model resources. 13b.2: del self._model + torch.cuda.empty_cache."""
        self._model = None
