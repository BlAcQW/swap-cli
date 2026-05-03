"""Voice cloning engine registry.

A `VoiceEngine` knows how to:
  - Tell whether its dependencies + weights are present locally.
  - Extract a 256-d (or longer) speaker embedding from a reference WAV
    — used by `swap voices add`.
  - Build a stateful `VoiceConverter` that streams mic chunks → cloned
    audio chunks — used by `swap gui --voice` and `swap voice`.

The registry lets the rest of swap-cli stay engine-agnostic. Currently:
  - OpenVoiceEngine: works for one-shot extraction + chunk-by-chunk
    inference, but doesn't streaming-fit (sprint 14a discovered this).
  - RVCEngine: real streaming engine, the standard for live voice
    cloning. Implementation lands in sprint 14b.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class VoiceConverter(Protocol):
    """Stateful tone-color converter for streaming voice cloning.

    Engine-supplied; callers should not construct this directly. Use
    `engine.make_converter(target_embedding)` to get one.
    """

    @property
    def is_warmed_up(self) -> bool:
        """True after warm_up() has run (i.e. source SE has been extracted)."""
        ...

    @property
    def model_sample_rate(self) -> int:
        """Native sample rate the engine's model operates at."""
        ...

    def ensure_loaded(self) -> None:
        """Eagerly load model weights onto the chosen device."""
        ...

    def warm_up(self, source_audio: "np.ndarray", sample_rate: int) -> None:
        """Compute source-side speaker embedding from mic audio. Call once
        with ~2-3 seconds of speech before the first convert()."""
        ...

    def convert(self, audio_chunk: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Convert a single chunk source→target. Pass-through until warmed."""
        ...

    def close(self) -> None:
        """Release model resources (GPU memory, file handles)."""
        ...


@runtime_checkable
class VoiceEngine(Protocol):
    """A pluggable voice-cloning engine (OpenVoice, RVC, etc.)."""

    @property
    def name(self) -> str:
        """Stable id used in config + CLI flags."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for the GUI."""
        ...

    def is_available(self) -> bool:
        """True iff Python deps + model weights are present locally."""
        ...

    def extract_embedding(
        self, wav_path: "Path | str", device: str | None = None
    ) -> list[float]:
        """One-shot: extract a target voice embedding from a WAV/MP3.

        Used by `swap voices add`. Runs on CPU in seconds.
        """
        ...

    def make_converter(
        self, target_embedding: list[float], device: str | None = None
    ) -> VoiceConverter:
        """Build a stateful streaming converter for the given target voice."""
        ...


# ── Registry ────────────────────────────────────────────────────────────────


_REGISTRY: dict[str, type] = {}


def register(engine_cls: type) -> type:
    """Class decorator: register an engine implementation."""
    instance = engine_cls()
    _REGISTRY[instance.name] = engine_cls
    return engine_cls


def get_engine(name: str) -> VoiceEngine:
    """Look up a registered engine by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown voice engine '{name}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def available_engines() -> list[str]:
    """Names of all registered engines (whether available or not)."""
    return sorted(_REGISTRY.keys())


def default_engine_name() -> str:
    """Pick the best available engine. RVC > OpenVoice once 14b.2 ships."""
    # Until RVC engine is implemented, OpenVoice is the only working choice.
    for candidate in ("rvc", "openvoice"):
        if candidate in _REGISTRY:
            engine = get_engine(candidate)
            if engine.is_available():
                return candidate
    return "openvoice"


# Trigger registration of the built-in engines. Imports are deferred so a
# missing optional dep on one engine doesn't break the whole module.
from . import openvoice_engine  # noqa: E402, F401
from . import rvc_engine  # noqa: E402, F401
