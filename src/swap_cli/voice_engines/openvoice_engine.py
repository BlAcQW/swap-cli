"""OpenVoice engine — wraps the existing voice_model.py functions behind
the VoiceEngine protocol.

This is a thin adapter — the real OpenVoice integration still lives in
voice_model.py (extract_embedding, VoiceConverter). Sprint 14b.1 is
purely about giving the rest of the codebase a uniform interface so RVC
can be slotted in without changing voice_track.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:
    import numpy as np


@register
class OpenVoiceEngine:
    """OpenVoice V2 tone-color converter, exposed through VoiceEngine.

    Good for one-shot extraction (`swap voices add`); known to struggle
    with chunked streaming inference — sprint 14a's SOLA crossfade
    helped but didn't fully fix the "phoneme repeat" failure mode. RVC
    engine (14b.2) is the production streaming path.
    """

    @property
    def name(self) -> str:
        return "openvoice"

    @property
    def display_name(self) -> str:
        return "OpenVoice V2 (one-shot · streaming-limited)"

    def is_available(self) -> bool:
        from .. import voice_model
        from ..voice_prereq import openvoice_weights_dir

        if not voice_model.voice_deps_present():
            return False
        weights_dir = openvoice_weights_dir() / "converter"
        return (weights_dir / "checkpoint.pth").exists() and (
            weights_dir / "config.json"
        ).exists()

    def extract_embedding(
        self, wav_path: "Path | str", device: str | None = None
    ) -> list[float]:
        from .. import voice_model

        return voice_model.extract_embedding(wav_path, device=device)

    def make_converter(
        self, target_embedding: list[float], device: str | None = None
    ):  # type: ignore[no-untyped-def]
        from .. import voice_model

        return voice_model.VoiceConverter(
            target_embedding=target_embedding, device=device
        )
