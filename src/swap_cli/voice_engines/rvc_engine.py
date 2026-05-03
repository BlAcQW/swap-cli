"""RVC (Retrieval-based Voice Conversion) engine — production streaming path.

Sprint 14b.1 ships the registration stub so the rest of the codebase
can already discover that "rvc" is a registered engine name. Real
implementation lands in 14b.2 — needs `rvc-python` package + Hubert
content encoder + RMVPE F0 extractor + a per-voice .pth model.

Why RVC over OpenVoice for streaming:
  - Designed for chunked real-time inference from day one.
  - Hubert content features + F0 frame-by-frame → preserves phonetic
    timing across chunk boundaries (OpenVoice's spectrogram resynthesis
    doesn't, hence the "hey hey hey" failure mode).
  - ~50-100 ms per chunk on a 4070 (vs 200-400 ms for OpenVoice).
  - Massive community catalog of pre-trained voices on HuggingFace,
    most commercially licensed.
  - This is what w-okada/voice-changer + every VTuber tool uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:
    import numpy as np


@register
class RVCEngine:
    """RVC streaming engine. Stub — implementation lands in sprint 14b.2."""

    @property
    def name(self) -> str:
        return "rvc"

    @property
    def display_name(self) -> str:
        return "RVC (streaming · coming in 14b.2)"

    def is_available(self) -> bool:
        # 14b.2: check for `rvc-python` import + RMVPE + hubert weights.
        return False

    def extract_embedding(
        self, wav_path: "Path | str", device: str | None = None
    ) -> list[float]:
        raise NotImplementedError(
            "RVC engine implementation lands in sprint 14b.2. "
            "For now, use the OpenVoice engine for `swap voices add` "
            "(it produces compatible embeddings)."
        )

    def make_converter(
        self, target_embedding: list[float], device: str | None = None
    ):  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "RVC engine implementation lands in sprint 14b.2. "
            "Live streaming currently routes through the OpenVoice engine."
        )
