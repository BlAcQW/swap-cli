"""RVC (Retrieval-based Voice Conversion) engine — streaming voice conversion.

Why RVC over OpenVoice:
  - Designed for chunked real-time inference; OpenVoice produces
    'phoneme repeat' garbage on streamed chunks.
  - Hubert content features + F0 frame-by-frame → preserves phonetic
    timing across boundaries.
  - Massive HuggingFace catalog of pre-trained voices, most commercial-safe.
  - Used by w-okada/voice-changer + every VTuber tool.

Sprint 14b.2.b status:
  - rvc-python's public API is batch-file only (infer_file: path → path).
  - We wrap it with temp WAV files per chunk for streaming. Latency
    will be higher than ideal (~200-400 ms) until we either reach into
    rvc-python's internals or replace with a streaming-native fork.
  - The 'voice identity' for RVC is a .pth file on disk — not a 256-d
    speaker embedding. Voice records for RVC have id='rvc-<slug>' and
    empty embedding; the engine looks up the .pth via voice_ops.
    rvc_model_path_for(voice).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:
    import numpy as np

    from ..voice_library import Voice


@register
class RVCEngine:
    """RVC streaming engine via rvc-python."""

    @property
    def name(self) -> str:
        return "rvc"

    @property
    def display_name(self) -> str:
        return "RVC (streaming · per-voice .pth model)"

    def is_available(self) -> bool:
        """True iff rvc-python is installed AND at least one .pth voice
        is registered. RVC has no built-in speakers — users must
        download a model and register via `swap voices add-rvc`."""
        if importlib.util.find_spec("rvc_python") is None:
            return False
        # Need at least one RVC voice registered for the engine to be useful.
        from ..voice_library import load_user_voices

        return any(v.id.startswith("rvc-") for v in load_user_voices())

    def extract_embedding(
        self, wav_path: "Path | str", device: str | None = None
    ) -> list[float]:
        """RVC doesn't extract speaker embeddings — its voice identity is
        the .pth model file. This method exists to satisfy the Protocol;
        users register RVC voices via `swap voices add-rvc`."""
        raise RuntimeError(
            "RVC voices are .pth model files, not speaker embeddings. "
            "Register an RVC model with `swap voices add-rvc /path/to/model.pth "
            "--name \"My voice\"`. To extract a speaker embedding from a WAV, "
            "use the OpenVoice engine (`swap voices engine openvoice`)."
        )

    def make_converter(
        self, target_voice: "Voice", device: str | None = None  # type: ignore[no-untyped-def]
    ):
        """Build a streaming RVCConverter for the given .pth-backed voice."""
        from ..voice_ops import rvc_index_path_for, rvc_model_path_for
        from .rvc_converter import RVCConverter

        if not target_voice.id.startswith("rvc-"):
            raise RuntimeError(
                f"Voice '{target_voice.id}' isn't an RVC voice. RVC voices "
                "have ids starting with 'rvc-'. Pick an RVC voice or switch "
                "engine: `swap voices engine openvoice`."
            )

        pth_path = rvc_model_path_for(target_voice)
        if pth_path is None:
            raise RuntimeError(
                f"RVC model file not found for voice '{target_voice.id}'. "
                "Was the model registered with `swap voices add-rvc`?"
            )

        return RVCConverter(
            model_pth=pth_path,
            index_pth=rvc_index_path_for(target_voice),
            device=device,
        )
