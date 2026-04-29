"""OpenVoice v2 wrapper — lazy-loaded, GPU-recommended.

This is the 13b.2 implementation: real tone-color extraction + chunked
realtime conversion via OpenVoice's ToneColorConverter.

Two functions are exposed to the rest of swap-cli:

    extract_embedding(wav_path) -> list[float]
        One-shot. Used by `swap voices add ./alice.wav`. Runs on CPU in
        a few seconds. Output is a 256-d list safe to JSON-serialize.

    VoiceConverter(target_embedding, device=None)
        Stateful. Used by the live session.
        - warm_up(source_audio_chunk, sample_rate) extracts the source SE
          from the first ~3s of mic input.
        - convert(audio_chunk, sample_rate) converts subsequent chunks
          using source SE → target SE on the GPU.
        - close() releases resources.

Heavy imports (torch, openvoice) are deferred until first use. Calling
code can `import voice_model` without the [voice] extra installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .voice_prereq import openvoice_weights_dir

if TYPE_CHECKING:
    import numpy as np


CONVERTER_SUBDIR = "converter"
CONFIG_FILENAME = "config.json"
CHECKPOINT_FILENAME = "checkpoint.pth"
DEFAULT_TAU = 0.3  # OpenVoice tone-similarity hyperparameter; 0.3 is the recommended live setting


def voice_deps_present() -> bool:
    """Cheap check — are torch + sounddevice + librosa + openvoice importable?"""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "torchaudio", "sounddevice", "librosa", "openvoice")
    )


def select_device() -> str:
    """Pick CUDA / MPS / CPU. Lazy-imports torch."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _converter_paths() -> tuple[Path, Path]:
    """Return (config_path, checkpoint_path) for OpenVoice's tone-color converter."""
    weights = openvoice_weights_dir()
    base = weights / CONVERTER_SUBDIR
    return base / CONFIG_FILENAME, base / CHECKPOINT_FILENAME


def _ensure_weights() -> None:
    cfg, ckpt = _converter_paths()
    if not cfg.exists() or not ckpt.exists():
        raise RuntimeError(
            "OpenVoice converter weights not found. "
            "Run `swap voices install` to download them."
        )


def _load_converter(device: str | None = None) -> tuple[Any, Any]:
    """Build the OpenVoice ToneColorConverter and load its checkpoint.

    Returns (converter, hps). hps holds the model's sample rate + spectrogram
    config we need for chunked conversion.
    """
    if not voice_deps_present():
        raise RuntimeError(
            "Voice deps not installed. Run `swap voices install` first."
        )

    _ensure_weights()
    cfg, ckpt = _converter_paths()

    # Lazy import.
    from openvoice.api import ToneColorConverter  # type: ignore[import-not-found]

    chosen = device or select_device()
    converter = ToneColorConverter(str(cfg), device=chosen)
    converter.load_ckpt(str(ckpt))
    return converter, converter.hps


# ── Embedding extraction (used at session-config time) ────────────────────


def extract_embedding(wav_path: Path | str, device: str | None = None) -> list[float]:
    """Extract a tone-color embedding from a reference WAV/MP3.

    Runs on CPU in a few seconds. Returns a 256-d list of floats that's
    safe to JSON-serialize (used by `swap voices add`).
    """
    converter, _ = _load_converter(device=device or "cpu")
    se, _audio_name = converter.extract_se(str(wav_path), vad=True)
    # se shape: (1, 256, 1) — flatten to a Python list.
    return se.detach().cpu().numpy().reshape(-1).tolist()


# ── Realtime conversion ────────────────────────────────────────────────────


class VoiceConverter:
    """Stateful tone-color converter for live mic streaming.

    Lifecycle:
      1. __init__: cheap — caches target embedding only, no model load.
      2. ensure_loaded(): loads OpenVoice + moves target SE to device.
      3. warm_up(source_audio, sample_rate): computes source SE from a
         few seconds of mic audio. Required before convert().
      4. convert(audio_chunk, sample_rate): runs source→target conversion.
      5. close(): drops the model + frees GPU memory.
    """

    def __init__(self, target_embedding: list[float], device: str | None = None) -> None:
        if not voice_deps_present():
            raise RuntimeError(
                "Voice deps not installed. Run `swap voices install` first."
            )
        self.target_embedding = target_embedding
        self._device_pref = device
        self._converter: Any = None
        self._hps: Any = None
        self._target_se: Any = None
        self._source_se: Any = None

    @property
    def device(self) -> str:
        if self._converter is None:
            return self._device_pref or "cpu"
        return str(self._converter.device)

    @property
    def is_warmed_up(self) -> bool:
        return self._source_se is not None

    @property
    def model_sample_rate(self) -> int:
        if self._hps is None:
            return 16_000
        return int(self._hps.data.sampling_rate)

    def ensure_loaded(self) -> None:
        """Load the OpenVoice model + place target SE on device."""
        if self._converter is not None:
            return
        import torch

        self._converter, self._hps = _load_converter(device=self._device_pref)
        target = torch.tensor(self.target_embedding, dtype=torch.float32)
        # OpenVoice expects (batch=1, embed=256, 1).
        self._target_se = target.reshape(1, 256, 1).to(self._converter.device)

    def warm_up(self, source_audio: "np.ndarray", sample_rate: int) -> None:
        """Compute source-side speaker embedding from accumulated mic audio.

        Call this with at least ~2–3 seconds of speech before calling
        convert(). Without warm_up, convert() passes audio through unchanged.
        """
        import tempfile

        import numpy as np
        import soundfile as sf

        self.ensure_loaded()

        # OpenVoice's extract_se reads from disk; cheapest path is a temp WAV.
        # The whole op runs once per session (~2s on GPU, ~5s on CPU) so the
        # disk hop is cheap.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, source_audio.astype(np.float32), sample_rate)
            try:
                se, _ = self._converter.extract_se(tmp.name, vad=True)
                self._source_se = se.to(self._converter.device)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def convert(self, audio_chunk: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Convert one mic chunk source→target. Pass-through until warmed up.

        Input/output: float32 numpy array in [-1, 1].
        Sample rate is converted to/from the model's native rate.
        """
        if not self.is_warmed_up or self._converter is None:
            return audio_chunk

        import numpy as np
        import torch
        from openvoice.mel_processing import spectrogram_torch  # type: ignore[import-not-found]

        target_sr = self.model_sample_rate

        # Resample if mic rate ≠ model rate. librosa is heavy but we already
        # depend on it; sounddevice can also resample on capture but doing
        # it here keeps the audio flow tight.
        if sample_rate != target_sr:
            import librosa

            audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=target_sr)

        with torch.no_grad():
            y = torch.from_numpy(audio_chunk.astype(np.float32)).unsqueeze(0).to(
                self._converter.device
            )
            spec = spectrogram_torch(
                y,
                self._hps.data.filter_length,
                self._hps.data.sampling_rate,
                self._hps.data.hop_length,
                self._hps.data.win_length,
                center=False,
            ).to(self._converter.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self._converter.device)

            converted = self._converter.model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=self._source_se,
                sid_tgt=self._target_se,
                tau=DEFAULT_TAU,
            )[0][0, 0].data.cpu().float().numpy()

        # Resample back to caller's rate so output matches the speaker stream.
        if sample_rate != target_sr:
            import librosa

            converted = librosa.resample(converted, orig_sr=target_sr, target_sr=sample_rate)

        return converted

    def close(self) -> None:
        """Drop the model and clear GPU cache."""
        self._converter = None
        self._hps = None
        self._target_se = None
        self._source_se = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
