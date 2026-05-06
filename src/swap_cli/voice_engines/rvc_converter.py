"""Streaming RVC converter — wraps rvc-python's batch infer_file API.

WHY THIS IS THE WAY IT IS

`rvc-python` exposes only `RVCInference.infer_file(input_path, output_path)`
— path in, path out. There's no in-memory or chunk-by-chunk method on its
public surface. For streaming we have two options:

1. (THIS FILE) Per-chunk temp WAV: write the input chunk to a tempfile,
   call infer_file, read the output, return the numpy array. ~5-15 ms of
   I/O overhead per chunk on SSD; acceptable but not ideal.

2. (NOT YET) Reach into rvc-python's internals — call self.vc.pipeline
   directly with numpy arrays. Faster but couples us to private API.

This file ships option 1. v1 ships, then we measure latency on real
hardware and decide whether to rewrite.

LIMITATIONS
- Latency: chunk size + ~150 ms model + ~10 ms I/O. Higher than the
  ~50-100 ms you'd get with a streaming-native pipeline.
- 'Warm-up': RVCInference loads the model at __init__ time, so warm_up()
  is effectively a no-op for RVC (no per-session source-SE extraction
  like OpenVoice needs). We expose the protocol method for parity.
- Source SE: RVC doesn't use one. The hubert encoder + F0 are the
  source-side processing, and they run per-chunk inside infer_file.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class RVCConverter:
    """RVC voice conversion via rvc-python, fronted with a temp-file shim."""

    def __init__(
        self,
        model_pth: Path,
        index_pth: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_pth = model_pth
        self.index_pth = index_pth
        self._device = device
        self._inference: Any = None
        self._loaded = False
        self._warmed_up = False  # RVC doesn't need per-session warm-up; no-op
        self._sample_rate = 16_000  # RVC pipeline operates internally at 16k

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def model_sample_rate(self) -> int:
        return self._sample_rate

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        from rvc_python.infer import RVCInference  # type: ignore[import-not-found]

        device = self._device or _select_rvc_device()
        # rvc-python expects 'cuda:0' / 'cpu:0' style device strings.
        if device == "cuda":
            device = "cuda:0"
        elif device == "cpu":
            device = "cpu:0"

        # rvc-python places downloaded weights into models_dir; we put them
        # next to the user's RVC voice models so swap-cli has one tree.
        from ..voice_prereq import rvc_models_dir

        models_dir = str(rvc_models_dir())
        Path(models_dir).mkdir(parents=True, exist_ok=True)

        # Sprint 14e: use fp16 on CUDA — free 2x speedup on RTX 30/40-series
        # with no audible quality loss (per the RVC realtime guide). On CPU
        # fp16 is unsupported, so fall back to fp32 there.
        is_half = device.startswith("cuda")
        try:
            self._inference = RVCInference(
                models_dir=models_dir, device=device, is_half=is_half
            )
        except TypeError:
            # Older rvc-python builds don't expose is_half on the constructor.
            self._inference = RVCInference(models_dir=models_dir, device=device)
        self._inference.load_model(
            str(self.model_pth),
            index_path=str(self.index_pth) if self.index_pth is not None else "",
        )
        self._loaded = True

    def warm_up(self, source_audio: "np.ndarray", sample_rate: int) -> None:
        """RVC has no per-session source extraction step. Just mark warmed."""
        self.ensure_loaded()
        self._warmed_up = True

    def set_index_rate(self, rate: float) -> None:
        """Tune the Faiss retrieval mix. 0.0 disables retrieval (fast path),
        0.75 is rvc-python's default (better timbre, slower on big indexes)."""
        if not self._loaded:
            self.ensure_loaded()
        self._inference.set_params(index_rate=float(max(0.0, min(1.0, rate))))

    def convert(self, audio_chunk: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Run one chunk through RVC.

        Sprint 14i: skip rvc-python's wasteful temp-file roundtrip on the
        OUTPUT side. infer_file writes wav_opt to disk then we'd re-read
        it; instead we call vc.vc_single directly (which is what infer_file
        does internally) and use the returned numpy array directly. We
        still write the INPUT to a temp file because rvc-python's load_audio
        only accepts a path.

        Returns the converted chunk as float32 numpy at `sample_rate`.
        On failure, returns the input unchanged (pass-through).
        """
        if not self._loaded:
            self.ensure_loaded()

        import numpy as np
        import soundfile as sf

        # Reserve temp path without holding open handles (Windows lock).
        ts = int(time.time() * 1000)
        pid = os.getpid()
        tmp_dir = Path(tempfile.gettempdir())
        in_path = tmp_dir / f"swap-rvc-in-{pid}-{ts}.wav"

        inf = self._inference
        try:
            sf.write(str(in_path), audio_chunk.astype(np.float32), sample_rate)

            model_info = inf.models[inf.current_model]
            file_index = model_info.get("index", "")

            wav_opt = inf.vc.vc_single(
                sid=0,
                input_audio_path=str(in_path),
                f0_up_key=inf.f0up_key,
                f0_method=inf.f0method,
                file_index=file_index,
                index_rate=inf.index_rate,
                filter_radius=inf.filter_radius,
                resample_sr=inf.resample_sr,
                rms_mix_rate=inf.rms_mix_rate,
                protect=inf.protect,
                f0_file="",
                file_index2="",
            )

            # vc_single returns a numpy array (int16 typically) on success,
            # or a tuple ("error", (None, None)) on failure inside its try.
            if isinstance(wav_opt, tuple):
                print(
                    f"[rvc_converter] vc_single failed: {wav_opt[0][:200]}",
                    flush=True,
                )
                return audio_chunk

            audio = np.asarray(wav_opt)
            if audio.dtype.kind == "i":
                # int16 → float32 in [-1, 1]
                converted = audio.astype(np.float32) / 32768.0
            else:
                converted = audio.astype(np.float32)

            # rvc-python's pipeline outputs at vc.tgt_sr (model's native rate,
            # usually 40000 or 48000 for v2 models). Resample to caller's rate.
            out_sr = int(getattr(inf.vc, "tgt_sr", sample_rate))
            if out_sr != sample_rate:
                import librosa

                converted = librosa.resample(
                    converted, orig_sr=out_sr, target_sr=sample_rate
                )
            return converted
        except Exception as err:  # noqa: BLE001
            print(f"[rvc_converter] convert error: {err}", flush=True)
            return audio_chunk
        finally:
            in_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._inference is not None:
            try:
                self._inference.unload_model()
            except Exception:  # noqa: BLE001
                pass
        self._inference = None
        self._loaded = False
        self._warmed_up = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _select_rvc_device() -> str:
    """CUDA > MPS > CPU. rvc-python doesn't support MPS in its current
    release, so MPS callers get CPU until that changes."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
