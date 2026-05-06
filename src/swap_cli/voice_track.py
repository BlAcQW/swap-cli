"""Microphone capture + chunked voice conversion loop.

Sprint 13b.2: real sounddevice streaming + OpenVoice tone-color conversion.

Flow per session:
  1. Open RawInputStream on the user's mic at 16 kHz mono.
  2. Open RawOutputStream on the chosen virtual cable device (BlackHole /
     VB-Cable / pulse-loopback). If no output device is set, audio is
     dropped — convert path still runs so the user can hear via a separate
     "monitor through speakers" toggle if we add it.
  3. Warm-up phase: collect first ~2 s of mic audio, extract source SE
     once. During warm-up we pass through unchanged so the user hears no
     dropout.
  4. Live phase: each captured chunk goes mic → asyncio.Queue →
     converter → output stream.

Latency target: ≤200 ms end-to-end. The chunk size is the main knob —
1600 samples at 16 kHz = 100 ms windows. Larger windows are more stable
but feel laggy.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .voice_library import Voice
# voice_model imports removed — VoiceTrack now resolves its converter
# via the voice_engines registry (sprint 14b.2.a).

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = 16_000
# Sprint 14i: below this RMS, skip the converter entirely and emit
# zeros. Mirrors w-okada's silentThreshold. Catches breaths, ambient
# noise, keyboard taps — saves 1-2.4 s of inference cost per silent
# chunk on a 4060 mobile.
SILENT_THRESHOLD_RMS = 0.01
# Mic delivers 250 ms blocks. We then BUFFER into 2 s windows and emit
# 1 s hops. Per the upstream RVC realtime guide:
#   "Sample length 0.5–1.0s. Lower = less latency but worse quality and
#    more artifacts. Start at 1.0s, push down only if you need to."
# Sprint 14e bumped HOP from 500 ms → 1.0 s (quality > latency).
#
# Latency: ~1.5–2 s from speaking → being heard. Acceptable for a voice
# call; lower it later if users complain.
CHUNK_SAMPLES = 4_000  # 250 ms — sounddevice callback granularity
WINDOW_SAMPLES = 32_000  # 2 s — converter input window (1 s context + 1 s output)
HOP_SAMPLES = 16_000  # 1.0 s — slide / output rate
# SOLA (Synchronized Overlap-Add) — borrowed from w-okada/voice-changer.
# Cross-correlate the new converted chunk's leading region with the saved
# tail to find the optimal alignment offset, then crossfade THERE. Beats
# linear crossfade because it phase-aligns the chunks (no chorus artifact).
# 14e: bumped crossfade to 0.15 s per the RVC guide's sweet-spot range.
CROSSFADE_SAMPLES = 2_400  # 150 ms blend region
SOLA_SEARCH_SAMPLES = 200  # ~12 ms search range
WARM_UP_SECONDS = 3.0
WARM_UP_TARGET_SAMPLES = int(SAMPLE_RATE * WARM_UP_SECONDS)


def sola_blend(
    converted: "np.ndarray",
    sola_buffer: "np.ndarray | None",
    ramp_in: "np.ndarray",
    ramp_out: "np.ndarray",
    crossfade: int = CROSSFADE_SAMPLES,
    search: int = SOLA_SEARCH_SAMPLES,
    hop: int = HOP_SAMPLES,
) -> "tuple[np.ndarray, np.ndarray]":
    """One step of SOLA-aligned overlap-add streaming.

    Pure function — no I/O, no async. Extracted so it can be unit-tested
    against synthetic numpy arrays without needing torch / sounddevice /
    a real GPU.

    Args:
        converted: model's converted audio for the current window.
        sola_buffer: previous iteration's saved tail (ramp_out applied),
            or None on the first call.
        ramp_in: linear crossfade ramp 0→1, length `crossfade`.
        ramp_out: linear crossfade ramp 1→0, length `crossfade`.
        crossfade: blend region length in samples.
        search: cross-correlation search range in samples.
        hop: output samples per call.

    Returns:
        (output, next_sola_buffer) — output is `hop` samples ready to
        write to the audio sink; next_sola_buffer is the tail to pass
        back into this function on the next call.
    """
    import numpy as np

    if sola_buffer is None:
        # First window — no previous tail to align against. Emit the head.
        output = converted[:hop].copy()
    else:
        # Cross-correlate new converted output's leading (crossfade + search)
        # samples against the saved tail (ramp_out already applied).
        search_region = converted[: crossfade + search]
        cor_nom = np.convolve(search_region, np.flip(sola_buffer), "valid")
        cor_den = np.sqrt(
            np.convolve(
                search_region**2,
                np.ones(crossfade, dtype=np.float32),
                "valid",
            )
            + 1e-3
        )
        sola_offset = int(np.argmax(cor_nom / cor_den))

        # Output: hop samples starting at the optimal offset.
        end = min(sola_offset + hop, len(converted))
        output = converted[sola_offset:end].copy()
        if len(output) < hop:
            output = np.pad(output, (0, hop - len(output)))

        # Apply ramp_in to the first `crossfade` samples of output, then
        # add the saved (already ramp_out'd) tail.
        output[:crossfade] *= ramp_in
        output[:crossfade] += sola_buffer

    # Save next iteration's tail: last `crossfade` samples of the converted
    # window with ramp_out pre-applied.
    next_buffer = converted[-crossfade:].copy() * ramp_out
    return output, next_buffer


@dataclass(frozen=True)
class VoiceTrackOptions:
    voice: Voice
    microphone_device: int
    output_device: int | None  # None = drop converted audio (silent operation)
    play_through_speakers: bool = False
    # Sprint 14b.2.a: which engine handles streaming inference. Defaults
    # to OpenVoice (current path); RVC engine lands in 14b.2.b.
    engine_name: str = "rvc"
    # Sprint 14i: when True, set RVC index_rate=0 — disables Faiss
    # retrieval. Massive speedup on big-index voices (calm-man's 607MB
    # index dominates inference time). Slight quality loss; preferred
    # over chunk drops on weaker GPUs.
    fast: bool = False


class VoiceTrack:
    """Owns the mic capture + converter + virtual-cable output for one session.

    Lifecycle: __init__ (cheap) → start() (opens streams, kicks loop) →
    stop() (drains queue, closes streams). Lives in the same asyncio loop
    as the Decart video session.
    """

    def __init__(self, opts: VoiceTrackOptions) -> None:
        # Resolve the engine via the registry instead of importing voice_model
        # directly. Sprint 14b.2.a refactor — keeps this file engine-agnostic
        # so RVC drops in (14b.2.b) without further changes here.
        from . import voice_engines

        try:
            engine = voice_engines.get_engine(opts.engine_name)
        except KeyError as err:
            raise RuntimeError(
                f"Unknown voice engine '{opts.engine_name}'. "
                f"Known: {voice_engines.available_engines()}"
            ) from err

        if not engine.is_available():
            raise RuntimeError(
                f"Engine '{opts.engine_name}' isn't installed/available. "
                "Run `swap voices install` first."
            )

        self.opts = opts
        self._engine = engine
        self._converter = engine.make_converter(target_voice=opts.voice)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._on_status: Callable[[str], None] = lambda _s: None
        self._chunk_q: asyncio.Queue["np.ndarray"] | None = None
        self._loop_ref: asyncio.AbstractEventLoop | None = None

    def start(self, on_status: Callable[[str], None] | None = None) -> None:
        if on_status is not None:
            self._on_status = on_status
        self._loop_ref = asyncio.get_running_loop()
        # Sized for ~5s of headroom at 100ms chunks. Big enough to absorb
        # the OpenVoice warm-up extraction (~2-5s blocking) without losing
        # context, small enough to bound steady-state latency.
        self._chunk_q = asyncio.Queue(maxsize=50)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as err:  # noqa: BLE001
                print(f"[voice_track] error during stop: {err}", flush=True)
        self._converter.close()

    # ── Internals ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        import numpy as np
        import sounddevice as sd  # type: ignore[import-not-found]

        # ── Diagnostic prints (Sprint 14h) ─────────────────────────────
        # Every audio-pipeline boundary logs unconditionally so we can
        # tell from the terminal where silence originates: mic, converter,
        # or output.
        try:
            mic_info = sd.query_devices(self.opts.microphone_device, "input")
            print(
                f"[voice_track] mic device #{self.opts.microphone_device}: "
                f"'{mic_info['name']}', native rate {int(mic_info['default_samplerate'])}Hz, "
                f"max in channels {mic_info['max_input_channels']}",
                flush=True,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[voice_track] mic device query failed: {err}", flush=True)
        if self.opts.output_device is not None:
            try:
                out_info = sd.query_devices(self.opts.output_device, "output")
                print(
                    f"[voice_track] output device #{self.opts.output_device}: "
                    f"'{out_info['name']}', native rate {int(out_info['default_samplerate'])}Hz, "
                    f"max out channels {out_info['max_output_channels']}",
                    flush=True,
                )
            except Exception as err:  # noqa: BLE001
                print(f"[voice_track] output device query failed: {err}", flush=True)
        else:
            print(
                "[voice_track] ⚠ NO OUTPUT DEVICE — converted audio will be dropped. "
                "Install VB-Cable/BlackHole + restart, then pick it as Output in the GUI.",
                flush=True,
            )

        # Pre-load the model on the asyncio loop (a few seconds on cold cache).
        self._on_status("Voice: loading model…")
        print("[voice_track] loading model (first run downloads ~370 MB) …", flush=True)
        try:
            await asyncio.to_thread(self._converter.ensure_loaded)
        except Exception as err:  # noqa: BLE001
            self._on_status(f"Voice: load failed ({err})")
            print(f"[voice_track] model load failed: {err}", flush=True)
            return
        print("[voice_track] model loaded", flush=True)

        # Sprint 14i: apply fast mode (skip Faiss retrieval) if requested.
        if self.opts.fast and hasattr(self._converter, "set_index_rate"):
            try:
                self._converter.set_index_rate(0.0)
                print(
                    "[voice_track] FAST mode: index_rate=0 (no Faiss retrieval — "
                    "faster but less accurate timbre)",
                    flush=True,
                )
            except Exception as err:  # noqa: BLE001
                print(f"[voice_track] failed to enable fast mode: {err}", flush=True)

        self._on_status("Voice: capturing… (warm-up ~2s)")

        # Bridge: sounddevice's input callback runs on PortAudio's thread.
        # We need to hand chunks back into the asyncio loop safely.
        loop = self._loop_ref
        chunk_q = self._chunk_q
        assert loop is not None and chunk_q is not None

        # Counters used by the input callback for periodic RMS reporting.
        in_chunk_count: list[int] = [0]

        def _put_on_loop(chunk: "np.ndarray") -> None:
            """Runs on the asyncio loop — safe to drain the queue here.

            If the GPU is falling behind (queue full), drop the oldest
            chunks to keep latency bounded. A brief glitch is better than
            ever-growing lag.
            """
            while chunk_q.full():
                try:
                    chunk_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                chunk_q.put_nowait(chunk)
            except asyncio.QueueFull:
                # Shouldn't happen after the drain above, but swallow just
                # in case to keep the PortAudio callback chain healthy.
                pass

        def _input_cb(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
            if status:
                print(f"[voice_track] input status: {status}", flush=True)
            # indata is shape (frames, channels). We capture mono.
            chunk = np.frombuffer(bytes(indata), dtype="float32").copy()
            in_chunk_count[0] += 1
            # Log first chunk + every 40th (~10s @ 250ms) — proves mic is live.
            n = in_chunk_count[0]
            if n == 1 or n % 40 == 0:
                rms = float(np.sqrt(np.mean(chunk**2)))
                peak = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0
                tag = "FIRST" if n == 1 else f"#{n}"
                print(
                    f"[voice_track] mic chunk {tag}: {len(chunk)} samples, "
                    f"RMS={rms:.4f}, peak={peak:.4f}"
                    + ("  ⚠ SILENT — wrong mic device?" if rms < 1e-5 else ""),
                    flush=True,
                )
            # Hand off to the asyncio loop. Drop-oldest happens inside
            # _put_on_loop, NOT in this PortAudio thread (asyncio.Queue
            # methods aren't thread-safe).
            loop.call_soon_threadsafe(_put_on_loop, chunk)

        try:
            in_stream = sd.RawInputStream(
                device=self.opts.microphone_device,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=_input_cb,
            )
        except Exception as err:  # noqa: BLE001
            print(
                f"[voice_track] ✗ failed to open mic stream at {SAMPLE_RATE}Hz: {err}",
                flush=True,
            )
            self._on_status(f"Voice: mic open failed ({err})")
            return
        out_stream = None
        if self.opts.output_device is not None:
            try:
                out_stream = sd.RawOutputStream(
                    device=self.opts.output_device,
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    dtype="float32",
                    blocksize=CHUNK_SAMPLES,
                )
            except Exception as err:  # noqa: BLE001
                print(
                    f"[voice_track] ✗ failed to open output stream at {SAMPLE_RATE}Hz: {err}",
                    flush=True,
                )
                self._on_status(f"Voice: output open failed ({err})")
                # Continue without output — at least the conversion runs.
                out_stream = None

        in_stream.start()
        if out_stream is not None:
            out_stream.start()
        print(
            f"[voice_track] streams started: mic 16kHz mono, "
            f"output={'16kHz mono' if out_stream else 'NONE (silent)'}",
            flush=True,
        )

        warm_buffer: list[float] = []
        # SOLA streaming state. We accumulate input into a sliding
        # WINDOW_SAMPLES window, run convert(), then phase-align the new
        # converted output against the previous chunk's saved tail before
        # crossfading. Adapted from w-okada/voice-changer's VoiceChangerV2:
        # cross-correlate the new output's leading region with the saved
        # tail, find argmax → that's the optimal blend offset.
        input_buf = np.zeros(0, dtype=np.float32)
        sola_buffer: np.ndarray | None = None  # last CROSSFADE_SAMPLES of prev output, ramp_out applied
        # Crossfade ramps for the SOLA blend region (much shorter than HOP).
        ramp_in = np.linspace(0.0, 1.0, CROSSFADE_SAMPLES, dtype=np.float32)
        ramp_out = 1.0 - ramp_in
        ticks = 0

        try:
            while not self._stop.is_set():
                try:
                    chunk = await asyncio.wait_for(chunk_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Warm-up: collect mic audio, extract source SE once.
                if not self._converter.is_warmed_up:
                    warm_buffer.extend(chunk.tolist())
                    if len(warm_buffer) >= WARM_UP_TARGET_SAMPLES:
                        warmup_audio = np.array(
                            warm_buffer[:WARM_UP_TARGET_SAMPLES], dtype=np.float32
                        )
                        warm_buffer = []
                        print(
                            "[voice_track] warm-up: extracting source SE "
                            f"({WARM_UP_TARGET_SAMPLES} samples)…",
                            flush=True,
                        )
                        self._on_status("Voice: extracting source identity…")
                        try:
                            await asyncio.to_thread(
                                self._converter.warm_up, warmup_audio, SAMPLE_RATE
                            )
                            print(
                                f"[voice_track] live · {self.opts.voice.name} "
                                f"(out_device={self.opts.output_device})",
                                flush=True,
                            )
                            self._on_status(f"Voice: live ({self.opts.voice.name})")
                        except Exception as err:  # noqa: BLE001
                            print(f"[voice_track] warm-up failed: {err}", flush=True)
                            self._on_status(f"Voice: warm-up failed ({err})")
                            return
                    # Pass-through during warm-up so the user isn't muted.
                    if out_stream is not None:
                        try:
                            out_stream.write(chunk.astype(np.float32).tobytes())
                        except Exception:
                            pass
                    continue

                # ── Overlap-add streaming inference ──────────────────────
                input_buf = np.concatenate([input_buf, chunk])

                while len(input_buf) >= WINDOW_SAMPLES:
                    window = input_buf[:WINDOW_SAMPLES].copy()
                    input_buf = input_buf[HOP_SAMPLES:]
                    win_rms = float(np.sqrt(np.mean(window**2)))

                    import time as _time

                    t0 = _time.perf_counter()
                    if win_rms < SILENT_THRESHOLD_RMS:
                        # Silent gate (Sprint 14i, w-okada-inspired). Skip
                        # the model entirely — no point running Hubert +
                        # F0 + retrieval on ambient noise. Emit zeros.
                        converted = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                    else:
                        try:
                            converted = await asyncio.to_thread(
                                self._converter.convert, window, SAMPLE_RATE
                            )
                        except Exception as err:  # noqa: BLE001
                            print(f"[voice_track] convert error: {err}", flush=True)
                            converted = window  # pass-through on failure
                    convert_ms = (_time.perf_counter() - t0) * 1000.0
                    out_rms = (
                        float(np.sqrt(np.mean(converted**2)))
                        if converted.size > 0
                        else 0.0
                    )

                    # First convert + every 10th: prove inference is producing audio.
                    if ticks == 0 or (ticks + 1) % 10 == 0:
                        flag = ""
                        if win_rms > 1e-4 and out_rms < 1e-5:
                            flag = "  ⚠ INPUT HAS AUDIO BUT CONVERTER RETURNED SILENCE"
                        elif convert_ms > 1500:
                            flag = (
                                f"  ⚠ SLOW: {convert_ms:.0f}ms > 1000ms hop, "
                                "queue will drop chunks"
                            )
                        tag = "FIRST" if ticks == 0 else f"#{ticks + 1}"
                        print(
                            f"[voice_track] convert {tag}: in_RMS={win_rms:.4f} → "
                            f"out_RMS={out_rms:.4f} in {convert_ms:.0f}ms{flag}",
                            flush=True,
                        )

                    # Pad/truncate to expected window length so slicing is safe.
                    if len(converted) < WINDOW_SAMPLES:
                        converted = np.pad(
                            converted, (0, WINDOW_SAMPLES - len(converted))
                        )
                    elif len(converted) > WINDOW_SAMPLES:
                        converted = converted[:WINDOW_SAMPLES]

                    output, sola_buffer = sola_blend(
                        converted, sola_buffer, ramp_in, ramp_out
                    )

                    write_ok = False
                    write_err: str | None = None
                    if out_stream is not None:
                        try:
                            out_stream.write(output.astype(np.float32).tobytes())
                            write_ok = True
                        except Exception as err:  # noqa: BLE001
                            write_err = str(err)
                            print(
                                f"[voice_track] output write failed: {err}",
                                flush=True,
                            )

                    # First write + every 10th: confirms output device is being fed.
                    if ticks == 0 or (ticks + 1) % 10 == 0:
                        if out_stream is None:
                            note = "no output device — converted audio dropped"
                        elif write_ok:
                            note = f"wrote {len(output)} samples"
                        else:
                            note = f"WRITE FAILED ({write_err})"
                        tag = "FIRST" if ticks == 0 else f"#{ticks + 1}"
                        print(f"[voice_track] output {tag}: {note}", flush=True)

                    ticks += 1
                    if ticks % 10 == 0:  # one tick per HOP (~500 ms) → ~5 s
                        secs = (ticks * HOP_SAMPLES) // SAMPLE_RATE
                        print(f"[voice_track] live · {secs}s", flush=True)
                        self._on_status(f"Voice: live · {secs}s")
        except asyncio.CancelledError:
            raise
        finally:
            try:
                in_stream.stop()
                in_stream.close()
            except Exception:
                pass
            if out_stream is not None:
                try:
                    out_stream.stop()
                    out_stream.close()
                except Exception:
                    pass
