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
from .voice_model import VoiceConverter, voice_deps_present

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = 16_000
# Mic delivers 250 ms blocks. We then BUFFER into 1 s windows with 50%
# overlap and run the converter once per window. Output uses linear
# crossfade between adjacent windows so words don't get cut mid-syllable
# at chunk boundaries (the cause of "clean but garbled" output).
#
# Latency: ~1 s from speaking → being heard. Acceptable for a voice call.
CHUNK_SAMPLES = 4_000  # 250 ms — sounddevice callback granularity
WINDOW_SAMPLES = 16_000  # 1 s — converter input window
HOP_SAMPLES = 8_000  # 500 ms — slide / output rate (50% overlap)
WARM_UP_SECONDS = 3.0
WARM_UP_TARGET_SAMPLES = int(SAMPLE_RATE * WARM_UP_SECONDS)


@dataclass(frozen=True)
class VoiceTrackOptions:
    voice: Voice
    microphone_device: int
    output_device: int | None  # None = drop converted audio (silent operation)
    play_through_speakers: bool = False


class VoiceTrack:
    """Owns the mic capture + converter + virtual-cable output for one session.

    Lifecycle: __init__ (cheap) → start() (opens streams, kicks loop) →
    stop() (drains queue, closes streams). Lives in the same asyncio loop
    as the Decart video session.
    """

    def __init__(self, opts: VoiceTrackOptions) -> None:
        if not voice_deps_present():
            raise RuntimeError(
                "Voice deps not installed. Run `swap voices install` first."
            )
        self.opts = opts
        self._converter = VoiceConverter(target_embedding=opts.voice.embedding)
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

        # Pre-load the model on the asyncio loop (a few seconds on cold cache).
        self._on_status("Voice: loading OpenVoice…")
        try:
            await asyncio.to_thread(self._converter.ensure_loaded)
        except Exception as err:  # noqa: BLE001
            self._on_status(f"Voice: load failed ({err})")
            print(f"[voice_track] model load failed: {err}", flush=True)
            return

        self._on_status("Voice: capturing… (warm-up ~2s)")

        # Bridge: sounddevice's input callback runs on PortAudio's thread.
        # We need to hand chunks back into the asyncio loop safely.
        loop = self._loop_ref
        chunk_q = self._chunk_q
        assert loop is not None and chunk_q is not None

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
            # Hand off to the asyncio loop. Drop-oldest happens inside
            # _put_on_loop, NOT in this PortAudio thread (asyncio.Queue
            # methods aren't thread-safe).
            loop.call_soon_threadsafe(_put_on_loop, chunk)

        in_stream = sd.RawInputStream(
            device=self.opts.microphone_device,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=_input_cb,
        )
        out_stream = None
        if self.opts.output_device is not None:
            out_stream = sd.RawOutputStream(
                device=self.opts.output_device,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
            )

        in_stream.start()
        if out_stream is not None:
            out_stream.start()

        warm_buffer: list[float] = []
        # Overlap-add streaming state. We accumulate input into a sliding
        # WINDOW_SAMPLES window, run convert(), then crossfade the first
        # HOP_SAMPLES of each new converted window with the saved tail of
        # the previous window. Smooths phoneme boundaries that 500 ms
        # non-overlapping chunks were chopping mid-syllable.
        input_buf = np.zeros(0, dtype=np.float32)
        prev_tail: np.ndarray | None = None  # last (WINDOW − HOP) samples
        # Linear crossfade ramps. Hann would be marginally smoother but
        # linear is shorter to read and good enough for speech.
        ramp_in = np.linspace(0.0, 1.0, HOP_SAMPLES, dtype=np.float32)
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

                    try:
                        converted = await asyncio.to_thread(
                            self._converter.convert, window, SAMPLE_RATE
                        )
                    except Exception as err:  # noqa: BLE001
                        print(f"[voice_track] convert error: {err}", flush=True)
                        converted = window  # pass-through on failure

                    # Pad/truncate to expected window length so slicing is safe.
                    if len(converted) < WINDOW_SAMPLES:
                        converted = np.pad(
                            converted, (0, WINDOW_SAMPLES - len(converted))
                        )
                    elif len(converted) > WINDOW_SAMPLES:
                        converted = converted[:WINDOW_SAMPLES]

                    head = converted[:HOP_SAMPLES]
                    new_tail = converted[HOP_SAMPLES:].copy()

                    if prev_tail is None:
                        # First window — output the head directly, no blend.
                        output = head
                    else:
                        # Crossfade: prev tail fading out, new head fading in.
                        output = (prev_tail * ramp_out) + (head * ramp_in)

                    prev_tail = new_tail

                    if out_stream is not None:
                        try:
                            out_stream.write(output.astype(np.float32).tobytes())
                        except Exception as err:  # noqa: BLE001
                            print(f"[voice_track] output write failed: {err}", flush=True)

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
