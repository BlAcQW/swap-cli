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
CHUNK_SAMPLES = 1_600  # 100 ms at 16 kHz
WARM_UP_SECONDS = 2.0
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
                        self._on_status("Voice: extracting source identity…")
                        try:
                            await asyncio.to_thread(
                                self._converter.warm_up, warmup_audio, SAMPLE_RATE
                            )
                            self._on_status(f"Voice: live ({self.opts.voice.name})")
                        except Exception as err:  # noqa: BLE001
                            self._on_status(f"Voice: warm-up failed ({err})")
                            print(f"[voice_track] warm-up failed: {err}", flush=True)
                            return
                    converted = chunk
                else:
                    try:
                        converted = await asyncio.to_thread(
                            self._converter.convert, chunk, SAMPLE_RATE
                        )
                    except Exception as err:  # noqa: BLE001
                        # Don't kill the session over a single chunk failure;
                        # pass-through and keep going.
                        print(f"[voice_track] convert error: {err}", flush=True)
                        converted = chunk

                if out_stream is not None:
                    try:
                        out_stream.write(converted.astype(np.float32).tobytes())
                    except Exception as err:  # noqa: BLE001
                        print(f"[voice_track] output write failed: {err}", flush=True)

                ticks += 1
                if ticks % 50 == 0:  # every ~5 s
                    self._on_status(f"Voice: live · {ticks // 10}s")
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
