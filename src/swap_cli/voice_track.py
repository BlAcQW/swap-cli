"""Microphone capture + chunked voice conversion loop.

Mirrors camera.py's shape: opens an input device, pulls fixed-size chunks,
runs each through the converter, hands the output to a router/queue.

13b.1: scaffolding only. Without sounddevice installed (i.e. user hasn't
run `swap voices install`), the constructor raises a clear RuntimeError.
With it installed, capture works but `convert()` is the no-op pass-through
from voice_model.VoiceConverter.

13b.2: real-time inference + warm-up + virtual cable output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .voice_library import Voice
from .voice_model import VoiceConverter, voice_deps_present

if TYPE_CHECKING:
    import numpy as np

# Keep chunk small enough for sub-200ms perceived latency, large enough
# that the GPU isn't constantly cold. 100ms at 16kHz = 1600 samples.
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_600
CHUNK_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE


@dataclass(frozen=True)
class VoiceTrackOptions:
    voice: Voice  # the target voice (id, embedding, etc.)
    microphone_device: int  # sounddevice input device index
    output_device: int | None  # sounddevice output device for routing (None = no routing)
    play_through_speakers: bool = False  # also write to default speakers


class VoiceTrack:
    """Owns the mic capture loop + converter + output routing.

    Lifecycle: __init__ (cheap) → start() (opens streams, starts loop) →
    stop() (closes streams). Run inside the same asyncio loop as the
    Decart video session so they cancel together cleanly.
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

    def start(self, on_status: callable | None = None) -> None:  # type: ignore[valid-type]
        self._on_status = on_status or (lambda s: None)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._converter.close()

    async def _loop(self) -> None:
        """Capture → convert → route loop. 13b.1 stub.

        13b.2 will use sounddevice.RawInputStream + RawOutputStream with
        a callback that posts numpy chunks into an asyncio.Queue, then
        consume the queue here to convert and write to the output stream.
        """
        # 13b.2: open sounddevice input + output streams, warm_up, loop:
        #   while not stop:
        #     chunk = await queue.get()
        #     converted = self._converter.convert(chunk, SAMPLE_RATE)
        #     output_stream.write(converted)
        #     post latency metric to on_status
        try:
            print(
                f"[voice_track] (13b.1 stub) target voice: {self.opts.voice.name}, "
                f"mic={self.opts.microphone_device}, "
                f"out={self.opts.output_device}",
                flush=True,
            )
            self._on_status("Voice: ready (no-op until 13b.2)")
            await self._stop.wait()
        except asyncio.CancelledError:
            raise
