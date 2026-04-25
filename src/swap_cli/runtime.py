"""Orchestrates a single Decart Lucy 2 realtime session end-to-end.

Camera → aiortc → Decart → remote track → cv2 window.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .camera import CameraTrack
from .display import Display, default_recording_path

DEFAULT_PROMPT = (
    "Substitute the person in the video with the person in the reference image. "
    "Match their face, body, and identity while preserving the original pose, "
    "motion, and lighting."
)

DEFAULT_MODEL_NAME = "lucy-2"


@dataclass(frozen=True)
class RunOptions:
    decart_api_key: str
    reference: str | None  # path or URL of the reference identity image
    prompt: str
    model_name: str = DEFAULT_MODEL_NAME
    camera_device: int = 0
    record: Path | None = None


async def run_session(opts: RunOptions) -> None:
    """Open a realtime Decart session and stream until the user quits."""
    # Lazy import — `decart` and `aiortc` pull in heavy native deps and we
    # don't want them loaded for `swap version` / `swap config` / etc.
    from decart import DecartClient, models  # type: ignore[import-not-found]
    from decart.realtime import (  # type: ignore[import-not-found]
        RealtimeClient,
        RealtimeConnectOptions,
    )
    from decart.types import ModelState, Prompt  # type: ignore[import-not-found]

    model = models.realtime(opts.model_name)
    camera = CameraTrack(
        device=opts.camera_device,
        width=int(model.width),
        height=int(model.height),
        fps=int(model.fps),
    )

    client = DecartClient(api_key=opts.decart_api_key)
    quit_event = asyncio.Event()

    realtime_client: Any = None
    display: Display | None = None
    try:
        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=camera,
            options=RealtimeConnectOptions(
                model=model,
                on_remote_stream=lambda remote_track: _on_remote_stream(
                    remote_track,
                    record=opts.record,
                    quit_event=quit_event,
                    display_box=display_box,
                ),
                initial_state=ModelState(
                    prompt=Prompt(text=opts.prompt, enhance=True),
                ),
            ),
        )

        def _on_connection_change(state: str) -> None:
            print(f"[runtime] connection: {state}")
            if state == "disconnected":
                quit_event.set()

        def _on_error(error: Any) -> None:
            print(f"[runtime] error: {getattr(error, 'message', error)}")
            quit_event.set()

        def _on_tick(message: Any) -> None:
            seconds = getattr(message, "seconds", 0)
            # carriage return so we update in-place
            print(f"\rstreaming · {seconds}s", end="", flush=True)

        realtime_client.on("connection_change", _on_connection_change)
        realtime_client.on("error", _on_error)
        realtime_client.on("generation_tick", _on_tick)

        # Apply the reference identity (URL or local file path; bytes also OK).
        if opts.reference:
            await realtime_client.set(
                _build_set_input(opts.prompt, opts.reference),
            )

        await quit_event.wait()
        print()  # newline after the tick line
    finally:
        if realtime_client is not None:
            with suppress(Exception):
                await realtime_client.disconnect()
        # Display is started inside `_on_remote_stream` and stored in display_box[0].
        if display_box and display_box[0] is not None:
            with suppress(Exception):
                await display_box[0].stop()
        with suppress(Exception):
            await client.close()
        camera.stop()


# Mutable container so the on_remote_stream callback can hand the display
# back to the run_session frame for cleanup. Cleaner than nonlocal in a lambda.
display_box: list[Display | None] = [None]


def _on_remote_stream(
    remote_track: Any,
    *,
    record: Path | None,
    quit_event: asyncio.Event,
    display_box: list[Display | None],
) -> None:
    record_path = record if record is not None else None
    if record is not None and not record.is_absolute():
        record_path = record
    if record == Path("auto"):
        record_path = default_recording_path()

    disp = Display(
        track=remote_track,
        record_path=record_path,
        on_quit=quit_event.set,
    )
    disp.start()
    display_box[0] = disp


def _build_set_input(prompt: str, reference: str) -> Any:
    """Build a `SetInput` accepting either a URL string or a local file path."""
    from decart import SetInput  # type: ignore[import-not-found]

    p = Path(reference)
    image: Any
    if p.exists() and p.is_file():
        image = p.read_bytes()
    else:
        image = reference  # treat as URL

    return SetInput(prompt=prompt, image=image, enhance=True)
