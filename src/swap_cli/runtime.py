"""Orchestrates a single Decart Lucy 2 realtime session end-to-end.

Camera → aiortc → Decart → remote track → cv2 window.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .camera import CameraTrack
from .display import Display, default_recording_path

CONNECT_TIMEOUT_S = 20.0


def _noop_status(_msg: str) -> None:
    return None

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
    # Optional callback for surfacing state to a UI status bar. Always called
    # from the asyncio worker thread; the GUI is responsible for marshalling
    # back to the tk main thread (e.g. via root.after).
    on_status_change: Callable[[str], None] = field(default=_noop_status)
    # Optional callback that hands the caller a thread-safe stop function once
    # the session is set up. Calling that function from any thread cleanly
    # winds down the live session.
    on_runtime_ready: Callable[[Callable[[], None]], None] | None = None


async def run_session(opts: RunOptions) -> None:
    """Open a realtime Decart session and stream until the user quits."""
    print("[runtime] entering run_session", flush=True)

    # Lazy import — `decart` and `aiortc` pull in heavy native deps and we
    # don't want them loaded for `swap version` / `swap config` / etc.
    from decart import DecartClient, models  # type: ignore[import-not-found]
    from decart.realtime import (  # type: ignore[import-not-found]
        RealtimeClient,
        RealtimeConnectOptions,
    )
    from decart.types import ModelState, Prompt  # type: ignore[import-not-found]

    print("[runtime] imports ok", flush=True)
    model = models.realtime(opts.model_name)
    print(f"[runtime] model={opts.model_name} {model.width}x{model.height}@{model.fps}fps", flush=True)

    camera = CameraTrack(
        device=opts.camera_device,
        width=int(model.width),
        height=int(model.height),
        fps=int(model.fps),
    )
    print(f"[runtime] camera opened on device {opts.camera_device}", flush=True)

    client = DecartClient(api_key=opts.decart_api_key)
    quit_event = asyncio.Event()
    notify = opts.on_status_change

    # Hand the caller a thread-safe stop function. The GUI's Stop button
    # calls this from the tk main thread; we schedule quit_event.set() on
    # this asyncio loop via call_soon_threadsafe.
    if opts.on_runtime_ready is not None:
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            loop.call_soon_threadsafe(quit_event.set)

        opts.on_runtime_ready(_request_stop)

    realtime_client: Any = None
    display: Display | None = None
    try:
        print("[runtime] about to call RealtimeClient.connect", flush=True)
        notify("Negotiating Decart session…")
        try:
            realtime_client = await asyncio.wait_for(
                RealtimeClient.connect(
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
                ),
                timeout=CONNECT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Decart connection timed out after {CONNECT_TIMEOUT_S:.0f}s. "
                "Check your API key and that UDP traffic is allowed by your firewall."
            ) from exc

        def _on_connection_change(state: str) -> None:
            print(f"[runtime] connection: {state}")
            notify(f"Connection: {state}")
            if state == "disconnected":
                quit_event.set()

        def _on_error(error: Any) -> None:
            msg = getattr(error, "message", str(error))
            print(f"[runtime] error: {msg}")
            notify(f"Error: {msg}")
            quit_event.set()

        def _on_tick(message: Any) -> None:
            seconds = getattr(message, "seconds", 0)
            # carriage return so we update in-place
            print(f"\rstreaming · {seconds}s", end="", flush=True)
            notify(f"Live · {seconds}s")

        realtime_client.on("connection_change", _on_connection_change)
        realtime_client.on("error", _on_error)
        realtime_client.on("generation_tick", _on_tick)

        # Apply the reference identity (URL or local file path; bytes also OK).
        if opts.reference:
            notify("Applying reference identity…")
            await realtime_client.set(
                _build_set_input(opts.prompt, opts.reference),
            )

        notify("Connected · waiting for first frame…")
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
