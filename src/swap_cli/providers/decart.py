"""Decart provider — the original (and default) backend.

This is the code that used to live inline in `runtime.run_session`, moved behind
the `Provider` contract with NO behaviour change: same lazy imports, same
version guard, same connect options, same timeout message, same teardown order.
The existing test suite is the regression guard for that claim.
"""

from __future__ import annotations

from contextlib import suppress
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import FrameSpec

if TYPE_CHECKING:
    from collections.abc import Callable

# Decart's realtime WebRTC negotiation can sit unanswered; fail with guidance
# rather than hanging the GUI forever.
CONNECT_TIMEOUT_S = 45.0


def _guard_decart_version() -> None:
    """decart >= 0.1 is a LiveKit rewrite that expects a LiveKit-native track,
    but we feed it an aiortc CameraTrack ('CameraTrack' object has no attribute
    '_ffi_handle'). Fail fast with the fix instead of retry-looping."""
    try:
        dv = _pkg_version("decart")
        dv_mm = tuple(int(p) for p in dv.split(".")[:2])
    except (PackageNotFoundError, ValueError):
        return
    if dv_mm >= (0, 1):
        raise RuntimeError(
            f"decart {dv} uses the new LiveKit API, which swap-cli doesn't support "
            'yet. Run:  pip install "decart==0.0.40"   then relaunch.'
        )


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


class _DecartSession:
    def __init__(self, client: Any, realtime_client: Any) -> None:
        self._client = client
        self._realtime = realtime_client

    async def set_identity(self, prompt: str, reference: str) -> None:
        await self._realtime.set(_build_set_input(prompt, reference))

    async def disconnect(self) -> None:
        import asyncio

        # aiortc's pc.close() can raise CancelledError (a BaseException, NOT
        # caught by suppress(Exception)) during teardown — suppress both so a
        # drop tears down cleanly instead of escaping as a traceback.
        with suppress(Exception, asyncio.CancelledError):
            await self._realtime.disconnect()
        with suppress(Exception, asyncio.CancelledError):
            await self._client.close()


class DecartProvider:
    """Realtime video-to-video on Decart (Lucy), over WebRTC."""

    name = "decart"

    def frame_spec(self, opts: Any) -> FrameSpec:
        _guard_decart_version()
        from decart import models  # type: ignore[import-not-found]

        model = models.realtime(opts.model_name)
        print(
            f"[runtime] model={opts.model_name} "
            f"{model.width}x{model.height}@{model.fps}fps",
            flush=True,
        )
        return FrameSpec(int(model.width), int(model.height), int(model.fps))

    async def connect(
        self,
        *,
        camera: Any,
        opts: Any,
        on_remote_stream: Callable[[Any], None],
        notify: Callable[[str], None],
        quit_event: Any,
    ) -> _DecartSession:
        import asyncio

        # Lazy import — decart/aiortc pull heavy native deps we don't want
        # loaded for `swap version` / `swap config` / etc.
        from decart import DecartClient, models  # type: ignore[import-not-found]
        from decart.realtime import (  # type: ignore[import-not-found]
            RealtimeClient,
            RealtimeConnectOptions,
        )
        from decart.types import ModelState, Prompt  # type: ignore[import-not-found]

        print("[runtime] imports ok", flush=True)
        model = models.realtime(opts.model_name)
        client = DecartClient(api_key=opts.decart_api_key)

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
                        on_remote_stream=on_remote_stream,
                        initial_state=ModelState(
                            prompt=Prompt(text=opts.prompt, enhance=True),
                        ),
                    ),
                ),
                timeout=CONNECT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Decart connection timed out after {CONNECT_TIMEOUT_S:.0f}s.\n"
                "Common causes:\n"
                "  • You just ended a previous session — wait 15s before "
                "retrying so Decart frees the WebRTC port.\n"
                "  • Firewall blocking UDP — Decart's realtime API needs "
                "UDP traversal for WebRTC ICE.\n"
                "  • Decart API key invalid or revoked — verify it in the "
                "⚙ Settings panel."
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
            print(f"\rstreaming · {seconds}s", end="", flush=True)  # in-place
            notify(f"Live · {seconds}s")

        realtime_client.on("connection_change", _on_connection_change)
        realtime_client.on("error", _on_error)
        realtime_client.on("generation_tick", _on_tick)

        return _DecartSession(client, realtime_client)
