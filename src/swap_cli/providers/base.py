"""Provider abstraction: a realtime video-to-video backend.

swap-cli was hard-wired to Decart. Decart stamps an "✦ AI Generated" badge on
realtime output and removal is an account-level privilege we've been refused, so
we need to be able to run the SAME job on another backend (fal.ai resells the
same Decart Lucy model, on a separate account, at the same $0.02/sec).

The contract is deliberately small. Everything provider-specific (connecting,
applying the reference identity, teardown) lives behind `Provider`; everything
shared (camera, preview, watermark removal, virtual camera, recording, voice,
quit/teardown) stays in `runtime.run_session` and is untouched by the choice.

Key design point — `connect()` hands the caller a **remote track**: any object
exposing `async recv()` returning a frame with `.to_ndarray(format=...)`.
`Display` only ever needs that, so a provider that isn't WebRTC-based (fal
streams individual frames over a websocket) can supply an adapter that quacks
the same way, and the whole downstream pipeline keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FrameSpec:
    """Capture geometry the provider wants the camera opened at."""

    width: int
    height: int
    fps: int


class ProviderSession(Protocol):
    """A live connection to a provider. Owns all provider-specific teardown."""

    async def set_identity(self, prompt: str, reference: str) -> None:
        """Apply the reference identity image (path or URL) + prompt."""
        ...

    async def disconnect(self) -> None:
        """Tear down. Must never raise — callers wrap in suppress() anyway."""
        ...


class Provider(Protocol):
    """Factory for a realtime session."""

    name: str

    def frame_spec(self, opts: Any) -> FrameSpec:
        """Geometry to open the camera at (may lazy-import the provider SDK)."""
        ...

    async def connect(
        self,
        *,
        camera: Any,
        opts: Any,
        on_remote_stream: Callable[[Any], None],
        notify: Callable[[str], None],
        quit_event: Any,
    ) -> ProviderSession:
        """Open the session. Invokes `on_remote_stream(track)` once the
        transformed stream is available (the provider may call it from a
        callback, or synthesise the track itself)."""
        ...


def get_provider(name: str) -> Provider:
    """Resolve a provider by config name. Unknown names fail loudly — a typo
    must not silently fall back to a different backend (and a different bill)."""
    key = (name or "decart").strip().lower()
    if key == "decart":
        from .decart import DecartProvider

        return DecartProvider()
    if key == "fal":
        from .fal import FalProvider

        return FalProvider()
    raise ValueError(f"unknown provider {name!r} — expected 'decart' or 'fal'")
