"""fal.ai provider — the same Decart Lucy realtime model, different account.

Why: Decart stamps the "✦ AI Generated" badge and refuses account-level removal.
fal resells the identical Lucy realtime model (`decart/lucy2-vton/realtime`) at
the same $0.02/sec pay-as-you-go, on fal's own Decart account — so its watermark
policy may differ. Same model = same quality.

Transport reality (both documented realtime transports are dead ends in Python):
  - WMA WebRTC bridge → 404: Decart's Lucy is a marketplace model, not a WMA app.
  - plain msgpack websocket per-frame → "Error while forwarding the request".
  - the only smooth path is fal's UNDOCUMENTED WebRTC signaling over the JS SDK.

So this provider runs the **queue (`subscribe`) path**: one frame per request. It's
slow (seconds/frame) — a LAGGY preview, not smooth realtime — but it actually
produces fal's output, which is all we need right now to answer the only question
that matters: does fal remove the badge and swap the face? If yes, the smooth
WebRTC transport becomes worth reverse-engineering (a separate, bigger task).

`FalTrack` exposes the same `async recv()` contract `Display` expects, so
watermark removal, the OBS virtual camera, recording and the preview are all
unaware of which provider is live.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import FrameSpec

if TYPE_CHECKING:
    from collections.abc import Callable

# fal's only realtime Lucy endpoint (both fal.ai/lucy-2 and /lucy-2.1 point here).
FAL_REALTIME_APP = "decart/lucy2-vton/realtime"

# fal gives no model-geometry endpoint, so we pick the capture size ourselves.
# Matches what Decart negotiates for lucy-2 so switching providers doesn't change
# the user's framing, and keeps the bundled watermark template's scale sane.
DEFAULT_FRAME_SPEC = FrameSpec(width=1280, height=720, fps=20)

# JPEG quality for the per-frame upload. Every frame crosses the wire, so this is
# a direct latency/bandwidth knob: 80 keeps the badge/text legible for detection
# while staying far smaller than PNG.
_JPEG_QUALITY = 80


def _data_uri(raw: bytes, content_type: str) -> str:
    """bytes → `data:<type>;base64,...`. Built here rather than via fal_client so
    the encode/decode helpers stay pure and testable without the package."""
    import base64

    return f"data:{content_type};base64,{base64.b64encode(raw).decode('ascii')}"


def _encode_frame_jpeg(bgr: Any, quality: int = _JPEG_QUALITY) -> str:
    """BGR ndarray → `data:image/jpeg;base64,...` (what fal's image_url wants)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame for fal")
    return _data_uri(buf.tobytes(), "image/jpeg")


def _encode_reference(reference: str) -> str:
    """Reference identity → data URI, or pass a URL straight through."""
    p = Path(reference)
    if p.exists() and p.is_file():
        suffix = p.suffix.lower().lstrip(".") or "jpeg"
        content_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        return _data_uri(p.read_bytes(), content_type)
    return reference  # already a URL


def _decode_result(result: dict[str, Any]) -> Any:
    """Pull the transformed frame out of fal's reply → BGR ndarray.

    fal returns the image under a few shapes depending on the model/version, so
    probe the known ones rather than assuming a single key.
    """
    import base64

    import cv2
    import numpy as np

    # fal signals a rejected request with an `error` field instead of an image —
    # surface its actual text so the cause is diagnosable, not just "no image".
    if "error" in result and not any(
        k in result for k in ("image", "images", "output", "frame")
    ):
        detail = result.get("error")
        for extra in ("detail", "message", "reason"):
            if extra in result:
                detail = f"{detail} · {extra}={result[extra]!r}"
        raise RuntimeError(f"fal rejected the request: {detail!r}")

    payload: Any = None
    for key in ("image", "images", "image_url", "output", "frame"):
        if key in result:
            payload = result[key]
            break
    if payload is None:
        raise RuntimeError(f"no image in fal response (keys: {sorted(result)})")
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        payload = payload.get("url") or payload.get("content") or payload.get("data")
    if isinstance(payload, str):
        if payload.startswith("data:"):
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload)
    elif isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raise RuntimeError(f"unsupported fal image payload: {type(payload)!r}")

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode the frame fal returned")
    return frame


class FalTrack:
    """Track-like adapter over fal's QUEUE path: pull a camera frame, round-trip
    it through `fal_client.subscribe_async`, return the transformed frame — so
    `Display` treats fal exactly like a Decart remote track.

    Serial by design (one request in flight): the model is stateful per frame and
    the queue round-trip (seconds) IS the laggy-preview latency. This is a
    verification transport, not the smooth-realtime one.
    """

    kind = "video"

    def __init__(self, camera: Any, app_id: str, state: dict[str, Any]) -> None:
        self._camera = camera
        self._app_id = app_id
        self._state = state  # {"prompt": str, "reference_image_url": str | None}
        self._closed = False

    async def recv(self) -> Any:
        if self._closed:
            raise RuntimeError("fal track closed")
        import av
        import fal_client

        frame = await self._camera.recv()
        bgr = frame.to_ndarray(format="bgr24")

        args: dict[str, Any] = {
            "prompt": self._state.get("prompt", ""),
            "image_url": _encode_frame_jpeg(bgr),
        }
        ref = self._state.get("reference_image_url")
        if ref:
            args["reference_image_url"] = ref

        result = await fal_client.subscribe_async(self._app_id, arguments=args)
        out = av.VideoFrame.from_ndarray(_decode_result(result), format="bgr24")
        # Preserve timing so downstream recording/vcam pace correctly. Guard the
        # copies: av rejects a None time_base, and not every source sets one.
        if frame.pts is not None:
            out.pts = frame.pts
        if frame.time_base is not None:
            out.time_base = frame.time_base
        return out

    def stop(self) -> None:
        self._closed = True


class _FalSession:
    def __init__(self, track: FalTrack, state: dict[str, Any]) -> None:
        self._track = track
        self._state = state

    async def set_identity(self, prompt: str, reference: str) -> None:
        # No live connection to update — subscribe re-sends prompt+reference with
        # every frame, so just update the state the track reads.
        if prompt:
            self._state["prompt"] = prompt
        if reference:
            self._state["reference_image_url"] = _encode_reference(reference)

    async def disconnect(self) -> None:
        # Teardown must never raise — the caller is already in a finally block.
        self._track.stop()


class FalProvider:
    """Runs the Lucy model on fal via the queue path (laggy verification mode)."""

    name = "fal"
    app_id = FAL_REALTIME_APP

    def frame_spec(self, opts: Any) -> FrameSpec:
        return DEFAULT_FRAME_SPEC

    async def connect(
        self,
        *,
        camera: Any,
        opts: Any,
        on_remote_stream: Callable[[Any], None],
        notify: Callable[[str], None],
        quit_event: Any,
    ) -> _FalSession:
        # Validate config BEFORE touching dependencies, so a missing key reports
        # the missing key — not an import error that sends the user down the
        # wrong path.
        key = (getattr(opts, "fal_api_key", "") or "").strip()
        if not key:
            raise RuntimeError(
                "No fal API key set. Get one at https://fal.ai/dashboard/keys, then "
                "run `swap setup` (or set it in ⚙ Settings)."
            )

        try:
            import fal_client  # noqa: F401 — ensure it's installed before we start
        except ImportError as err:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "The fal provider needs the fal client. Run:  pip install fal-client"
            ) from err
        import os

        os.environ["FAL_KEY"] = key  # fal_client reads the key from the env

        print(f"[runtime] fal: queue mode (laggy preview) · {self.app_id}", flush=True)
        notify("fal: queue mode — expect a slow preview (a frame every few seconds).")

        state: dict[str, Any] = {
            "prompt": opts.prompt,
            "reference_image_url": _encode_reference(opts.reference)
            if opts.reference
            else None,
        }
        track = FalTrack(camera, self.app_id, state)
        # No remote track to await — hand our adapter straight to the same
        # consumer the Decart path uses.
        on_remote_stream(track)
        return _FalSession(track, state)
