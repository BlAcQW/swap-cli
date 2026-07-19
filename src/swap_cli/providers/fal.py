"""fal.ai provider — the same Decart Lucy realtime model, different account.

Why: Decart stamps the "✦ AI Generated" badge and refuses account-level removal.
fal resells the identical Lucy realtime model (`decart/lucy2-vton/realtime`) at
the same $0.02/sec pay-as-you-go, on fal's own Decart account — so its watermark
policy may differ. Same model = same quality.

Transport: fal's realtime is a plain **msgpack websocket** (there is NO WebRTC —
confirmed by reading fal's own SDK, `fal-ai/fal-js`). The wire protocol, extracted
from `realtime.ts` + the `camera-turbo` webcam demo:

  1. POST https://rest.fal.ai/tokens/ (Authorization: Key <fal-key>) with
     {"allowed_apps": ["<alias>"], "token_expiration": 120} → a JWT.
  2. Open  wss://fal.run/<app>/realtime?fal_jwt_token=<JWT>
  3. Send each frame as msgpack {"prompt", "image_bytes": <raw JPEG bytes>,
     "reference_image_url": <ref>} — RAW bytes, not a base64 data-URI (that's the
     queue API's shape; sending it here is likely why earlier attempts got
     "Error while forwarding the request").
  4. Read msgpack results; the frame comes back as raw bytes under images[0].content.

`FalTrack` wraps the persistent websocket and exposes the same `async recv()`
contract `Display` expects, so watermark removal, the OBS virtual camera,
recording and the preview are all unaware of which provider is live.
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


# --- realtime websocket protocol (mirrors fal-ai/fal-js realtime.ts) ----------

REST_TOKENS_URL = "https://rest.fal.ai/tokens/"
_WS_BASE = "wss://fal.run"


def _app_alias(app: str) -> str:
    """`decart/lucy2-vton/realtime` → `lucy2-vton` (owner/alias/path…), matching
    fal-js parseEndpointId — the token's `allowed_apps` wants the alias only."""
    parts = app.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def _realtime_url(app: str, jwt: str) -> str:
    """wss URL for the app. The id already ends in `/realtime`, so (per fal-js
    resolveEndpointPath) we do NOT append another path segment."""
    from urllib.parse import quote

    return f"{_WS_BASE}/{app}?fal_jwt_token={quote(jwt, safe='')}"


async def _fetch_realtime_token(key: str, alias: str) -> str:
    """POST /tokens/ for a short-lived JWT (fal-js getTemporaryAuthToken)."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            REST_TOKENS_URL,
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json={"allowed_apps": [alias], "token_expiration": 120},
        )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"fal token request failed (HTTP {resp.status_code}): {resp.text[:200]!r} — "
            "check the fal API key."
        )
    token: Any = resp.json()
    if isinstance(token, dict):  # some proxies wrap it as {"detail": <jwt>}
        token = token.get("detail", token)
    if not isinstance(token, str):
        raise RuntimeError(f"unexpected token response shape: {type(token)!r}")
    return token


def _jpeg_bytes(bgr: Any, quality: int = _JPEG_QUALITY) -> bytes:
    """BGR ndarray → raw JPEG bytes (the realtime `image_bytes` field)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame for fal")
    return buf.tobytes()


def _encode_input(prompt: str, reference: str | None, jpeg: bytes) -> bytes:
    """msgpack-encode one realtime request (raw JPEG bytes, per the webcam demo)."""
    import msgpack

    msg: dict[str, Any] = {"prompt": prompt or "", "image_bytes": jpeg}
    if reference:
        msg["reference_image_url"] = reference
    return msgpack.packb(msg, use_bin_type=True)


def _is_control_message(msg: Any) -> bool:
    """True for fal's non-result frames we must skip (keepalive / timeout)."""
    if not isinstance(msg, dict):
        return False
    if msg.get("type") == "x-fal-message":
        return True
    return msg.get("type") == "x-fal-error" and msg.get("error") == "TIMEOUT"


class FalTrack:
    """Track-like adapter over fal's realtime websocket: pull a camera frame, send
    it as one msgpack request, read the transformed frame back — so `Display`
    treats fal exactly like a Decart remote track.

    Serial (send → recv per frame): the model is stateful per connection, so we
    keep one request in flight.
    """

    kind = "video"

    def __init__(self, camera: Any, ws: Any, state: dict[str, Any]) -> None:
        self._camera = camera
        self._ws = ws
        self._state = state  # {"prompt": str, "reference_image_url": str | None}
        self._closed = False
        self._logged_first = False

    async def recv(self) -> Any:
        if self._closed:
            raise RuntimeError("fal track closed")
        import av
        import msgpack

        frame = await self._camera.recv()
        bgr = frame.to_ndarray(format="bgr24")
        await self._ws.send(
            _encode_input(
                self._state.get("prompt", ""),
                self._state.get("reference_image_url"),
                _jpeg_bytes(bgr),
            )
        )

        # Read until an actual result (skip keepalive/timeout control frames).
        while True:
            raw = await self._ws.recv()
            msg = msgpack.unpackb(raw, raw=False) if isinstance(raw, (bytes, bytearray)) else raw
            if not self._logged_first:
                self._logged_first = True
                keys = sorted(msg) if isinstance(msg, dict) else type(msg).__name__
                print(f"[runtime] fal: first reply keys={keys}", flush=True)
            if _is_control_message(msg):
                continue
            break

        out = av.VideoFrame.from_ndarray(_decode_result(msg), format="bgr24")
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
    def __init__(self, ws: Any, track: FalTrack, state: dict[str, Any]) -> None:
        self._ws = ws
        self._track = track
        self._state = state

    async def set_identity(self, prompt: str, reference: str) -> None:
        # No separate "set" — prompt+reference ride with every frame; just update
        # the state the track sends.
        if prompt:
            self._state["prompt"] = prompt
        if reference:
            self._state["reference_image_url"] = _encode_reference(reference)

    async def disconnect(self) -> None:
        # Teardown must never raise — the caller is already in a finally block.
        self._track.stop()
        from contextlib import suppress

        with suppress(Exception):
            await self._ws.close()


class FalProvider:
    """Runs the Lucy realtime model on fal over a hand-rolled msgpack websocket."""

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
            import msgpack  # noqa: F401 — needed by the track's encode/decode
            import websockets
        except ImportError as err:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "The fal provider needs websockets + msgpack. Run:  "
                "pip install websockets msgpack"
            ) from err

        notify("fal: fetching token…")
        token = await _fetch_realtime_token(key, _app_alias(self.app_id))
        notify("fal: connecting…")
        ws = await websockets.connect(_realtime_url(self.app_id, token), max_size=None)
        print(f"[runtime] fal realtime websocket open · {self.app_id}", flush=True)

        state: dict[str, Any] = {
            "prompt": opts.prompt,
            "reference_image_url": _encode_reference(opts.reference)
            if opts.reference
            else None,
        }
        track = FalTrack(camera, ws, state)
        # No remote track to await — hand our adapter straight to the same
        # consumer the Decart path uses.
        on_remote_stream(track)
        return _FalSession(ws, track, state)
