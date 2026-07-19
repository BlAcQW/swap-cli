"""fal.ai provider — the same Decart Lucy realtime model, different account.

Why: Decart stamps the "✦ AI Generated" badge and refuses account-level removal.
fal resells the identical Lucy realtime model (`decart/lucy2-vton/realtime`) at
the same $0.02/sec pay-as-you-go, on fal's own Decart account — so its watermark
policy may differ. Same model = same quality; only the transport changes.

Transport: fal's realtime VIDEO path is **WebRTC** (its "World Model Accelerator"
bridge), NOT per-frame msgpack — sending frames over the plain websocket returns
"Error while forwarding the request". So this is a real WebRTC client:

  1. aiortc RTCPeerConnection, add the webcam track, open a data channel.
  2. Create the offer; aiortc gathers all ICE candidates into the SDP (the bridge
     does NOT support trickle ICE).
  3. POST the offer + app_id to https://wma.fal.run/session → get the SDP answer.
  4. The transformed video comes back as a WebRTC REMOTE TRACK — handed to the
     exact same `on_remote_stream` callback Decart uses, so Display (watermark
     removal, OBS virtual camera, recording, preview) is unchanged.
  5. Prompt + reference image go over the data channel; heartbeat keeps it alive.

Two things are unconfirmed until this runs live (documented in the plan): whether
the marketplace model is reachable via the WMA bridge from a non-browser client,
and the exact data-channel message schema. Both fail loud.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import FrameSpec

if TYPE_CHECKING:
    from collections.abc import Callable

# fal's only realtime Lucy endpoint (both fal.ai/lucy-2 and /lucy-2.1 point here).
FAL_REALTIME_APP = "decart/lucy2-vton/realtime"

# fal's WebRTC signaling bridge (World Model Accelerator). POST an SDP offer +
# app_id, get an SDP answer; heartbeat keeps the session alive.
WMA_BRIDGE_URL = "https://wma.fal.run"
HEARTBEAT_INTERVAL_S = 5.0
# Public STUN so the offer carries reachable candidates. Strict-NAT users may
# need TURN; the bridge answer can also advertise ICE servers we could adopt.
_STUN_URL = "stun:stun.l.google.com:19302"

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


def _bridge_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Key {key}", "Content-Type": "application/json"}


def _data_channel_payload(prompt: str, reference: str | None) -> str:
    """The message we send over the data channel to steer the model. Frames now
    flow over the media track, so this carries only prompt + reference identity.
    Schema is fal's undocumented VTON input minus image_url; iterate if needed."""
    msg: dict[str, Any] = {"prompt": prompt or ""}
    if reference:
        msg["reference_image_url"] = _encode_reference(reference)
    return json.dumps(msg)


async def _post_offer(app_id: str, offer_sdp: str, key: str) -> dict[str, Any]:
    """POST the SDP offer to the WMA bridge, return the answer JSON. Non-2xx is
    the answer to 'is this marketplace model reachable from a non-browser client'
    — surface it loudly rather than as a generic connection failure."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{WMA_BRIDGE_URL}/session",
            headers=_bridge_headers(key),
            json={"app_id": app_id, "sdp": offer_sdp, "type": "offer"},
        )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"fal WMA bridge rejected app_id {app_id!r} (HTTP {resp.status_code}): "
            f"{resp.text[:300]!r} — this marketplace model may not be reachable "
            "from a non-browser client."
        )
    return resp.json()


async def _heartbeat(session_id: str, key: str, stop: asyncio.Event) -> None:
    """Keep the WMA session alive until stop is set (every HEARTBEAT_INTERVAL_S)."""
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        while not stop.is_set():
            with suppress(Exception):
                await client.post(
                    f"{WMA_BRIDGE_URL}/session/heartbeat",
                    headers=_bridge_headers(key),
                    json={"session_id": session_id},
                )
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL_S)


class _FalSession:
    def __init__(self, pc: Any, channel: Any, hb: Any, stop: asyncio.Event) -> None:
        self._pc = pc
        self._channel = channel
        self._hb = hb
        self._stop = stop

    async def set_identity(self, prompt: str, reference: str) -> None:
        with suppress(Exception):
            if getattr(self._channel, "readyState", None) == "open":
                self._channel.send(_data_channel_payload(prompt, reference))

    async def disconnect(self) -> None:
        # Teardown must never raise — the caller is already in a finally block.
        self._stop.set()
        if self._hb is not None:
            self._hb.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self._hb
        with suppress(Exception, asyncio.CancelledError):
            await self._pc.close()


class FalProvider:
    """Runs the Lucy realtime model on fal instead of Decart, over WebRTC."""

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

        from aiortc import (  # type: ignore[import-not-found]
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )

        notify("Negotiating fal session…")
        pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(urls=_STUN_URL)])
        )
        pc.addTrack(camera)  # webcam → fal
        channel = pc.createDataChannel("input")

        @channel.on("open")
        def _on_open() -> None:  # send prompt + reference once the channel is up
            with suppress(Exception):
                channel.send(_data_channel_payload(opts.prompt, opts.reference))
                print("[runtime] fal: sent prompt/reference on data channel", flush=True)

        @pc.on("track")
        def _on_track(track: Any) -> None:
            # fal's transformed video → hand to the SAME consumer Decart uses, so
            # watermark removal / OBS vcam / recording / preview are untouched.
            if track.kind == "video":
                print("[runtime] fal: remote video track received", flush=True)
                on_remote_stream(track)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            st = pc.connectionState
            print(f"[runtime] fal connection: {st}", flush=True)
            notify(f"Connection: {st}")
            if st in ("failed", "closed", "disconnected"):
                quit_event.set()

        # Offer + FULL ICE gathering (aiortc completes gathering in
        # setLocalDescription, so localDescription.sdp carries all candidates —
        # the WMA bridge does not support trickle ICE).
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        answer = await _post_offer(self.app_id, pc.localDescription.sdp, key)
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer.get("type", "answer"))
        )
        print(f"[runtime] fal WMA connected · {self.app_id}", flush=True)

        stop = asyncio.Event()
        session_id = answer.get("session_id")
        hb = (
            asyncio.ensure_future(_heartbeat(session_id, key, stop))
            if session_id
            else None
        )
        return _FalSession(pc, channel, hb, stop)
