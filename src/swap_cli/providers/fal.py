"""fal.ai provider — the same Decart Lucy realtime model, different account.

Why: Decart stamps the "✦ AI Generated" badge and refuses account-level removal.
fal resells the identical Lucy realtime model (`decart/lucy2-vton/realtime`) at
the same $0.02/sec pay-as-you-go, on fal's own Decart account — so its watermark
policy may differ. Same model = same quality.

Transport: fal's realtime is **WebRTC**, negotiated over a websocket. The
websocket is a *signaling channel only* — the transformed video flows over a
peer connection, exactly like Decart's own realtime API (fal just proxies it).
Confirmed against fal's client (`fal.realtime.connect` + `handleResult`, which
builds `new RTCPeerConnection({iceServers})` from an `iceservers` result) and the
Decart SDK's `decart/realtime/webrtc_connection.py` (a working aiortc build of the
same offer/answer/ice-candidate flow).

Wire sequence, all over the one websocket we open (token → `wss://fal.run/<app>?fal_jwt_token=`):

  1. POST https://rest.fal.ai/tokens/ (Authorization: Key <key>) → short-lived JWT.
  2. Open the websocket.
  3. Send the initial identity: {"prompt", "reference_image_url"} (fal's first `send`).
  4. Server → {"type":"iceservers","iceservers":[{urls,username,credential}]} →
     build the aiortc peer with those ICE servers, add the camera track.
  5. Client → {"type":"offer","sdp"}; server → {"type":"answer","sdp"};
     {"type":"ice-candidate","candidate":{candidate,sdpMid,sdpMLineIndex}} both ways.
  6. The remote WebRTC track carries the transformed video → handed to
     `on_remote_stream`, the SAME consumer the Decart provider feeds. So watermark
     removal, the OBS virtual camera, recording and the preview are unaware of
     which provider is live.

Client sends are msgpack dicts (fal's `connection.send`); server frames arrive as
msgpack OR JSON strings — both decoded.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
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

# WebRTC negotiation (offer → answer → ICE → connected) can involve TURN relays,
# so give it a little more room than Decart's direct path before giving up.
FAL_CONNECT_TIMEOUT_S = 60.0

REST_TOKENS_URL = "https://rest.fal.ai/tokens/"
_WS_BASE = "wss://fal.run"


def _data_uri(raw: bytes, content_type: str) -> str:
    """bytes → `data:<type>;base64,...`. Built here rather than via fal_client so
    the encode/decode helpers stay pure and testable without the package."""
    import base64

    return f"data:{content_type};base64,{base64.b64encode(raw).decode('ascii')}"


def _encode_reference(reference: str) -> str:
    """Reference identity → data URI, or pass a URL straight through."""
    p = Path(reference)
    if p.exists() and p.is_file():
        suffix = p.suffix.lower().lstrip(".") or "jpeg"
        content_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        return _data_uri(p.read_bytes(), content_type)
    return reference  # already a URL


# --- fal realtime websocket protocol (auth + URL; mirrors fal-ai/fal-js) --------


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


# --- WebRTC signaling helpers (pure, unit-testable) -----------------------------


def _ice_servers_from(msg: dict[str, Any]) -> list[Any]:
    """`{type:'iceservers', iceservers:[{urls,username,credential}]}` → aiortc
    RTCIceServer list. Tolerates the casing fal's client itself guards for
    (`iceservers` / `iceServers` / `ice_servers`)."""
    from aiortc import RTCIceServer

    raw = msg.get("iceservers") or msg.get("iceServers") or msg.get("ice_servers") or []
    servers: list[Any] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        urls = s.get("urls") or s.get("url")
        if not urls:
            continue
        servers.append(
            RTCIceServer(
                urls=urls,
                username=s.get("username"),
                credential=s.get("credential"),
            )
        )
    return servers


def _identity_message(prompt: str, reference: str | None) -> dict[str, Any]:
    """The initial (and update) identity payload fal expects — {prompt,
    reference_image_url}. `reference` is already a URL or data-URI."""
    msg: dict[str, Any] = {"prompt": prompt or ""}
    if reference:
        msg["reference_image_url"] = reference
    return msg


def _parse_ice_candidate(data: Any) -> Any:
    """`{candidate,sdpMid,sdpMLineIndex}` → aiortc RTCIceCandidate, or None."""
    if not isinstance(data, dict):
        return None
    cand_str = data.get("candidate") or ""
    if not cand_str:
        return None
    from aiortc.sdp import candidate_from_sdp

    # aiortc parses the SDP attribute value, i.e. without the leading "candidate:".
    sdp = cand_str[len("candidate:") :] if cand_str.startswith("candidate:") else cand_str
    candidate = candidate_from_sdp(sdp)
    candidate.sdpMid = data.get("sdpMid")
    candidate.sdpMLineIndex = data.get("sdpMLineIndex")
    return candidate


class FalRealtimeConnection:
    """aiortc WebRTC peer negotiated over fal's signaling websocket.

    Modeled on `decart/realtime/webrtc_connection.py` (the reference implementation
    of this exact protocol), but speaking fal's envelope — an `iceservers` message
    supplies the ICE config, and it does not block on Decart-specific acks.
    """

    def __init__(
        self,
        *,
        ws: Any,
        camera: Any,
        on_remote_stream: Callable[[Any], None],
        notify: Callable[[str], None],
        quit_event: Any,
    ) -> None:
        self._ws = ws
        self._camera = camera
        self._on_remote_stream = on_remote_stream
        self._notify = notify
        self._quit_event = quit_event

        self._pc: Any = None
        self._recv_task: asyncio.Task[Any] | None = None
        self._ice_queue: list[Any] = []
        self._done = asyncio.Event()  # negotiation resolved (connected OR errored)
        self._error: str | None = None
        self._connected_once = False
        self._logged_first = False

    # -- signaling I/O --------------------------------------------------------

    async def _signal_send(self, obj: dict[str, Any]) -> None:
        import msgpack

        await self._ws.send(msgpack.packb(obj, use_bin_type=True))

    def _decode_incoming(self, raw: Any) -> Any:
        import msgpack

        if isinstance(raw, (bytes, bytearray)):
            return msgpack.unpackb(raw, raw=False)
        if isinstance(raw, str):
            import json

            return json.loads(raw)
        return raw

    # -- lifecycle ------------------------------------------------------------

    async def connect(self, prompt: str, reference: str | None) -> None:
        """Kick off the handshake and return once the peer is connected.

        Sends the initial identity (which triggers fal to send `iceservers`),
        then drives the offer/answer exchange from the receive loop. Raises if
        negotiation fails; caller wraps this in `asyncio.wait_for` for a timeout.
        """
        await self._signal_send(_identity_message(prompt, reference))
        self._recv_task = asyncio.create_task(self._receive_loop())
        await self._done.wait()
        if self._error:
            raise RuntimeError(self._error)

    async def send_identity(self, prompt: str, reference: str | None) -> None:
        """Re-send {prompt, reference_image_url} on a live connection."""
        await self._signal_send(_identity_message(prompt, reference))

    async def _receive_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                await self._handle(self._decode_incoming(raw))
        except asyncio.CancelledError:
            raise
        except Exception as err:  # ws closed / decode error → end negotiation
            self._fail(f"fal signaling error: {err}")
        finally:
            # If the socket closed before we ever reached "connected", unblock
            # connect() with a clear reason instead of hanging until the timeout.
            if not self._done.is_set():
                self._fail("fal websocket closed during negotiation")

    async def _handle(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        if not self._logged_first:
            self._logged_first = True
            print(f"[runtime] fal signaling: first msg type={mtype!r}", flush=True)

        if mtype in ("iceservers", "iceServers", "ice_servers"):
            await self._on_iceservers(msg)
        elif mtype == "answer":
            await self._on_answer(msg)
        elif mtype in ("ice-candidate", "candidate", "iceCandidate"):
            await self._on_candidate(msg)
        elif mtype == "error":
            self._fail(f"fal error: {msg.get('error') or msg.get('message') or msg}")
        else:
            # Status/handshake frames (e.g. the initial {action,timing,type}) —
            # log the type so an unhandled step is nameable, then ignore.
            print(f"[runtime] fal signaling: ignoring type={mtype!r}", flush=True)

    async def _on_iceservers(self, msg: dict[str, Any]) -> None:
        from aiortc import RTCConfiguration, RTCPeerConnection

        servers = _ice_servers_from(msg)
        print(f"[runtime] fal signaling: iceservers ({len(servers)} servers)", flush=True)
        self._notify("fal: negotiating…")

        pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self._pc = pc

        @pc.on("track")
        def _on_track(track: Any) -> None:
            print(f"[runtime] fal: remote track ({track.kind})", flush=True)
            if track.kind == "video":
                self._on_remote_stream(track)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            st = pc.connectionState
            print(f"[runtime] fal connection: {st}", flush=True)
            self._notify(f"Connection: {st}")
            if st == "connected":
                self._connected_once = True
                self._done.set()
            elif st in ("failed", "closed"):
                if self._connected_once:
                    # A live drop after we were streaming → tear the session down.
                    with suppress(Exception):
                        self._quit_event.set()
                else:
                    self._fail(f"fal connection {st} before it came up")

        # sendrecv: our camera goes up, the transformed video returns on the same
        # m-line (mirrors the Decart SDK's addTrack-only setup).
        pc.addTrack(self._camera)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await self._signal_send({"type": "offer", "sdp": pc.localDescription.sdp})
        print("[runtime] fal signaling: offer sent", flush=True)

    async def _on_answer(self, msg: dict[str, Any]) -> None:
        from aiortc import RTCSessionDescription

        sdp = msg.get("sdp")
        if not sdp or self._pc is None:
            return
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
        print("[runtime] fal signaling: answer received", flush=True)
        # Flush any ICE candidates that arrived before the remote description.
        for cand in self._ice_queue:
            with suppress(Exception):
                await self._pc.addIceCandidate(cand)
        self._ice_queue.clear()

    async def _on_candidate(self, msg: dict[str, Any]) -> None:
        try:
            candidate = _parse_ice_candidate(msg.get("candidate"))
        except Exception as err:
            print(f"[runtime] fal: skipping bad ICE candidate ({err})", flush=True)
            return
        if candidate is None:
            return
        if self._pc is not None and self._pc.remoteDescription is not None:
            with suppress(Exception):
                await self._pc.addIceCandidate(candidate)
        else:
            self._ice_queue.append(candidate)  # queue until the answer lands

    def _fail(self, message: str) -> None:
        if self._error is None:
            self._error = message
        self._done.set()

    async def cleanup(self) -> None:
        """Tear down. Never raises — callers are already in a finally/suppress."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self._recv_task
        if self._pc is not None:
            with suppress(Exception, asyncio.CancelledError):
                await self._pc.close()
        with suppress(Exception, asyncio.CancelledError):
            await self._ws.close()


class _FalSession:
    def __init__(self, conn: FalRealtimeConnection) -> None:
        self._conn = conn

    async def set_identity(self, prompt: str, reference: str) -> None:
        # fal carries the identity in a {prompt, reference_image_url} message;
        # re-send it so a later identity change still takes effect.
        ref = _encode_reference(reference) if reference else None
        with suppress(Exception):
            await self._conn.send_identity(prompt, ref)

    async def disconnect(self) -> None:
        await self._conn.cleanup()


class FalProvider:
    """Runs the Lucy realtime model on fal over WebRTC (signaled on a websocket)."""

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
            import aiortc  # noqa: F401 — the WebRTC peer
            import msgpack  # noqa: F401 — needed by the signaling encode/decode
            import websockets
        except ImportError as err:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "The fal provider needs aiortc + websockets + msgpack. Run:  "
                "pip install aiortc websockets msgpack"
            ) from err

        notify("fal: fetching token…")
        token = await _fetch_realtime_token(key, _app_alias(self.app_id))
        notify("fal: connecting…")
        ws = await websockets.connect(_realtime_url(self.app_id, token), max_size=None)
        print(f"[runtime] fal realtime websocket open · {self.app_id}", flush=True)

        conn = FalRealtimeConnection(
            ws=ws,
            camera=camera,
            on_remote_stream=on_remote_stream,
            notify=notify,
            quit_event=quit_event,
        )
        reference = (
            _encode_reference(opts.reference) if getattr(opts, "reference", "") else None
        )
        try:
            await asyncio.wait_for(
                conn.connect(opts.prompt, reference),
                timeout=FAL_CONNECT_TIMEOUT_S,
            )
        except TimeoutError as exc:
            await conn.cleanup()
            raise RuntimeError(
                f"fal connection timed out after {FAL_CONNECT_TIMEOUT_S:.0f}s.\n"
                "Common causes:\n"
                "  • Firewall blocking UDP — fal's realtime API needs UDP "
                "traversal for WebRTC ICE.\n"
                "  • fal API key invalid or out of credits — verify it at "
                "https://fal.ai/dashboard.\n"
                "  • You just ended a previous session — wait ~15s and retry."
            ) from exc
        except Exception:
            await conn.cleanup()
            raise

        return _FalSession(conn)
