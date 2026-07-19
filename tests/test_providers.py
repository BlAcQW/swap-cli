"""Tests for the realtime provider abstraction (Sprint 18).

swap-cli can run the swap on Decart (default) or fal.ai (the same Lucy model on
fal's own account). BOTH are WebRTC — fal just negotiates the peer connection over
a signaling websocket. These cover provider selection + fal's signaling protocol
(token/URL, the iceservers→offer→answer handshake), which is the piece that lets
`Display` stay provider-agnostic (it consumes the remote aiortc track either way).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Same stub-eviction dance as test_watermark.py. test_virtual_camera inserts bare
# ModuleType stubs for cv2/av so it can run in a bare CI sandbox; we need the REAL
# ones for av.VideoFrame construction the fal path touches.
for _name, _marker in (("cv2", "imencode"), ("av", "VideoFrame")):
    _stub = sys.modules.get(_name)
    if _stub is not None and not hasattr(_stub, _marker):
        del sys.modules[_name]
        for _m in ("swap_cli.watermark", "swap_cli.display", "swap_cli.providers.fal"):
            sys.modules.pop(_m, None)

cv2 = pytest.importorskip("cv2")
av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from swap_cli.providers import get_provider  # noqa: E402


@pytest.fixture(autouse=True)
def _real_media_modules(monkeypatch):
    """`providers/fal.py` imports cv2/av/numpy lazily (inside functions), so they
    resolve at CALL time — and a sibling test module stubs cv2/av in sys.modules.
    Depending on import order that would poison these tests. Pin the real ones."""
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "numpy", np)


# --- selection --------------------------------------------------------------

def test_default_provider_is_decart() -> None:
    assert get_provider("decart").name == "decart"
    assert get_provider("").name == "decart"  # empty config → default


def test_fal_provider_resolves() -> None:
    assert get_provider("fal").name == "fal"


def test_provider_name_is_case_and_space_insensitive() -> None:
    assert get_provider("  FAL ").name == "fal"


def test_unknown_provider_fails_loudly() -> None:
    """A typo must NOT silently fall back to a different backend (and bill)."""
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("decrat")


# --- fal auth + URL ---------------------------------------------------------

def test_app_alias_and_realtime_url() -> None:
    from swap_cli.providers.fal import _app_alias, _realtime_url

    assert _app_alias("decart/lucy2-vton/realtime") == "lucy2-vton"
    url = _realtime_url("decart/lucy2-vton/realtime", "JWT.TOK EN")
    # app id kept verbatim (already ends in /realtime — no doubling); token in query
    assert url.startswith("wss://fal.run/decart/lucy2-vton/realtime?fal_jwt_token=")
    assert "JWT.TOK%20EN" in url  # url-encoded


async def test_fetch_realtime_token_builds_request(monkeypatch) -> None:
    """Token POST must hit /tokens/ with Key auth + the alias in allowed_apps,
    and unwrap the {"detail": jwt} proxy shape."""
    import httpx

    from swap_cli.providers import fal as falmod

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"detail": "the.jwt.token"}  # wrapped shape

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            captured.update(url=url, headers=headers, json=json)
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    tok = await falmod._fetch_realtime_token("fal_sk_k", "lucy2-vton")

    assert captured["url"] == falmod.REST_TOKENS_URL
    assert captured["headers"]["Authorization"] == "Key fal_sk_k"
    assert captured["json"] == {"allowed_apps": ["lucy2-vton"], "token_expiration": 120}
    assert tok == "the.jwt.token"


def test_encode_reference_passthrough_and_file(tmp_path) -> None:
    from swap_cli.providers.fal import _encode_reference

    assert _encode_reference("https://x/ref.png") == "https://x/ref.png"  # URL untouched
    p = tmp_path / "face.jpg"
    p.write_bytes(b"\xff\xd8jpeg")
    assert _encode_reference(str(p)).startswith("data:image/jpeg;base64,")


# --- fal WebRTC signaling ---------------------------------------------------

def test_identity_message_shape() -> None:
    """fal's first send (and identity updates) is {prompt, reference_image_url}."""
    from swap_cli.providers.fal import _identity_message

    assert _identity_message("hi", "https://x/r.png") == {
        "prompt": "hi",
        "reference_image_url": "https://x/r.png",
    }
    assert _identity_message("hi", None) == {"prompt": "hi"}  # no ref → key omitted
    assert _identity_message("", None) == {"prompt": ""}


def test_ice_servers_from_handles_casing(monkeypatch) -> None:
    """fal's own client guards for iceservers / iceServers / ice_servers — so do we.

    Uses the fake aiortc (real aiortc isn't a CI dep, and sibling tests stub it) so
    the mapping is verified deterministically regardless of the environment."""
    _fake_aiortc(monkeypatch, {})
    from swap_cli.providers.fal import _ice_servers_from

    for key in ("iceservers", "iceServers", "ice_servers"):
        entry = {"urls": "turn:r:3478", "username": "u", "credential": "c"}
        servers = _ice_servers_from({"type": "iceservers", key: [entry]})
        assert len(servers) == 1
        assert servers[0].urls == "turn:r:3478"
        assert servers[0].username == "u"
        assert servers[0].credential == "c"
    # A server entry with no urls is skipped, not crashed on.
    assert _ice_servers_from({"iceservers": [{"username": "x"}]}) == []


def test_parse_ice_candidate_roundtrips(monkeypatch) -> None:
    """Strips the leading `candidate:` and carries sdpMid/sdpMLineIndex through."""
    _fake_aiortc(monkeypatch, {})
    from swap_cli.providers.fal import _parse_ice_candidate

    cand = _parse_ice_candidate(
        {
            "candidate": "candidate:1 1 udp 2130706431 192.168.1.5 54321 typ host",
            "sdpMid": "0",
            "sdpMLineIndex": 0,
        }
    )
    assert cand is not None
    assert cand.sdpMid == "0"
    assert cand.sdpMLineIndex == 0
    assert cand.parsed_from == "1 1 udp 2130706431 192.168.1.5 54321 typ host"  # prefix stripped
    assert _parse_ice_candidate({"candidate": ""}) is None  # empty → skip
    assert _parse_ice_candidate("not a dict") is None


def _fake_aiortc(monkeypatch, events: dict):
    """Install a fake `aiortc` module so the signaling handshake runs without a
    real peer connection or any network. Mirrors just the surface fal.py uses."""
    import types

    class FakePC:
        def __init__(self, config=None):
            self.configuration = config
            self.remoteDescription = None
            self.connectionState = "new"
            self.localDescription = types.SimpleNamespace(sdp="OFFER_SDP")
            self._handlers: dict = {}

        def on(self, name):
            def deco(fn):
                self._handlers[name] = fn
                return fn

            return deco

        # camelCase mirrors aiortc's real RTCPeerConnection API (fal.py calls these).
        def addTrack(self, track):  # noqa: N802
            events["added_track"] = track

        async def createOffer(self):  # noqa: N802
            return types.SimpleNamespace(sdp="OFFER_SDP", type="offer")

        async def setLocalDescription(self, desc):  # noqa: N802
            events["local_sdp"] = desc.sdp

        async def setRemoteDescription(self, desc):  # noqa: N802
            self.remoteDescription = desc
            events["remote_sdp"] = desc.sdp
            # Simulate the peer coming up once the answer is applied: deliver the
            # remote track, then flip to "connected".
            th = self._handlers.get("track")
            if th:
                th(types.SimpleNamespace(kind="video"))
            self.connectionState = "connected"
            cs = self._handlers.get("connectionstatechange")
            if cs:
                await cs()

        async def addIceCandidate(self, c):  # noqa: N802
            events.setdefault("candidates", []).append(c)

        async def close(self):
            events["closed"] = True

    class FakeConfig:
        def __init__(self, iceServers=None):  # noqa: N803 — aiortc's kwarg name
            self.iceServers = iceServers

    class FakeIceServer:
        def __init__(self, urls=None, username=None, credential=None):
            self.urls, self.username, self.credential = urls, username, credential

    class FakeDesc:
        def __init__(self, sdp=None, type=None):
            self.sdp, self.type = sdp, type

    mod = types.ModuleType("aiortc")
    mod.RTCPeerConnection = FakePC
    mod.RTCConfiguration = FakeConfig
    mod.RTCIceServer = FakeIceServer
    mod.RTCSessionDescription = FakeDesc
    monkeypatch.setitem(sys.modules, "aiortc", mod)

    # `_parse_ice_candidate` does `from aiortc.sdp import candidate_from_sdp`.
    def _candidate_from_sdp(sdp):
        # Record what we were handed (post-`candidate:`-strip) so the caller can
        # assert the prefix handling; real aiortc parses this into a candidate.
        return types.SimpleNamespace(parsed_from=sdp, sdpMid=None, sdpMLineIndex=None)

    sdp_mod = types.ModuleType("aiortc.sdp")
    sdp_mod.candidate_from_sdp = _candidate_from_sdp
    mod.sdp = sdp_mod
    monkeypatch.setitem(sys.modules, "aiortc.sdp", sdp_mod)
    return FakePC


async def test_fal_signaling_handshake(monkeypatch) -> None:
    """The core protocol: send identity → iceservers builds the peer + offer →
    answer sets the remote description → the remote track reaches on_remote_stream.
    Also exercises the dual decoder (iceservers as msgpack, answer as JSON string)."""
    import asyncio
    import json

    msgpack = pytest.importorskip("msgpack")
    from swap_cli.providers import fal as falmod

    events: dict = {}
    _fake_aiortc(monkeypatch, events)

    class _FakeWS:
        def __init__(self):
            self.sent: list = []
            self._msgs = [
                msgpack.packb(  # binary frame
                    {
                        "type": "iceservers",
                        "iceservers": [
                            {"urls": "turn:relay:3478", "username": "u", "credential": "c"}
                        ],
                    },
                    use_bin_type=True,
                ),
                json.dumps({"type": "answer", "sdp": "ANSWER_SDP"}),  # string frame
            ]

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            if self._msgs:
                return self._msgs.pop(0)
            await asyncio.Event().wait()  # negotiation done — block, don't error

        async def close(self):
            pass

    ws = _FakeWS()
    got: dict = {}
    conn = falmod.FalRealtimeConnection(
        ws=ws,
        camera=object(),
        on_remote_stream=lambda t: got.__setitem__("track", t),
        notify=lambda _s: None,
        quit_event=asyncio.Event(),
    )

    await asyncio.wait_for(conn.connect("swap me", "https://x/ref.png"), timeout=2.0)

    # 1. Identity sent FIRST (triggers fal to send iceservers).
    first = msgpack.unpackb(ws.sent[0], raw=False)
    assert first == {"prompt": "swap me", "reference_image_url": "https://x/ref.png"}
    # 2. Peer built from the ICE servers fal supplied.
    assert conn._pc.configuration.iceServers[0].urls == "turn:relay:3478"
    # 3. Offer sent after iceservers.
    offer = msgpack.unpackb(ws.sent[1], raw=False)
    assert offer == {"type": "offer", "sdp": "OFFER_SDP"}
    # 4. Answer applied as the remote description.
    assert events["remote_sdp"] == "ANSWER_SDP"
    # 5. Remote (video) track handed to the shared consumer.
    assert got["track"].kind == "video"

    await conn.cleanup()


async def test_fal_signaling_surfaces_error(monkeypatch) -> None:
    """An `error` frame during negotiation must raise, not hang until timeout."""
    import asyncio

    msgpack = pytest.importorskip("msgpack")
    from swap_cli.providers import fal as falmod

    _fake_aiortc(monkeypatch, {})

    class _FakeWS:
        def __init__(self):
            self.sent: list = []
            self._msgs = [
                msgpack.packb({"type": "error", "error": "insufficient credits"}, use_bin_type=True)
            ]

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            if self._msgs:
                return self._msgs.pop(0)
            await asyncio.Event().wait()

        async def close(self):
            pass

    conn = falmod.FalRealtimeConnection(
        ws=_FakeWS(),
        camera=object(),
        on_remote_stream=lambda _t: None,
        notify=lambda _s: None,
        quit_event=asyncio.Event(),
    )
    with pytest.raises(RuntimeError, match="insufficient credits"):
        await asyncio.wait_for(conn.connect("x", None), timeout=2.0)
    await conn.cleanup()


async def test_fal_signaling_closed_socket_unblocks(monkeypatch) -> None:
    """If the socket closes before 'connected', connect() must fail fast."""
    import asyncio

    from swap_cli.providers import fal as falmod

    _fake_aiortc(monkeypatch, {})

    class _ClosedWS:
        async def send(self, data):
            pass

        async def recv(self):
            raise ConnectionError("socket closed")

        async def close(self):
            pass

    conn = falmod.FalRealtimeConnection(
        ws=_ClosedWS(),
        camera=object(),
        on_remote_stream=lambda _t: None,
        notify=lambda _s: None,
        quit_event=asyncio.Event(),
    )
    with pytest.raises(RuntimeError, match=r"signaling error|closed"):
        await asyncio.wait_for(conn.connect("x", None), timeout=2.0)
    await conn.cleanup()


# --- config plumbing --------------------------------------------------------

async def test_fal_provider_requires_a_key() -> None:
    """Missing fal key must be an actionable message, not a traceback."""
    from swap_cli.providers.fal import FalProvider

    class _Opts:
        fal_api_key = ""
        prompt = "x"

    with pytest.raises(RuntimeError, match="No fal API key"):
        await FalProvider().connect(
            camera=None,
            opts=_Opts(),
            on_remote_stream=lambda _t: None,
            notify=lambda _s: None,
            quit_event=None,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
