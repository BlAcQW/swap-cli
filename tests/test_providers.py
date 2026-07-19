"""Tests for the realtime provider abstraction (Sprint 18).

swap-cli can run the swap on Decart (default, WebRTC) or fal.ai (the same Lucy
model, over a websocket). These cover selection + the fal frame adapter, which is
the piece that lets `Display` stay provider-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Same stub-eviction dance as test_watermark.py. test_virtual_camera inserts bare
# ModuleType stubs for cv2/av so it can run in a bare CI sandbox; we need the REAL
# ones to exercise JPEG encode/decode and frame construction in the fal adapter.
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


# --- fal frame adapter ------------------------------------------------------

def test_fal_encode_frame_is_jpeg_data_uri() -> None:
    from swap_cli.providers.fal import _encode_frame_jpeg

    frame = np.zeros((32, 64, 3), np.uint8)
    frame[:, :32] = 200
    uri = _encode_frame_jpeg(frame)
    assert uri.startswith("data:image/jpeg;base64,")


def test_fal_decode_result_roundtrips_a_frame() -> None:
    """fal replies with a base64 image; we must get a BGR ndarray back."""
    from swap_cli.providers.fal import _decode_result, _encode_frame_jpeg

    original = np.full((24, 48, 3), 90, np.uint8)
    uri = _encode_frame_jpeg(original)
    out = _decode_result({"image": uri})
    assert out.shape == original.shape
    assert out.dtype == np.uint8


def test_fal_decode_result_handles_shapes() -> None:
    """fal's payload shape varies by model/version — probe the known ones."""
    from swap_cli.providers.fal import _decode_result, _encode_frame_jpeg

    uri = _encode_frame_jpeg(np.zeros((16, 32, 3), np.uint8))
    for payload in (
        {"image": uri},
        {"images": [uri]},
        {"image": {"url": uri}},
        {"output": uri},
    ):
        assert _decode_result(payload).shape == (16, 32, 3)


def test_fal_decode_result_reports_missing_image() -> None:
    from swap_cli.providers.fal import _decode_result

    with pytest.raises(RuntimeError, match="no image in fal response"):
        _decode_result({"unexpected": 1})


def test_fal_decode_result_surfaces_error_text() -> None:
    """fal signals a rejected request with an `error` field — the actual text
    must reach the user, not a generic 'no image' message."""
    from swap_cli.providers.fal import _decode_result

    with pytest.raises(RuntimeError, match=r"fal rejected the request.*bad reference"):
        _decode_result({"error": "bad reference", "detail": "unsupported format"})


# --- fal realtime websocket protocol ----------------------------------------

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


def test_encode_input_is_msgpack_with_raw_bytes() -> None:
    """The realtime frame goes as raw JPEG BYTES under image_bytes (not a data
    URI) — that's the shape fal's own webcam demo uses."""
    msgpack = pytest.importorskip("msgpack")
    from swap_cli.providers.fal import _encode_input

    packed = _encode_input("swap me", "https://x/ref.png", b"\xff\xd8jpeg")
    msg = msgpack.unpackb(packed, raw=False)
    assert msg["prompt"] == "swap me"
    assert msg["image_bytes"] == b"\xff\xd8jpeg"  # raw bytes, not str
    assert msg["reference_image_url"] == "https://x/ref.png"


def test_is_control_message() -> None:
    from swap_cli.providers.fal import _is_control_message

    assert _is_control_message({"type": "x-fal-message"}) is True
    assert _is_control_message({"type": "x-fal-error", "error": "TIMEOUT"}) is True
    assert _is_control_message({"images": [{"content": b"x"}]}) is False
    assert _is_control_message({"type": "x-fal-error", "error": "BOOM"}) is False


async def test_fal_track_roundtrips_over_websocket() -> None:
    """FalTrack: camera frame → msgpack send → read result → transformed frame.
    Also skips a leading control (keepalive) frame."""
    from fractions import Fraction

    from swap_cli.providers import fal as falmod

    msgpack = pytest.importorskip("msgpack")
    swapped = np.full((16, 32, 3), 180, np.uint8)
    ok, buf = cv2.imencode(".jpg", swapped)
    assert ok

    class _FakeCamera:
        async def recv(self):
            f = av.VideoFrame.from_ndarray(np.zeros((16, 32, 3), np.uint8), format="bgr24")
            f.pts, f.time_base = 7, Fraction(1, 20)
            return f

    sent: list = []

    class _FakeWS:
        def __init__(self):
            self._replies = [
                msgpack.packb({"type": "x-fal-message"}, use_bin_type=True),  # skipped
                msgpack.packb({"images": [{"content": buf.tobytes()}]}, use_bin_type=True),
            ]

        async def send(self, data):
            sent.append(data)

        async def recv(self):
            return self._replies.pop(0)

    state = {"prompt": "swap me", "reference_image_url": "https://x/ref.png"}
    track = falmod.FalTrack(_FakeCamera(), _FakeWS(), state)
    out = await track.recv()

    # Sent a msgpack request with raw image_bytes + prompt + reference.
    req = msgpack.unpackb(sent[0], raw=False)
    assert req["prompt"] == "swap me"
    assert isinstance(req["image_bytes"], (bytes, bytearray))
    assert req["reference_image_url"] == "https://x/ref.png"
    # Got the transformed frame back (control frame skipped), timing preserved.
    assert out.to_ndarray(format="bgr24").shape == (16, 32, 3)
    assert out.pts == 7


async def test_fal_track_raises_when_closed() -> None:
    from swap_cli.providers.fal import FalTrack

    track = FalTrack(camera=None, ws=None, state={"prompt": ""})
    track.stop()
    with pytest.raises(RuntimeError, match="closed"):
        await track.recv()


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
