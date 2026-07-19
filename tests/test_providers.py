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


# --- fal WebRTC (WMA bridge) signaling seams --------------------------------

def test_bridge_headers_carry_the_key() -> None:
    from swap_cli.providers.fal import _bridge_headers

    h = _bridge_headers("fal_sk_abc")
    assert h["Authorization"] == "Key fal_sk_abc"
    assert h["Content-Type"] == "application/json"


def test_data_channel_payload_is_prompt_plus_reference() -> None:
    """Frames flow over the media track now, so the data channel carries only the
    prompt and (encoded) reference identity — NOT per-frame image_url."""
    import json

    from swap_cli.providers.fal import _data_channel_payload

    msg = json.loads(_data_channel_payload("swap me", None))
    assert msg == {"prompt": "swap me"}  # no reference → no key, no image_url
    assert "image_url" not in msg


def test_data_channel_payload_encodes_reference_url() -> None:
    import json

    from swap_cli.providers.fal import _data_channel_payload

    msg = json.loads(_data_channel_payload("p", "https://example.com/face.png"))
    assert msg["reference_image_url"] == "https://example.com/face.png"


async def test_post_offer_builds_the_request(monkeypatch) -> None:
    """The offer POST must hit the bridge with app_id + type:offer + auth."""
    import httpx

    from swap_cli.providers import fal as falmod

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"session_id": "s1", "sdp": "ANSWER", "type": "answer"}

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
    answer = await falmod._post_offer("decart/lucy2-vton/realtime", "OFFERSDP", "fal_sk_k")

    assert captured["url"].endswith("/session")
    assert captured["json"] == {
        "app_id": "decart/lucy2-vton/realtime",
        "sdp": "OFFERSDP",
        "type": "offer",
    }
    assert captured["headers"]["Authorization"] == "Key fal_sk_k"
    assert answer["session_id"] == "s1" and answer["sdp"] == "ANSWER"


async def test_post_offer_raises_actionably_on_non_2xx(monkeypatch) -> None:
    """A bridge rejection answers 'is this model reachable from Python' — the
    error must say so, with the app_id and status, not a generic failure."""
    import httpx

    from swap_cli.providers import fal as falmod

    class _Resp:
        status_code = 404
        text = "not found"

        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError, match=r"WMA bridge rejected.*404.*non-browser"):
        await falmod._post_offer("decart/lucy2-vton/realtime", "SDP", "k")


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
