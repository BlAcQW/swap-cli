"""Managed mode: prepaid credit client and the mode-dependent config gates.

The riskiest parts of this feature aren't the network calls — they're the small
decisions around them. A managed licence has no provider key by design, so any
check that assumes one locks the customer out; and BYOK must keep working
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import config, credits  # noqa: E402


def _cfg(**kwargs) -> config.Config:
    base = dict(
        license_key=None,
        decart_api_key=None,
        license_cached_at=None,
        license_cached_valid_until=None,
    )
    return config.Config(**{**base, **kwargs})


# --- the lockout guard ------------------------------------------------------

def test_managed_licence_is_complete_without_any_provider_key() -> None:
    """The bug this prevents: a managed customer has no API key — that's the
    whole point — so an `is_complete` that demands one turns every one of them
    away at the Live button before any credit check can run."""
    cfg = _cfg(license_key="SWAP-CLI-AAAA", license_mode="managed")
    assert cfg.is_managed is True
    assert cfg.has_provider_key is True
    assert cfg.is_complete is True


def test_byok_still_requires_a_key() -> None:
    """BYOK behaviour must be untouched."""
    cfg = _cfg(license_key="SWAP-CLI-AAAA")
    assert cfg.is_managed is False
    assert cfg.has_provider_key is False
    assert cfg.is_complete is False

    keyed = _cfg(license_key="SWAP-CLI-AAAA", fal_api_key="fal_sk_1")
    assert keyed.is_complete is True


def test_mode_parsing_is_forgiving() -> None:
    assert _cfg(license_mode="MANAGED ").is_managed is True
    assert _cfg(license_mode="").is_managed is False
    assert _cfg().is_managed is False  # default


# --- duration formatting ----------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [(240, "4m 00s"), (0, "0s"), (-5, "0s"), (59, "59s"), (60, "1m 00s"), (605, "10m 05s")],
)
def test_format_duration(seconds: int, expected: str) -> None:
    """Credit is only ever shown as time — customers think in call minutes."""
    assert credits.format_duration(seconds) == expected


# --- envelope + error decoding ----------------------------------------------

class _Resp:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _client(resp):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return resp

    return _Client


@pytest.fixture(autouse=True)
def _licence(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    config.update(license_key="SWAP-CLI-AAAA", license_mode="managed")


async def test_balance_unwraps_the_success_envelope(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _client(_Resp(200, {"success": True, "data": {
            "tokens": 240, "tokensPerSec": 1, "seconds": 240,
        }})),
    )
    bal = await credits.balance()
    assert bal.tokens == 240
    assert bal.label == "4m 00s"


async def test_server_error_code_is_preserved_for_translation(monkeypatch) -> None:
    """The server's machine code must survive so `license.explain` can render it
    in the same English as the rest of the app, instead of a second vocabulary."""
    import httpx

    from swap_cli import license

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _client(_Resp(402, {"success": False, "error": "insufficient_credit"})),
    )
    with pytest.raises(credits.CreditsError) as excinfo:
        await credits.balance()

    assert excinfo.value.code == "insufficient_credit"
    # Rendered, not raw.
    assert "insufficient_credit" not in str(excinfo.value)
    assert "redeem a voucher" in str(excinfo.value)
    assert license.explain("insufficient_credit") == str(excinfo.value)


async def test_non_json_reply_does_not_crash(monkeypatch) -> None:
    """A proxy or CDN can return HTML here; that must surface as a clean error."""
    import httpx

    monkeypatch.setattr(
        httpx, "AsyncClient", _client(_Resp(502, ValueError("not json")))
    )
    with pytest.raises(credits.CreditsError, match="Unexpected reply"):
        await credits.balance()


async def test_network_failure_is_reported_not_swallowed(monkeypatch) -> None:
    import httpx

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "AsyncClient", _Boom)
    with pytest.raises(credits.CreditsError, match="Couldn't reach the credit server"):
        await credits.balance()


async def test_session_close_never_raises(monkeypatch) -> None:
    """It runs during teardown, often while something else already went wrong —
    failing to close must not mask the original error."""
    import httpx

    monkeypatch.setattr(
        httpx, "AsyncClient", _client(_Resp(500, {"success": False, "error": "boom"}))
    )
    await credits.session_close("abc")  # must not raise


async def test_tick_sends_no_duration(monkeypatch) -> None:
    """The client must never report elapsed time: it runs on the customer's
    machine, so anything it claims is forgeable, and under-reporting would eat
    the whole margin. The server derives elapsed from its own clock."""
    import httpx

    sent: dict = {}

    class _Spy:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            sent.update(json or {})
            return _Resp(200, {"success": True, "data": {
                "balance": 100, "secondsBilled": 30, "shouldDisconnect": False,
            }})

    monkeypatch.setattr(httpx, "AsyncClient", _Spy)
    tick = await credits.session_tick("sess-1")

    assert "seconds" not in sent and "duration" not in sent and "elapsed" not in sent
    assert sent["sessionId"] == "sess-1"
    assert tick.should_disconnect is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
