"""Regression guard for the Decart connect timeout (Sprint 14m).

Sprint 14m bumped CONNECT_TIMEOUT_S from 20 → 45 seconds because real-
world WebRTC cold reconnects (user clicks Stop then Live again) routinely
take 25–35 s. Going below 30 s in the future would re-introduce the
bug where back-to-back sessions appear to "time out" mid-handshake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _install_stubs() -> None:
    """Stub the heavy transitive imports the runtime module pulls in, so it can
    be imported in a test env without decart/aiortc/av installed."""
    import types
    from unittest.mock import MagicMock

    for name, attrs in (
        ("cv2", {"VideoWriter_fourcc": MagicMock(), "VideoWriter": MagicMock()}),
        ("av", {}),
        ("aiortc", {"VideoStreamTrack": type("X", (object,), {})}),
        ("decart", {"DecartClient": MagicMock(), "models": MagicMock()}),
        ("decart.realtime", {"RealtimeClient": MagicMock(), "RealtimeConnectOptions": MagicMock()}),
        ("decart.types", {"ModelState": MagicMock(), "Prompt": MagicMock()}),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m


def test_connect_timeout_above_reconnect_floor() -> None:
    """Must be ≥ 30 s so back-to-back sessions don't false-timeout."""
    _install_stubs()
    from swap_cli.runtime import CONNECT_TIMEOUT_S

    assert CONNECT_TIMEOUT_S >= 30.0, (
        f"CONNECT_TIMEOUT_S={CONNECT_TIMEOUT_S} — below 30 s reintroduces "
        "the Sprint 14m bug where back-to-back sessions timeout mid-handshake"
    )


def test_harmless_ice_noise_filter() -> None:
    """The loop-exception filter swallows ONLY the benign aioice STUN teardown
    noise (InvalidStateError from the Transaction.__retry timer), and lets every
    other loop exception through to the default handler."""
    import asyncio

    _install_stubs()
    from swap_cli.runtime import _is_harmless_ice_noise

    # Benign: the exact aioice STUN-retry race that printed during drops.
    assert _is_harmless_ice_noise(
        {
            "exception": asyncio.InvalidStateError("invalid state"),
            "message": "Exception in callback Transaction.__retry()",
        }
    )
    # Also matched by the callback name even without the exception object.
    assert _is_harmless_ice_noise(
        {"message": "Exception in callback Transaction.__retry()"}
    )
    # A real error must NOT be swallowed.
    assert not _is_harmless_ice_noise(
        {"exception": ValueError("boom"), "message": "Exception in callback foo"}
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
