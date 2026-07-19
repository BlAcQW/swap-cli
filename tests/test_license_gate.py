"""A session must not start without a valid license.

`swap gui` used to do only a "best-effort, non-blocking" license check that set
a status label — so any non-empty key string could stream indefinitely, while
`swap run` (CLI) correctly refused. The gate now lives in run_session, the one
choke point every caller funnels through, so the hole can't reopen in a new
entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _install_stubs() -> None:
    """runtime pulls camera → aiortc/cv2/av at import; stub them so the gate can
    be tested without the native media stack (same approach as sibling tests)."""
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


_install_stubs()

from swap_cli import license, runtime  # noqa: E402


def _opts(**kwargs):
    base = dict(decart_api_key="dct_x", reference=None, prompt="p")
    return runtime.RunOptions(**{**base, **kwargs})


async def test_invalid_license_blocks_the_session() -> None:
    """A refused key must stop the run with the ENGLISH reason, not a code."""
    from swap_cli.license import LicenseStatus

    async def _refused(**_kw):
        return LicenseStatus(valid=False, reason="http_403")

    orig = license.validate
    license.validate = _refused
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await runtime._require_valid_license(lambda _s: None)
    finally:
        license.validate = orig

    msg = str(excinfo.value)
    assert "http_403" not in msg  # code never shown to the user
    assert "refused the key" in msg
    assert "swap.storelygh.com" in msg


async def test_unverifiable_license_blocks_the_session() -> None:
    """Never validated + server unreachable → refuse (matches `swap run`).
    Failing open here would let anyone block the domain and run for free."""

    async def _unreachable(**_kw):
        raise license.LicenseError("License server unreachable: timeout")

    orig = license.validate
    license.validate = _unreachable
    try:
        with pytest.raises(RuntimeError, match="Couldn't verify your license"):
            await runtime._require_valid_license(lambda _s: None)
    finally:
        license.validate = orig


async def test_valid_license_passes_the_gate() -> None:
    from swap_cli.license import LicenseStatus

    async def _ok(**_kw):
        return LicenseStatus(valid=True, reason="ok")

    orig = license.validate
    license.validate = _ok
    try:
        await runtime._require_valid_license(lambda _s: None)  # must not raise
    finally:
        license.validate = orig


async def test_cached_license_passes_offline() -> None:
    """A paying user on a flaky network keeps working inside the 24h grace."""
    from swap_cli.license import LicenseStatus

    async def _cached(**_kw):
        return LicenseStatus(valid=True, reason="offline_grace", cached=True)

    orig = license.validate
    license.validate = _cached
    try:
        await runtime._require_valid_license(lambda _s: None)
    finally:
        license.validate = orig


async def test_gate_forces_an_online_check_so_revocation_bites() -> None:
    """Revocation must take effect at the next session, not up to 24h later.

    The plain cached path honours a stale "valid" for OFFLINE_GRACE_SECONDS, so
    a revoked user could keep streaming for a day. The gate must ask the server.
    """
    seen: dict = {}

    async def _spy(**kwargs):
        from swap_cli.license import LicenseStatus

        seen.update(kwargs)
        return LicenseStatus(valid=True, reason="ok")

    orig = license.validate
    license.validate = _spy
    try:
        await runtime._require_valid_license(lambda _s: None)
    finally:
        license.validate = orig

    assert seen.get("force_online") is True


async def test_revoked_user_is_blocked_even_with_a_fresh_cache(monkeypatch) -> None:
    """End-to-end through the real validate(): a cached-valid licence that the
    server now rejects must NOT be allowed through."""
    import time

    import httpx

    from swap_cli import config

    now = int(time.time())
    monkeypatch.setattr(
        config,
        "load",
        lambda: config.Config(
            license_key="SWAP-CLI-REVOKED",
            decart_api_key="dct_x",
            license_cached_at=now - 60,  # validated a minute ago
            license_cached_valid_until=now + 30 * 24 * 3600,  # cache says valid
        ),
    )

    class _Resp:
        status_code = 403
        text = "forbidden"

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

    with pytest.raises(RuntimeError, match="refused the key"):
        await runtime._require_valid_license(lambda _s: None)


async def test_gate_runs_before_the_camera_is_opened(monkeypatch) -> None:
    """The refusal must happen before we grab the webcam or contact a provider —
    an unlicensed run should never hold the camera or bill a backend."""
    from swap_cli.license import LicenseStatus

    opened: list = []

    def _camera(*a, **k):
        opened.append((a, k))
        raise AssertionError("camera opened despite an invalid license")

    async def _refused(**_kw):
        return LicenseStatus(valid=False, reason="expired")

    monkeypatch.setattr(runtime, "CameraTrack", _camera)
    monkeypatch.setattr(license, "validate", _refused)

    with pytest.raises(RuntimeError, match="License check failed"):
        await runtime.run_session(_opts())
    assert opened == []  # never touched the camera


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
