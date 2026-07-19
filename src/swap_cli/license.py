"""License validation against the swap backend with 24h offline grace."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from . import config

# Backend base URL. Defaults to production; override with SWAP_BACKEND_URL so a
# future domain change needs no re-release (e.g. SWAP_BACKEND_URL=https://swap.example.com).
DEFAULT_BACKEND_URL = "https://swap.storelygh.com"
BACKEND_URL = os.environ.get("SWAP_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")
LICENSE_VALIDATE_URL = f"{BACKEND_URL}/api/cli/license/validate"
OFFLINE_GRACE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class LicenseStatus:
    valid: bool
    reason: str
    plan: str | None = None
    expires_at: int | None = None
    cached: bool = False


class LicenseError(Exception):
    """Surfaced when license validation fails outright (network + no cache)."""


# Plain-English text for each machine reason code. These strings are shown to
# end users (doctor, the GUI status bar, `swap run`), so they say what happened
# AND what to do — a bare "http_403" told the user nothing.
_REASON_TEXT = {
    "ok": "valid",
    "cached": "valid (using cached result)",
    "offline_grace": "valid (cached — license server unreachable)",
    "no_license_key": "no license key saved — run `swap setup`",
    "expired": "expired — renew to keep using swap-cli",
    "revoked": "revoked — contact support",
    "not_found": "not recognised — check the key with `swap setup`",
    "machine_limit": "already in use on another machine",
}

_HTTP_TEXT = {
    401: "the license server didn't accept the key — check it with `swap setup`",
    403: (
        "the license server refused the key — it may be invalid, expired, or "
        "already in use on another machine. Check it with `swap setup`, or "
        "contact support if it should be active"
    ),
    404: "the license service wasn't found — update swap-cli (`pip install -U swap-cli`)",
    429: "too many checks in a row — wait a minute and try again",
}


def explain(reason: str) -> str:
    """Machine reason code → something a user can act on.

    Unknown codes fall through unchanged: the backend may send its own message
    text, which is already more useful than anything we'd substitute.
    """
    if reason in _REASON_TEXT:
        return _REASON_TEXT[reason]
    if reason.startswith("http_"):
        try:
            code = int(reason[len("http_") :])
        except ValueError:
            return reason
        if code in _HTTP_TEXT:
            return _HTTP_TEXT[code]
        if 500 <= code <= 599:
            return f"the license server had an error ({code}) — try again shortly"
        return f"the license server rejected the check ({code})"
    return reason


async def validate(*, force_online: bool = False) -> LicenseStatus:
    """Validate the user's license. Uses cached result within OFFLINE_GRACE_SECONDS."""
    cfg = config.load()
    if not cfg.license_key:
        return LicenseStatus(valid=False, reason="no_license_key")

    now = int(time.time())

    if (
        not force_online
        and cfg.license_cached_at is not None
        and cfg.license_cached_valid_until is not None
        and cfg.license_cached_valid_until > now
        and (now - cfg.license_cached_at) < OFFLINE_GRACE_SECONDS
    ):
        return LicenseStatus(
            valid=True,
            reason="cached",
            cached=True,
            expires_at=cfg.license_cached_valid_until,
        )

    payload = {
        "licenseKey": cfg.license_key,
        "machineId": config.machine_id(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(LICENSE_VALIDATE_URL, json=payload)
    except httpx.RequestError as err:
        # Network down — fall back to the last cached result if it's still
        # within grace AND the license itself hasn't expired. The expiry check
        # matters: without it, offline grace would extend a lapsed license by up
        # to OFFLINE_GRACE_SECONDS (the cached path above already checks it).
        if (
            cfg.license_cached_at is not None
            and cfg.license_cached_valid_until is not None
            and cfg.license_cached_valid_until > now
            and (now - cfg.license_cached_at) < OFFLINE_GRACE_SECONDS
        ):
            return LicenseStatus(
                valid=True,
                reason="offline_grace",
                cached=True,
                expires_at=cfg.license_cached_valid_until,
            )
        raise LicenseError(f"License server unreachable: {err}") from err

    if response.status_code != 200:
        return LicenseStatus(valid=False, reason=f"http_{response.status_code}")

    body = response.json()
    if not body.get("success"):
        return LicenseStatus(
            valid=False,
            reason=body.get("error", "unknown"),
        )

    data = body.get("data", {})
    expires_at = int(data.get("expiresAt", now + 7 * 24 * 3600))
    plan = data.get("plan")

    config.update(
        license_cached_at=now,
        license_cached_valid_until=expires_at,
    )

    return LicenseStatus(valid=True, reason="ok", plan=plan, expires_at=expires_at)
