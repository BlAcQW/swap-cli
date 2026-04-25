"""License validation against the swap backend with 24h offline grace."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from . import config

LICENSE_VALIDATE_URL = "https://swap.ikieguy.online/api/cli/license/validate"
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
        # Network down — fall back to last cached result if still within grace
        if (
            cfg.license_cached_at is not None
            and cfg.license_cached_valid_until is not None
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
