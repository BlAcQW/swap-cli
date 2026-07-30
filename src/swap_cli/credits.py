"""Prepaid credit client — the managed-mode half of swap-cli.

In managed mode the customer never holds a provider API key. They buy prepaid
credit as voucher codes, and the backend mints a short-lived provider token per
session and meters the seconds against their balance.

Everything here talks to `/api/cli/*` on the licence backend, authenticating the
same way `license.validate` does — licence key plus the persisted device id.

Two deliberate properties:

* **We never report elapsed time.** `tick()` sends no duration; the server
  derives it from its own clock. A client running on the customer's machine
  can't be trusted to say how long it streamed, and at our margin an
  under-reporting client would consume all of it.
* **Errors keep the server's machine-readable code** so `license.explain()`
  renders them in the same English the rest of the app uses, rather than
  inventing a second error vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from . import config
from .license import BACKEND_URL, explain

# Short: these calls sit in front of the user pressing Live, and a slow backend
# should surface quickly rather than look like a hang.
TIMEOUT_S = 15.0


class CreditsError(RuntimeError):
    """A managed-mode request failed.

    Carries the server's machine-readable `code` alongside an English message so
    callers can branch on the cause (e.g. `insufficient_credit`) while still
    having something to show the user.
    """

    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or explain(code))
        self.code = code


@dataclass(frozen=True)
class Balance:
    tokens: int
    tokens_per_sec: float
    seconds: int

    @property
    def label(self) -> str:
        """Human-facing remaining time. Customers think in call minutes, never
        tokens, so the token count is never shown on its own."""
        return format_duration(self.seconds)


@dataclass(frozen=True)
class ManagedSession:
    session_id: str
    token: str
    app: str
    tokens_per_sec: float
    balance: int
    max_seconds: int
    heartbeat_interval: int


def format_duration(seconds: int) -> str:
    """Seconds → `4m 00s`. Used everywhere credit is displayed."""
    if seconds <= 0:
        return "0s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes <= 0:
        return f"{secs}s"
    return f"{minutes}m {secs:02d}s"


def _auth() -> dict[str, str]:
    cfg = config.load()
    if not cfg.license_key:
        raise CreditsError("no_license_key")
    return {"licenseKey": cfg.license_key, "machineId": config.machine_id()}


async def _post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST to a CLI endpoint and unwrap the `{success, data}` envelope."""
    body = {**_auth(), **(payload or {})}
    url = f"{BACKEND_URL}/api/cli/{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.post(url, json=body)
    except httpx.RequestError as err:
        raise CreditsError(
            "network", f"Couldn't reach the credit server: {err}"
        ) from err

    try:
        parsed = response.json()
    except Exception:  # HTML error page from a proxy, empty body, …
        parsed = None

    if not isinstance(parsed, dict):
        raise CreditsError("server_error", f"Unexpected reply (HTTP {response.status_code})")

    if response.status_code >= 300 or parsed.get("success") is False:
        code = parsed.get("error") or f"http_{response.status_code}"
        raise CreditsError(str(code))

    data = parsed.get("data")
    return data if isinstance(data, dict) else {}


async def balance() -> Balance:
    data = await _post("balance")
    return Balance(
        tokens=int(data.get("tokens", 0)),
        tokens_per_sec=float(data.get("tokensPerSec", 1) or 1),
        seconds=int(data.get("seconds", 0)),
    )


async def redeem(code: str) -> Balance:
    """Redeem a voucher. Raises CreditsError with the reason on failure.

    A brand-new device (fresh `swap setup`) isn't registered against the licence
    yet, so the read-only redeem auth rejects it with `machine_not_registered`.
    Instead of making the user run `swap doctor` first, we register the device
    with a validate on that specific error and retry once — so `setup → redeem`
    just works.
    """
    payload = {"code": code.strip().upper()}
    try:
        data = await _post("voucher/redeem", payload)
    except CreditsError as err:
        if err.code != "machine_not_registered":
            raise
        from . import license  # local import: license imports config, not credits

        try:
            status = await license.validate(force_online=True)
        except license.LicenseError as lic_err:
            raise CreditsError("server_error", str(lic_err)) from lic_err
        if not status.valid:
            raise CreditsError(status.reason) from err
        data = await _post("voucher/redeem", payload)
    return Balance(
        tokens=int(data.get("balance", 0)),
        tokens_per_sec=float(data.get("tokensPerSec", 1) or 1),
        seconds=int(data.get("seconds", 0)),
    )


async def session_open() -> ManagedSession:
    """Reserve a metered session and get the provider token to stream with.

    Nothing is charged here — billing starts on the first tick, which the client
    sends once frames are actually flowing.
    """
    data = await _post("session/open")
    return ManagedSession(
        session_id=str(data.get("sessionId", "")),
        token=str(data.get("token", "")),
        app=str(data.get("app", "")),
        tokens_per_sec=float(data.get("tokensPerSec", 1) or 1),
        balance=int(data.get("balance", 0)),
        max_seconds=int(data.get("maxSeconds", 0)),
        heartbeat_interval=int(data.get("heartbeatInterval", 10) or 10),
    )


@dataclass(frozen=True)
class Tick:
    balance: int
    seconds_billed: int
    should_disconnect: bool
    seconds_left: int


async def session_tick(session_id: str) -> Tick:
    """Heartbeat. Sends no duration — see the module docstring."""
    data = await _post("session/tick", {"sessionId": session_id})
    return Tick(
        balance=int(data.get("balance", 0)),
        seconds_billed=int(data.get("secondsBilled", 0)),
        should_disconnect=bool(data.get("shouldDisconnect", False)),
        seconds_left=int(data.get("seconds", 0)),
    )


async def session_close(session_id: str, reason: str = "client_close") -> None:
    """End a session. Best-effort: this runs during teardown, often while
    something else has already gone wrong, and failing to close cleanly must not
    mask the original error. The server closes abandoned sessions on its own
    heartbeat timeout regardless."""
    from contextlib import suppress

    with suppress(Exception):
        await _post("session/close", {"sessionId": session_id, "reason": reason})
