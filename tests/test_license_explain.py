"""License reason codes must reach users as English, not machine codes.

`swap doctor` used to print `license validate ✗ http_403`, which told the user
nothing about what was wrong or what to do. Every surface (doctor, `swap run`,
the GUI status bar) now runs the code through `license.explain`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import license  # noqa: E402


def test_http_403_says_what_is_wrong_and_what_to_do() -> None:
    text = license.explain("http_403")
    assert "http_403" not in text
    assert "refused" in text
    # names the likely causes and the next step
    assert "expired" in text and "another machine" in text
    assert "swap setup" in text


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("ok", "valid"),
        ("no_license_key", "swap setup"),
        ("expired", "renew"),
        ("machine_limit", "another machine"),
        ("http_401", "swap setup"),
        ("http_404", "update swap-cli"),
        ("http_429", "wait a minute"),
    ],
)
def test_known_reasons_are_translated(reason: str, expected: str) -> None:
    text = license.explain(reason)
    assert expected in text
    # Machine-shaped codes must never leak through. (Plain-word reasons like
    # "expired" legitimately appear in their own message, so they're exempt.)
    if "_" in reason:
        assert reason not in text


def test_server_errors_are_grouped_as_retryable() -> None:
    for code in ("http_500", "http_502", "http_503"):
        text = license.explain(code)
        assert "try again" in text
        assert code not in text


def test_unknown_http_code_still_reads_as_english() -> None:
    text = license.explain("http_418")
    assert "rejected the check" in text
    assert "418" in text  # keep the number for support/debugging


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("license_not_found", "not recognised"),
        ("license_revoked", "revoked"),
        ("license_expired", "expired"),
        ("invalid_format", "valid key"),
        ("too_many_machines", "maximum number of machines"),
    ],
)
def test_real_backend_codes_are_translated(reason: str, expected: str) -> None:
    """These are the exact codes services/licenses.ts returns — they reach the
    user through a 403 body, so they must not surface as raw snake_case."""
    text = license.explain(reason)
    assert expected in text
    assert reason not in text


class _Resp:
    def __init__(self, payload, *, raises=False):
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not JSON")
        return self._payload


def test_reason_is_read_from_the_details_field() -> None:
    """The backend's envelope is {success, error, details:{reason}} — the
    details.reason is the authoritative code."""
    resp = _Resp({"success": False, "error": "too_many_machines",
                  "details": {"reason": "too_many_machines"}})
    assert license._reason_from_body(resp) == "too_many_machines"


def test_reason_falls_back_to_the_error_field() -> None:
    assert license._reason_from_body(_Resp({"success": False, "error": "rate_limited"})) == (
        "rate_limited"
    )


def test_non_json_body_yields_none_so_the_status_code_is_used() -> None:
    """A proxy/CDN can return HTML here. A broken error path must not mask the
    original failure — we fall back to http_<status>."""
    assert license._reason_from_body(_Resp(None, raises=True)) is None
    assert license._reason_from_body(_Resp("just a string")) is None
    assert license._reason_from_body(_Resp({"success": False})) is None


def test_backend_supplied_text_passes_through_untouched() -> None:
    """The server may send its own message — that's more useful than anything
    we'd substitute, so don't mangle it."""
    assert license.explain("Your subscription lapsed on 2026-01-01") == (
        "Your subscription lapsed on 2026-01-01"
    )
    assert license.explain("http_notanumber") == "http_notanumber"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
