"""`swap setup` / `swap doctor` surface.

Setup collects the LICENSE only — the backend API key (Decart or fal) is entered
in ⚙ Settings, and either provider is enough to go live. These cover the bits
that decide whether a user can actually start a session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _install_stubs() -> None:
    """`swap_cli.cli` imports runtime → camera → aiortc/cv2/av at module load.
    Stub them (same approach as test_runtime_timeout) so the CLI surface can be
    tested without the native media stack installed."""
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

from swap_cli import config  # noqa: E402


def _cfg(**kwargs) -> config.Config:
    base = dict(
        license_key=None,
        decart_api_key=None,
        license_cached_at=None,
        license_cached_valid_until=None,
    )
    return config.Config(**{**base, **kwargs})


def test_provider_key_follows_the_selected_provider() -> None:
    """Holding a key for the OTHER backend doesn't help at Live — doctor must
    report the key belonging to the provider actually selected."""
    from swap_cli.cli import _provider_key_for

    fal_user = _cfg(provider="fal", fal_api_key="fal_sk_1", decart_api_key=None)
    assert _provider_key_for(fal_user) == "fal_sk_1"

    decart_user = _cfg(provider="decart", decart_api_key="dct_1")
    assert _provider_key_for(decart_user) == "dct_1"

    # Selected fal, but only a Decart key on file → nothing usable for fal.
    mismatched = _cfg(provider="fal", decart_api_key="dct_1", fal_api_key=None)
    assert _provider_key_for(mismatched) is None
    assert mismatched.has_provider_key is True  # a key exists, just the wrong one


def test_provider_key_defaults_to_decart_and_ignores_case() -> None:
    from swap_cli.cli import _provider_key_for

    assert _provider_key_for(_cfg(provider="", decart_api_key="dct_1")) == "dct_1"
    assert _provider_key_for(_cfg(provider="  FAL ", fal_api_key="fal_sk_1")) == "fal_sk_1"


def test_setup_saves_license_without_any_provider_key(tmp_path, monkeypatch) -> None:
    """The core change: a bare license is enough for setup to succeed."""
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    from typer.testing import CliRunner

    from swap_cli.cli import app

    result = CliRunner().invoke(app, ["setup", "--license", "SWAP-CLI-AAAA"])
    assert result.exit_code == 0, result.output

    saved = config.load()
    assert saved.license_key == "SWAP-CLI-AAAA"
    assert saved.has_provider_key is False
    # And it tells the user what's still missing rather than looking "done".
    assert "Settings" in result.output


def test_setup_rejects_an_empty_license(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    from typer.testing import CliRunner

    from swap_cli.cli import app

    result = CliRunner().invoke(app, ["setup", "--license", "   "])
    assert result.exit_code == 2
    assert config.load().license_key is None  # nothing saved


def test_setup_does_not_wipe_provider_keys_set_in_settings(tmp_path, monkeypatch) -> None:
    """Re-running setup to re-enter a license must not clear the fal key the
    user configured in ⚙ Settings."""
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    config.update(license_key="OLD", fal_api_key="fal_sk_keepme")

    from typer.testing import CliRunner

    from swap_cli.cli import app

    result = CliRunner().invoke(app, ["setup", "--license", "SWAP-CLI-NEW"])
    assert result.exit_code == 0, result.output

    saved = config.load()
    assert saved.license_key == "SWAP-CLI-NEW"
    assert saved.fal_api_key == "fal_sk_keepme"  # preserved


def test_config_flags_a_provider_key_mismatch(tmp_path, monkeypatch) -> None:
    """provider=fal with only a Decart key on file is the failure that only
    shows up at Live — `swap config` must call it out, not look healthy."""
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    config.update(license_key="L", provider="fal", decart_api_key="dct_1")

    from typer.testing import CliRunner

    from swap_cli.cli import app

    out = CliRunner().invoke(app, ["config"]).output
    assert "fal" in out
    assert "isn't set" in out  # the mismatch is surfaced


def test_doctor_exits_nonzero_when_a_check_fails(tmp_path, monkeypatch) -> None:
    """Regression: rich's `table.rows` holds Row metadata, not cell text, so the
    old `"✗" in str(row)` scrape never matched and doctor always exited 0 —
    meaning a broken install reported success to any script gating on it."""
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    # No license and no provider key → guaranteed ✗ rows.
    from typer.testing import CliRunner

    from swap_cli.cli import app

    result = CliRunner().invoke(app, ["doctor"])
    assert "✗" in result.output
    assert result.exit_code == 1


def test_setup_accepts_provider_keys_for_headless_installs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    from typer.testing import CliRunner

    from swap_cli.cli import app

    result = CliRunner().invoke(
        app, ["setup", "--license", "SWAP-CLI-AAAA", "--fal-key", "fal_sk_9"]
    )
    assert result.exit_code == 0, result.output

    saved = config.load()
    assert saved.fal_api_key == "fal_sk_9"
    assert saved.is_complete is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
