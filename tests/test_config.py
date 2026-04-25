"""Config module: load/save/update round-trip + machine_id stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from swap_cli import config


@pytest.fixture(autouse=True)
def isolate_config_dir(monkeypatch, tmp_path: Path) -> Path:
    """Redirect platformdirs to a tmp dir so tests never touch the real config."""
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    return tmp_path


def test_load_when_missing_returns_empty_config() -> None:
    cfg = config.load()
    assert cfg.license_key is None
    assert cfg.decart_api_key is None
    assert cfg.is_complete is False


def test_save_and_load_round_trips(tmp_path: Path) -> None:
    cfg = config.Config(
        license_key="SWAP-CLI-AAAA-BBBB-CCCC",
        decart_api_key="dct_swap_test123",
        license_cached_at=1_700_000_000,
        license_cached_valid_until=1_700_086_400,
    )
    config.save(cfg)
    reloaded = config.load()
    assert reloaded == cfg
    assert reloaded.is_complete is True


def test_update_patches_existing_values() -> None:
    config.update(license_key="L1", decart_api_key="D1")
    config.update(decart_api_key="D2")
    cfg = config.load()
    assert cfg.license_key == "L1"
    assert cfg.decart_api_key == "D2"


def test_save_strips_whitespace_on_load() -> None:
    config.save(config.Config("  spaced  ", "dct_x", None, None))
    # On load we only strip; the file itself preserves what we wrote.
    cfg = config.load()
    # _clean strips the loaded value — verify no leading/trailing whitespace.
    assert cfg.license_key == "spaced"


def test_machine_id_is_stable_and_hex_32() -> None:
    a = config.machine_id()
    b = config.machine_id()
    assert a == b
    assert len(a) == 32
    assert all(c in "0123456789abcdef" for c in a)


def test_escape_handles_quotes_and_backslashes() -> None:
    cfg = config.Config(
        license_key='has "quotes" and \\backslashes',
        decart_api_key="dct_x",
        license_cached_at=None,
        license_cached_valid_until=None,
    )
    config.save(cfg)
    reloaded = config.load()
    assert reloaded.license_key == 'has "quotes" and \\backslashes'
