"""Regression tests for the config write path (WinError 32 crash).

The GUI writes config from several threads at once — the launch-time licence
check, the credit refresh, and the Live worker's force-online re-validate can
all reach save() together. That raced on the single "config.toml.tmp" staging
file and, on Windows, one os.replace() blew up with:

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process

These cover the two defences: a process-wide write lock (no lost updates, no
staging collision) and a bounded retry around the atomic replace (survives an
antivirus/indexer holding the file for a beat).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from swap_cli import config


@pytest.fixture(autouse=True)
def isolate_config_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    # Don't actually sleep between retries in tests.
    monkeypatch.setattr(config, "_REPLACE_BACKOFF_SEC", 0.0)
    return tmp_path


def test_concurrent_updates_preserve_all_distinct_fields(tmp_path: Path) -> None:
    """Many threads each set a different field at once. With the write lock,
    every field survives (a read-modify-write race would drop some) and nothing
    raises."""
    ops = [
        lambda: config.update(license_key="SWAP-CLI-XYZ"),
        lambda: config.update(fal_api_key="fal_key_1234567890"),
        lambda: config.update(provider="fal"),
        lambda: config.update(last_microphone=7),
        lambda: config.update(voice_enabled=True),
    ]
    errors: list[Exception] = []

    def run(op) -> None:
        try:
            op()
        except Exception as exc:  # pragma: no cover - only trips on a real bug
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(op,)) for op in ops for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    cfg = config.load()
    assert cfg.license_key == "SWAP-CLI-XYZ"
    assert cfg.fal_api_key == "fal_key_1234567890"
    assert cfg.provider == "fal"
    assert cfg.last_microphone == 7
    assert cfg.voice_enabled is True
    # No staging files left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_retries_transient_replace_lock(tmp_path: Path, monkeypatch) -> None:
    """A couple of WinError-32-style failures then success → save retries and
    wins, the config lands, and no staging file is left behind."""
    real_replace = Path.replace
    calls = {"n": 0}

    def flaky_replace(self: Path, target) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "used by another process")
        real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    config.update(license_key="SWAP-CLI-RETRY")

    assert calls["n"] == 3  # failed twice, succeeded on the third
    assert config.load().license_key == "SWAP-CLI-RETRY"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_gives_up_and_cleans_tmp_when_never_unlocks(
    tmp_path: Path, monkeypatch
) -> None:
    """If the lock never clears, save() re-raises after its retries — but must
    not leave a half-written staging file lying around."""

    def always_locked(self: Path, target) -> None:
        raise PermissionError(32, "used by another process")

    monkeypatch.setattr(Path, "replace", always_locked)

    with pytest.raises(PermissionError):
        config.update(license_key="SWAP-CLI-NEVER")

    assert list(tmp_path.glob("*.tmp")) == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
