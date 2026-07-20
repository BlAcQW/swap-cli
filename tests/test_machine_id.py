"""`machine_id` must be stable across launches.

It used to be derived from hardware on every call. That isn't stable —
`uuid.getnode()` returns a random value when it can't read a real MAC, and it
shifts when adapters appear (installing a virtual camera driver is enough).
The licence server registers each new id against the licence and refuses past
`maxMachines`, so a drifting id silently burned seats until validation failed
with `too_many_machines`. With prepaid credit attached to the licence key, the
same drift would lock a customer out of balance they paid for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_config_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.toml")
    return tmp_path


def test_machine_id_is_persisted_on_first_use() -> None:
    generated = config.machine_id()
    assert config.load().machine_id == generated


def test_machine_id_survives_hardware_changes(monkeypatch) -> None:
    """The whole point: once persisted, a new MAC/hostname must NOT change it."""
    first = config.machine_id()

    # Simulate a new network adapter + a rename — both feed the legacy derivation.
    monkeypatch.setattr(config.uuid, "getnode", lambda: 0xDEADBEEF)
    monkeypatch.setattr(config.platform, "node", lambda: "renamed-laptop")

    assert config.machine_id() == first


def test_existing_installs_freeze_their_current_id() -> None:
    """Upgrades must keep the id the server already registered, or they'd burn
    another seat. So the first persisted value is the LEGACY derived one, not a
    fresh random UUID."""
    assert config.machine_id() == config._derive_machine_id()


def test_a_previously_saved_id_is_respected() -> None:
    config.update(machine_id="a" * 32)
    assert config.machine_id() == "a" * 32


def test_machine_id_keeps_the_format_the_server_validates() -> None:
    """The backend's zod schema is /^[0-9a-f]{32}$/ — a malformed id is a 400."""
    import re

    assert re.fullmatch(r"[0-9a-f]{32}", config.machine_id())


def test_unwritable_config_still_returns_a_usable_id(monkeypatch) -> None:
    """A read-only config dir must not break licence validation entirely."""

    def _boom(**_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(config, "update", _boom)
    assert len(config.machine_id()) == 32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
