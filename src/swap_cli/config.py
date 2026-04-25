"""Local config: Decart API key + license key, stored in user config dir."""

from __future__ import annotations

import hashlib
import os
import platform
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

APP_NAME = "swap-cli"
CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class Config:
    """User configuration. Read-only at runtime."""

    license_key: str | None
    decart_api_key: str | None
    license_cached_at: int | None  # unix seconds; for offline grace
    license_cached_valid_until: int | None  # unix seconds

    @property
    def is_complete(self) -> bool:
        return bool(self.license_key) and bool(self.decart_api_key)


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / CONFIG_FILENAME


def load() -> Config:
    """Load config from disk. Returns an empty config if file is missing."""
    path = config_path()
    if not path.exists():
        return Config(None, None, None, None)
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return Config(None, None, None, None)

    return Config(
        license_key=_clean(data.get("license_key")),
        decart_api_key=_clean(data.get("decart_api_key")),
        license_cached_at=_int_or_none(data.get("license_cached_at")),
        license_cached_valid_until=_int_or_none(data.get("license_cached_valid_until")),
    )


def save(cfg: Config) -> Path:
    """Write config atomically with restrictive perms."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    body: list[str] = []
    if cfg.license_key:
        body.append(f'license_key = "{_escape(cfg.license_key)}"')
    if cfg.decart_api_key:
        body.append(f'decart_api_key = "{_escape(cfg.decart_api_key)}"')
    if cfg.license_cached_at is not None:
        body.append(f"license_cached_at = {cfg.license_cached_at}")
    if cfg.license_cached_valid_until is not None:
        body.append(f"license_cached_valid_until = {cfg.license_cached_valid_until}")

    text = "\n".join(body) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def update(**kwargs: Any) -> Config:
    """Patch the on-disk config with new values."""
    current = load()
    merged = Config(
        license_key=kwargs.get("license_key", current.license_key),
        decart_api_key=kwargs.get("decart_api_key", current.decart_api_key),
        license_cached_at=kwargs.get("license_cached_at", current.license_cached_at),
        license_cached_valid_until=kwargs.get(
            "license_cached_valid_until", current.license_cached_valid_until
        ),
    )
    save(merged)
    return merged


def machine_id() -> str:
    """Stable, hashed device identifier. Never sends raw MAC/serial off-device."""
    raw = f"{uuid.getnode()}|{platform.node()}|{platform.machine()}|{platform.system()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
