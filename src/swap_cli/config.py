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
    # Voice cloning preferences (sprint 13). All optional — older config files
    # without these fields load fine. voice_enabled is the sticky toggle state.
    voice_enabled: bool = False
    last_voice_id: str | None = None
    last_microphone: int | None = None
    last_voice_output: int | None = None
    # Sprint 14e: voice path is RVC-only. Field kept for forward
    # compatibility (future engines: Applio, GPT-SoVITS).
    voice_engine: str = "rvc"
    # Sprint 14i: when True, the streaming engine sets index_rate=0
    # (skip Faiss retrieval). Trades timbre quality for big speedup —
    # essential on weak GPUs or when using voices with huge .index files.
    voice_fast: bool = False

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
        voice_enabled=bool(data.get("voice_enabled", False)),
        last_voice_id=_clean(data.get("last_voice_id")),
        last_microphone=_int_or_none(data.get("last_microphone")),
        last_voice_output=_int_or_none(data.get("last_voice_output")),
        voice_engine=_clean(data.get("voice_engine")) or "rvc",
        voice_fast=bool(data.get("voice_fast", False)),
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
    if cfg.voice_enabled:
        body.append("voice_enabled = true")
    if cfg.last_voice_id:
        body.append(f'last_voice_id = "{_escape(cfg.last_voice_id)}"')
    if cfg.last_microphone is not None:
        body.append(f"last_microphone = {cfg.last_microphone}")
    if cfg.last_voice_output is not None:
        body.append(f"last_voice_output = {cfg.last_voice_output}")
    if cfg.voice_engine and cfg.voice_engine != "rvc":
        body.append(f'voice_engine = "{_escape(cfg.voice_engine)}"')
    if cfg.voice_fast:
        body.append("voice_fast = true")

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
        voice_enabled=kwargs.get("voice_enabled", current.voice_enabled),
        last_voice_id=kwargs.get("last_voice_id", current.last_voice_id),
        last_microphone=kwargs.get("last_microphone", current.last_microphone),
        last_voice_output=kwargs.get("last_voice_output", current.last_voice_output),
        voice_engine=kwargs.get("voice_engine", current.voice_engine),
        voice_fast=kwargs.get("voice_fast", current.voice_fast),
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
