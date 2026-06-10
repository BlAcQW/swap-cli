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
    # Sprint 15: per-frame watermark removal. The Decart "AI Generated"
    # badge roams the frame, so removal runs template-match + inpaint on
    # every frame. All optional — older config files load fine. Off by
    # default so existing users aren't surprised by the latency cost.
    remove_watermark: bool = False
    watermark_template: str | None = None  # path to captured watermark PNG
    watermark_method: str = "template"  # "template" | "threshold"
    # How a located badge is hidden: "reconstruct" (invisible rebuild) or
    # "blur" (smear into an unreadable soft patch). Detection is separate.
    watermark_removal: str = "reconstruct"  # "reconstruct" | "blur"
    watermark_threshold: float = 0.50  # matchTemplate confidence gate (0..1)
    watermark_inpaint_radius: int = 3
    # Frame width the captured template was grabbed at. Decart's output
    # resolution varies, so this centers the multi-scale match exactly for a
    # user-captured template. None → the 1280px bundled-default assumption.
    watermark_template_width: int | None = None

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
        remove_watermark=bool(data.get("remove_watermark", False)),
        watermark_template=_clean(data.get("watermark_template")),
        watermark_method=_clean(data.get("watermark_method")) or "template",
        watermark_removal=_clean(data.get("watermark_removal")) or "reconstruct",
        watermark_threshold=_float_or_none(data.get("watermark_threshold")) or 0.50,
        watermark_inpaint_radius=_int_or_none(data.get("watermark_inpaint_radius")) or 3,
        watermark_template_width=_int_or_none(data.get("watermark_template_width")),
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
    if cfg.remove_watermark:
        body.append("remove_watermark = true")
    if cfg.watermark_template:
        body.append(f'watermark_template = "{_escape(cfg.watermark_template)}"')
    if cfg.watermark_method and cfg.watermark_method != "template":
        body.append(f'watermark_method = "{_escape(cfg.watermark_method)}"')
    if cfg.watermark_removal and cfg.watermark_removal != "reconstruct":
        body.append(f'watermark_removal = "{_escape(cfg.watermark_removal)}"')
    if cfg.watermark_threshold != 0.50:
        body.append(f"watermark_threshold = {cfg.watermark_threshold}")
    if cfg.watermark_inpaint_radius != 3:
        body.append(f"watermark_inpaint_radius = {cfg.watermark_inpaint_radius}")
    if cfg.watermark_template_width is not None:
        body.append(f"watermark_template_width = {cfg.watermark_template_width}")

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
        remove_watermark=kwargs.get("remove_watermark", current.remove_watermark),
        watermark_template=kwargs.get("watermark_template", current.watermark_template),
        watermark_method=kwargs.get("watermark_method", current.watermark_method),
        watermark_removal=kwargs.get("watermark_removal", current.watermark_removal),
        watermark_threshold=kwargs.get("watermark_threshold", current.watermark_threshold),
        watermark_inpaint_radius=kwargs.get(
            "watermark_inpaint_radius", current.watermark_inpaint_radius
        ),
        watermark_template_width=kwargs.get(
            "watermark_template_width", current.watermark_template_width
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
