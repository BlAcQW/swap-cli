"""Local config: Decart API key + license key, stored in user config dir."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import threading
import time
import tomllib
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

APP_NAME = "swap-cli"
CONFIG_FILENAME = "config.toml"

# Serialises config writes within the process. The GUI writes config from
# several threads at once — the launch-time licence check, the credit refresh,
# and the Live worker's force-online re-validate can all land on save() at the
# same instant. Without this they race on the temp file and, on Windows, one
# os.replace() fails with WinError 32 ("used by another process"). Reentrant so
# update() can hold it across load→merge→save without deadlocking on save().
_WRITE_LOCK = threading.RLock()

# Windows only: os.replace() also fails transiently with WinError 32 when an
# antivirus / Search indexer / cloud-sync agent has the target open for a beat.
# Retry a few times before giving up.
_REPLACE_RETRIES = 10
_REPLACE_BACKOFF_SEC = 0.05


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
    # every frame. All optional — older config files load fine. None = never
    # chosen: the CLI treats that as off (latency cost), the GUI as on; an
    # explicit False is saved so turning the GUI switch off sticks.
    remove_watermark: bool | None = None
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
    # Sprint 17: independent color/shape safety net that catches faint badges
    # template matching misses. On by default (bullet-proofing); flip off if it
    # ever covers a non-badge region.
    watermark_signature_fallback: bool = True
    # Device identifier sent with every licence check. Persisted rather than
    # recomputed — see machine_id() for why a hardware-derived value drifts.
    machine_id: str | None = None
    # Sprint 18: realtime backend — "decart" (default) or "fal". fal resells the
    # same Decart Lucy model on its own account, so it's both a second source and
    # a possible way around Decart's watermark (which they refuse to lift for us).
    provider: str = "decart"
    # Which camera driver receives the output. "obs" = OBS Virtual Camera (a
    # DirectShow filter — Zoom/Meet/Discord see it, WhatsApp Desktop does NOT).
    # "droidcam" = DroidCam's real camera driver, which Store/Media Foundation
    # apps enumerate, so WhatsApp accepts it. Windows only.
    vcam_backend: str = "obs"
    fal_api_key: str | None = None
    # How this licence pays for compute, as reported by the licence server:
    # "byok" = the user's own API key, "managed" = prepaid credit we meter.
    # Cached from the last validation so the UI can render correctly before the
    # first network round-trip; the server remains the authority.
    license_mode: str = "byok"

    @property
    def is_managed(self) -> bool:
        """True when compute is billed against prepaid credit rather than the
        user's own API key."""
        return (self.license_mode or "byok").strip().lower() == "managed"

    @property
    def has_provider_key(self) -> bool:
        """True when a realtime backend is usable.

        A managed licence has NO provider key by design — that's the whole point,
        the key is ours and the server mints per-session tokens. Without the
        managed branch here `is_complete` is False for every managed customer,
        and they're turned away at the Live button with "run `swap setup`"
        before any credit check can run.
        """
        if self.is_managed:
            return True
        return bool(self.decart_api_key) or bool(self.fal_api_key)

    @property
    def is_complete(self) -> bool:
        return bool(self.license_key) and self.has_provider_key


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / CONFIG_FILENAME


def load() -> Config:
    """Load config from disk. Returns an empty config if file is missing."""
    path = config_path()
    if not path.exists():
        return Config(None, None, None, None)
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as err:
        # Don't pretend the config is empty — that turned a corrupt/unreadable
        # file into a confusing "run setup" loop. Surface it so it's fixable.
        print(f"[config] could not read {path}: {err}", file=sys.stderr)
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
        remove_watermark=(
            None if data.get("remove_watermark") is None else bool(data["remove_watermark"])
        ),
        watermark_template=_clean(data.get("watermark_template")),
        watermark_method=_clean(data.get("watermark_method")) or "template",
        watermark_removal=_clean(data.get("watermark_removal")) or "reconstruct",
        watermark_threshold=_float_or_none(data.get("watermark_threshold")) or 0.50,
        watermark_inpaint_radius=_int_or_none(data.get("watermark_inpaint_radius")) or 3,
        watermark_template_width=_int_or_none(data.get("watermark_template_width")),
        watermark_signature_fallback=bool(data.get("watermark_signature_fallback", True)),
        machine_id=_clean(data.get("machine_id")),
        provider=_clean(data.get("provider")) or "decart",
        vcam_backend=_clean(data.get("vcam_backend")) or "obs",
        fal_api_key=_clean(data.get("fal_api_key")),
        license_mode=_clean(data.get("license_mode")) or "byok",
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
    if cfg.remove_watermark is not None:
        body.append(f"remove_watermark = {'true' if cfg.remove_watermark else 'false'}")
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
    if not cfg.watermark_signature_fallback:  # only persist the non-default (off)
        body.append("watermark_signature_fallback = false")
    if cfg.machine_id:
        body.append(f'machine_id = "{_escape(cfg.machine_id)}"')
    if cfg.provider and cfg.provider != "decart":
        body.append(f'provider = "{_escape(cfg.provider)}"')
    if cfg.vcam_backend and cfg.vcam_backend != "obs":
        body.append(f'vcam_backend = "{_escape(cfg.vcam_backend)}"')
    if cfg.fal_api_key:
        body.append(f'fal_api_key = "{_escape(cfg.fal_api_key)}"')
    if cfg.license_mode and cfg.license_mode != "byok":
        body.append(f'license_mode = "{_escape(cfg.license_mode)}"')

    text = "\n".join(body) + "\n"
    # Per-writer temp name (pid + thread + random) so two concurrent saves can
    # never clobber each other's staging file — the shared "config.toml.tmp"
    # was the collision that surfaced as WinError 32 on Windows.
    tmp = path.with_suffix(
        f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    with _WRITE_LOCK:
        try:
            tmp.write_text(text, encoding="utf-8")
            if os.name != "nt":
                os.chmod(tmp, 0o600)
            _replace_with_retry(tmp, path)
        finally:
            # Never leave a staging file behind (e.g. if replace ultimately failed).
            with suppress(OSError):
                if tmp.exists():
                    tmp.unlink()
    return path


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """os.replace, retried against transient Windows file locks (WinError 32).

    On POSIX replace is atomic and never hits this, so the first attempt wins.
    On Windows an AV/indexer/sync agent can briefly hold the destination open;
    a short backoff clears it. Re-raises the last error if it never clears.
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_SEC * (attempt + 1))


def update(**kwargs: Any) -> Config:
    """Patch the on-disk config with new values.

    Held under the write lock so a concurrent load→merge→save from another
    thread can't lose this update (last-writer-wins) or collide on the file.
    """
    with _WRITE_LOCK:
        return _update_locked(**kwargs)


def _update_locked(**kwargs: Any) -> Config:
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
        watermark_signature_fallback=kwargs.get(
            "watermark_signature_fallback", current.watermark_signature_fallback
        ),
        machine_id=kwargs.get("machine_id", current.machine_id),
        provider=kwargs.get("provider", current.provider),
        vcam_backend=kwargs.get("vcam_backend", current.vcam_backend),
        fal_api_key=kwargs.get("fal_api_key", current.fal_api_key),
        license_mode=kwargs.get("license_mode", current.license_mode),
    )
    save(merged)
    return merged


def _derive_machine_id() -> str:
    """Hash the machine's identifying traits. Never sends raw MAC/serial off-device."""
    raw = f"{uuid.getnode()}|{platform.node()}|{platform.machine()}|{platform.system()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def machine_id() -> str:
    """Stable device identifier, persisted on first use.

    It used to be derived from hardware on every call, which is NOT stable:
    `uuid.getnode()` returns a *random* value whenever it can't read a real MAC,
    and it changes when network adapters come and go (installing a virtual
    camera driver or connecting a VPN is enough). `platform.node()` is the
    hostname, which changes on rename.

    That mattered because the licence server registers each new id against the
    licence and refuses once `maxMachines` is exceeded — so a drifting id
    silently burned seats until validation started failing with
    `too_many_machines`. Once prepaid credit is attached to a licence key, the
    same drift would lock a customer out of balance they paid for.

    We persist the value the first time we're asked. On an existing install that
    freezes whatever the server has *already* registered, so upgrading doesn't
    consume another seat; a fresh install simply freezes whatever it starts with.
    Best-effort: if the config can't be written we still return a usable id.
    """
    cfg = load()
    if cfg.machine_id:
        return cfg.machine_id

    derived = _derive_machine_id()
    try:
        update(machine_id=derived)
    except OSError as err:  # read-only config dir, permissions, disk full…
        print(f"[config] could not persist machine_id: {err}", file=sys.stderr)
    return derived


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
