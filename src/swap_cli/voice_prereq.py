"""Prerequisite checks for the optional voice-cloning feature.

Pure-Python — never imports torch / sounddevice. Uses `importlib.util.find_spec`
so it works even before the `[voice]` extra is installed. The Enable Voice
modal in the GUI calls this to decide what to show the user.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "swap-cli"
OPENVOICE_WEIGHTS_DIRNAME = "openvoice-v2"
RVC_MODELS_DIRNAME = "rvc"


def models_dir() -> Path:
    """Where OpenVoice/RVC weights live once `swap voices install` runs."""
    return Path(user_data_dir(APP_NAME)) / "models"


def openvoice_weights_dir() -> Path:
    return models_dir() / OPENVOICE_WEIGHTS_DIRNAME


def rvc_models_dir() -> Path:
    """Where RVC voice models (.pth + .index) live. Each subdir = one voice."""
    return models_dir() / RVC_MODELS_DIRNAME


@dataclass(frozen=True)
class Check:
    ok: bool
    label: str
    hint: str | None = None  # remediation step shown to the user when not ok


@dataclass(frozen=True)
class PrereqResult:
    gpu: Check
    deps_installed: Check
    weights: Check
    audio_cable: Check

    @property
    def all_ok(self) -> bool:
        return self.gpu.ok and self.deps_installed.ok and self.weights.ok and self.audio_cable.ok

    @property
    def gpu_blocked(self) -> bool:
        """No supported GPU at all — voice can't run on this machine."""
        return not self.gpu.ok


def check_all() -> PrereqResult:
    return PrereqResult(
        gpu=_check_gpu(),
        deps_installed=_check_deps(),
        weights=_check_weights(),
        audio_cable=_check_audio_cable(),
    )


# ── GPU ───────────────────────────────────────────────────────────────────


def _check_gpu() -> Check:
    """Detect a supported GPU without importing torch.

    On Windows / Linux: shell out to `nvidia-smi` and check return code.
    On macOS: any Apple Silicon machine (arm64) gets MPS — assume yes.
    Anything else: blocked.
    """
    if sys.platform == "darwin":
        if platform.machine() == "arm64":
            return Check(ok=True, label="Apple Silicon (MPS)")
        return Check(
            ok=False,
            label="Intel Mac",
            hint="Voice requires Apple Silicon or NVIDIA GPU.",
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=4.0,
            )
            if proc.returncode == 0:
                first_gpu = proc.stdout.splitlines()[0].strip() if proc.stdout else ""
                return Check(ok=True, label=first_gpu or "NVIDIA GPU")
        except (subprocess.TimeoutExpired, OSError):
            pass

    return Check(
        ok=False,
        label="No supported GPU",
        hint="Voice requires NVIDIA RTX 3060+ or Apple Silicon.",
    )


# ── Voice deps installed ──────────────────────────────────────────────────


def _check_deps() -> Check:
    """True iff every voice-cloning runtime import is available.

    MUST stay in sync with voice_model.voice_deps_present() — otherwise
    `swap voices install` reports ✓ while the live session refuses to
    start because OpenVoice (or another module) is missing.
    """
    required = ("torch", "torchaudio", "sounddevice", "librosa", "openvoice")
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        return Check(ok=True, label="voice deps installed")
    return Check(
        ok=False,
        label=f"missing: {', '.join(missing)}",
        hint="run `swap voices install`",
    )


# ── OpenVoice weights ─────────────────────────────────────────────────────


def _check_weights() -> Check:
    weights = openvoice_weights_dir()
    # We don't validate file contents here — too expensive for the modal.
    # `swap voices install` writes a sentinel file `installed.ok` on success.
    sentinel = weights / "installed.ok"
    if sentinel.exists():
        return Check(ok=True, label=f"weights ready ({weights})")
    return Check(
        ok=False,
        label="OpenVoice weights not downloaded (~5 GB)",
        hint="run `swap voices install`",
    )


# ── Virtual audio cable ───────────────────────────────────────────────────


def _check_audio_cable() -> Check:
    """Detect a virtual audio cable so swap can route the cloned voice
    into Zoom / OBS / Discord. Hint depends on platform.
    """
    if sys.platform == "darwin":
        # BlackHole installs a kext that registers an audio device named
        # "BlackHole 2ch" (or 16ch). Cheapest detect: look in /Library/Audio
        # for the driver bundle.
        if Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver").exists():
            return Check(ok=True, label="BlackHole 2ch installed")
        if Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver").exists():
            return Check(ok=True, label="BlackHole 16ch installed")
        return Check(
            ok=False,
            label="BlackHole not installed",
            hint="brew install blackhole-2ch",
        )

    if sys.platform == "win32":
        # VB-Cable installs as DLLs under Program Files; cheapest detect is
        # checking for the driver folder.
        candidates = [
            Path("C:/Program Files/VB/CABLE"),
            Path("C:/Program Files (x86)/VB/CABLE"),
        ]
        if any(p.exists() for p in candidates):
            return Check(ok=True, label="VB-Cable installed")
        return Check(
            ok=False,
            label="VB-Cable not installed",
            hint="https://vb-audio.com/Cable/",
        )

    # Linux: PulseAudio's `module-null-sink` or PipeWire equivalents work.
    # We can't reliably probe without importing pulse-bindings, so we treat
    # this as a soft-pass: the real gate is `swap voices install` time.
    return Check(ok=True, label="loopback available via pulseaudio")
