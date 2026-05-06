"""Prerequisite checks for the optional voice-cloning feature.

Pure-Python — never imports torch / sounddevice. Uses `importlib.util.find_spec`
so it works even before the `[voice]` extra is installed. The Enable Voice
modal in the GUI calls this to decide what to show the user.

Sprint 14e: OpenVoice removed. Voice = RVC only. Adds ffmpeg + Visual C++
Build Tools preflight checks per the user-supplied RVC setup guide.
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
RVC_MODELS_DIRNAME = "rvc"


def models_dir() -> Path:
    """Where RVC weights live once `swap voices install` runs."""
    return Path(user_data_dir(APP_NAME)) / "models"


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
    ffmpeg: Check
    build_tools: Check
    audio_cable: Check

    @property
    def all_ok(self) -> bool:
        return (
            self.gpu.ok
            and self.deps_installed.ok
            and self.ffmpeg.ok
            and self.build_tools.ok
            and self.audio_cable.ok
        )

    @property
    def gpu_blocked(self) -> bool:
        """No supported GPU at all — voice can't run on this machine."""
        return not self.gpu.ok


def check_all() -> PrereqResult:
    return PrereqResult(
        gpu=_check_gpu(),
        deps_installed=_check_deps(),
        ffmpeg=_check_ffmpeg(),
        build_tools=_check_build_tools(),
        audio_cable=_check_audio_cable(),
    )


# ── GPU ───────────────────────────────────────────────────────────────────


def _check_gpu() -> Check:
    """Detect a supported GPU without importing torch.

    On Windows / Linux: shell out to `nvidia-smi` and check return code.
    On macOS: Apple Silicon CPU is too slow for live RVC streaming
    (per the upstream RVC project — PyTorch 2.6 + fairseq + MPS combo
    is broken). Mac is honestly not supported for v1.
    """
    if sys.platform == "darwin":
        if platform.machine() == "arm64":
            return Check(
                ok=False,
                label="Apple Silicon — CPU only",
                hint=(
                    "RVC live streaming requires NVIDIA. "
                    "Mac CPU path is too slow for real-time."
                ),
            )
        return Check(
            ok=False,
            label="Intel Mac",
            hint="Voice requires NVIDIA RTX 3060+ on Windows/Linux.",
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
        label="No NVIDIA GPU",
        hint="Voice requires NVIDIA RTX 3060+.",
    )


# ── Voice deps installed ──────────────────────────────────────────────────


def _check_deps() -> Check:
    """True iff every voice-cloning runtime import is available.

    `swap voices install` short-circuits when this returns ok, so this
    list MUST mirror everything install_voice_deps() pulls in. Sprint 14e
    drops OpenVoice — voice path is now RVC-only.
    """
    required = (
        "torch",
        "torchaudio",
        "sounddevice",
        "librosa",
        "rvc_python",
        "fairseq",   # rvc-python imports fairseq for HuBERT loader
        # Sprint 14g.3: fairseq's runtime deps (we install fairseq itself
        # with --no-deps, so any of these missing means a half-install
        # that crashes at session start).
        "hydra",     # hydra-core
        "bitarray",
        "regex",
        "sacrebleu",
        "sklearn",   # scikit-learn import name
    )
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        return Check(ok=True, label="voice deps installed")
    return Check(
        ok=False,
        label=f"missing: {', '.join(missing)}",
        hint="run `swap voices install`",
    )


# ── ffmpeg on PATH ────────────────────────────────────────────────────────


def _check_ffmpeg() -> Check:
    """Half the 'file failed to load' errors in RVC trace to missing
    ffmpeg. Detect early."""
    if shutil.which("ffmpeg") is None:
        if sys.platform == "win32":
            hint = "winget install Gyan.FFmpeg  (or grab from https://gyan.dev/ffmpeg/)"
        elif sys.platform == "darwin":
            hint = "brew install ffmpeg"
        else:
            hint = "sudo apt install ffmpeg  (or your distro's equivalent)"
        return Check(ok=False, label="ffmpeg not on PATH", hint=hint)
    return Check(ok=True, label="ffmpeg on PATH")


# ── Visual C++ Build Tools (Windows-only) ─────────────────────────────────


def _check_build_tools() -> Check:
    """Some RVC deps (pyworld, faiss-cpu when no wheel) compile from
    source on Windows — that needs Visual C++ Build Tools. Cryptic
    wheel-build failures otherwise. Linux/Mac use system gcc/clang
    (xcode-select on Mac is checked separately in install_voice_deps).
    """
    if sys.platform != "win32":
        return Check(ok=True, label="not applicable on this platform")

    # Detect by looking for the standard install path. Microsoft puts
    # Build Tools (MSVC) under VS BuildTools / VS Installer.
    candidates = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
    ]
    if any(p.exists() for p in candidates):
        return Check(ok=True, label="Visual C++ Build Tools installed")
    return Check(
        ok=False,
        label="Visual C++ Build Tools not detected",
        hint=(
            "Some RVC deps build from source. Install from "
            "https://visualstudio.microsoft.com/visual-cpp-build-tools/ "
            "(check 'Desktop development with C++')."
        ),
    )


# ── Virtual audio cable ───────────────────────────────────────────────────


def _check_audio_cable() -> Check:
    """Detect a virtual audio cable so swap can route the cloned voice
    into Zoom / OBS / Discord. Hint depends on platform.
    """
    if sys.platform == "darwin":
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
