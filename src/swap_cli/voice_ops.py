"""Implementations for the `swap voices` subcommand group.

Sprint 13b.1: list/add/remove are real (operate on the user voice
directory). install runs `pip install '[voice]'` and downloads OpenVoice
weights. Actual voice cloning still no-ops — wired in 13b.2.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .voice_library import (
    Voice,
    delete_user_voice,
    load_all_voices,
    load_library_voices,
    load_user_voices,
    save_user_voice,
    slugify,
    user_voices_dir,
)
from .voice_model import extract_embedding
from .voice_prereq import (
    OPENVOICE_WEIGHTS_DIRNAME,
    check_all,
    models_dir,
    openvoice_weights_dir,
)


# ── Install: pip dep + weight download ────────────────────────────────────


@dataclass(frozen=True)
class InstallResult:
    deps_installed: bool
    weights_installed: bool
    error: str | None = None


# OpenVoice v2 weights are ~5 GB. The real download URL belongs to
# MyShell (the OpenVoice authors) — we mirror it here as a constant so
# 13b.2 can flip to our own CDN if needed.
OPENVOICE_RELEASE_URL = (
    "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main"
)


def install_voice_deps() -> bool:
    """Run `pip install 'swap-cli[voice]'` against the current interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", "swap-cli[voice]"]
    if (Path.cwd() / "pyproject.toml").exists():
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[voice]"]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def download_openvoice_weights(progress: callable | None = None) -> Path:  # type: ignore[valid-type]
    """Download OpenVoice v2 weights to ~/.local/share/swap-cli/models/openvoice-v2/.

    13b.1: marks the directory as ready with a sentinel file so the prereq
    check passes. Real weight download arrives in 13b.2 once we've nailed
    which files OpenVoice's converter actually needs.
    """
    target = openvoice_weights_dir()
    target.mkdir(parents=True, exist_ok=True)

    # 13b.2: actually download these. For 13b.1 we just write the sentinel
    # so the prereq check progresses. This unblocks UX testing of the
    # Enable wizard without blocking on the 5 GB download.
    sentinel = target / "installed.ok"
    sentinel.write_text(
        f"sprint=13b.1\nplaceholder=true\ndownloaded_at={int(time.time())}\n",
        encoding="utf-8",
    )

    if progress:
        progress(1.0)
    return target


def uninstall_openvoice_weights() -> bool:
    """Delete the OpenVoice weights directory. Returns True if removed."""
    target = openvoice_weights_dir()
    if not target.exists():
        return False
    import shutil

    shutil.rmtree(target)
    return True


# ── Library/user voice operations ─────────────────────────────────────────


def list_all() -> tuple[list[Voice], list[Voice]]:
    """Return (library_voices, user_voices) for display."""
    return load_library_voices(), load_user_voices()


def add_user_voice(wav_path: Path, display_name: str | None = None) -> Voice:
    """Extract an OpenVoice embedding from a reference audio file and
    save it as a user voice.

    Raises:
        FileNotFoundError: if wav_path doesn't exist.
        RuntimeError: if voice deps aren't installed.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {wav_path}")

    name = display_name or wav_path.stem.replace("_", " ").replace("-", " ").strip()
    voice_id = slugify(name)

    # 13b.1: extract_embedding currently returns a deterministic placeholder
    # so the round-trip works. 13b.2 plugs in real OpenVoice extraction.
    embedding = extract_embedding(wav_path)

    voice = Voice(
        id=voice_id,
        name=name,
        description=f"Custom voice from {wav_path.name}",
        source="user",
        embedding=embedding,
        sample_rate=16_000,
        created_at=int(time.time()),
    )
    save_user_voice(voice)
    return voice


def remove_user_voice(name_or_id: str) -> bool:
    """Remove a user voice by id or name. Returns True if removed."""
    target_id = slugify(name_or_id)
    if delete_user_voice(target_id):
        return True
    # Fall back: maybe they passed the display name as-is.
    for v in load_user_voices():
        if v.name.lower() == name_or_id.lower() or v.id == name_or_id:
            return delete_user_voice(v.id)
    return False
