"""Implementations for the `swap voices` subcommand group.

Sprint 13b.2: download_openvoice_weights pulls the real ToneColorConverter
checkpoint from HuggingFace (~300 MB). list/add/remove operate on the
user voice directory; add now uses the real OpenVoice tone-color
extractor via voice_model.extract_embedding.
"""

from __future__ import annotations

import subprocess
import sys
import time
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


# OpenVoice v2's tone-color converter checkpoint lives in MyShell's
# HuggingFace repo. We only need the `converter/` subtree (~300 MB) — the
# `base_speakers/` and `bert/` trees are TTS-specific and unused here.
OPENVOICE_HF_REPO = "myshell-ai/OpenVoiceV2"
OPENVOICE_INCLUDE_PATTERNS = ["converter/*"]


# OpenVoice's package metadata pins librosa==0.9.1, numpy==1.22, av==10.*
# (av 10 has no Win+Py3.11 wheel and source-builds break on modern Cython).
# We install OpenVoice with --no-deps so its broken constraints don't poison
# the resolver. Tone-color conversion at runtime only needs torch + numpy +
# librosa + soundfile, all of which the [voice] extra already provides.
OPENVOICE_GIT_URL = "git+https://github.com/myshell-ai/OpenVoice.git@main"


def install_voice_deps() -> bool:
    """Two-step install: clean [voice] extra, then OpenVoice with --no-deps.

    Returns True if both steps succeed.
    """
    base_cmd = [sys.executable, "-m", "pip", "install"]
    extra = "swap-cli[voice]"
    if (Path.cwd() / "pyproject.toml").exists():
        extra = ".[voice]"
        base_cmd.extend(["-e"])

    try:
        # Step 1: well-behaved deps from our [voice] extra.
        subprocess.check_call(base_cmd + [extra])
        # Step 2: OpenVoice without its bad transitive pins.
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", OPENVOICE_GIT_URL]
        )
        return True
    except subprocess.CalledProcessError:
        return False


def download_openvoice_weights(progress: callable | None = None) -> Path:  # type: ignore[valid-type]
    """Download OpenVoice v2 tone-color converter weights via HF Hub (~300 MB).

    Pulls only `converter/*` from the OpenVoiceV2 repo — base TTS speakers
    and BERT trees are unused by tone-color conversion. Writes a sentinel
    `installed.ok` once complete so the prereq check passes.
    """
    target = openvoice_weights_dir()
    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as err:
        raise RuntimeError(
            "huggingface-hub not installed. Run `swap voices install` first."
        ) from err

    if progress:
        progress(0.0)

    snapshot_download(
        repo_id=OPENVOICE_HF_REPO,
        local_dir=str(target),
        allow_patterns=OPENVOICE_INCLUDE_PATTERNS,
        local_dir_use_symlinks=False,
    )

    sentinel = target / "installed.ok"
    sentinel.write_text(
        f"repo={OPENVOICE_HF_REPO}\ndownloaded_at={int(time.time())}\n",
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

    # Real OpenVoice tone-color extraction (13b.2). Runs on CPU in ~5 s.
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
