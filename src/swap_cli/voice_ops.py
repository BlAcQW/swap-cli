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


# OpenVoice's setup.py pins librosa==0.9.1, numpy==1.22, av==10.* (av 10 has
# no Win+Py3.11 wheel; source builds break on modern Cython). We install it
# with --no-deps so its broken constraints don't poison the resolver. The
# tone-color converter at runtime only needs torch + numpy + librosa +
# soundfile, which we install directly below.
OPENVOICE_GIT_URL = "git+https://github.com/myshell-ai/OpenVoice.git@main"

# Voice-cloning dependencies we install directly (NOT via `pip install
# '.[voice]'` because that would also try to reinstall swap-cli, which on
# Windows fails with "process cannot access the file" — Windows holds
# swap.exe open while the install runs from inside it).
VOICE_PACKAGES = (
    "torch>=2.2",
    "torchaudio>=2.2",
    "sounddevice>=0.4",
    "librosa>=0.10",
    "soundfile>=0.12",
    "huggingface-hub>=0.20",
    # OpenVoice's package __init__ eagerly imports its full runtime deps
    # even for tone-color-only conversion. Install the lot here so future
    # installs don't trickle errors one missing module at a time.
    #
    # We use NEWER versions than OpenVoice's pyproject.toml pins where:
    #   - older pin has no Win/Py3.11 wheel (faster-whisper 0.9 → av==10)
    #   - newer version is API-compatible enough for tone-color path
    "inflect>=7.0",
    "unidecode>=1.3",
    "eng-to-ipa>=0.0.2",
    "pypinyin>=0.50",
    "cn2an>=0.5",
    "jieba>=0.42",
    "langid>=1.1",
    "pydub>=0.25",
    "wavmark>=0.0.3",
    "faster-whisper>=1.0",
    "whisper-timestamped>=1.14",
)


def install_voice_deps() -> bool:
    """Install voice deps without touching swap-cli itself.

    Two pip calls:
      1. install VOICE_PACKAGES directly — modern resolved versions, no
         swap-cli reinstall (avoids the Windows swap.exe lock).
      2. install OpenVoice with --no-deps — keeps its broken transitive
         pins (faster-whisper → av==10.*, etc.) out of the resolver.

    Returns True iff both steps succeed.
    """
    pip = [sys.executable, "-m", "pip", "install"]
    try:
        subprocess.check_call(pip + list(VOICE_PACKAGES))
        subprocess.check_call(pip + ["--no-deps", OPENVOICE_GIT_URL])
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
