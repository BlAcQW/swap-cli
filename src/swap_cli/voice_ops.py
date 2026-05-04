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


# OpenVoice v2 lives in MyShell's HuggingFace repo. We need:
#   - converter/   the tone-color converter weights (~300 MB)
#   - base_speakers/ses/*.pth   real speaker embeddings shipped with the
#                               release (English/Spanish/French/etc.).
#                               Tiny (~4 KB each) and gives users
#                               actual-cloning library voices out of the box.
OPENVOICE_HF_REPO = "myshell-ai/OpenVoiceV2"
OPENVOICE_INCLUDE_PATTERNS = ["converter/*", "base_speakers/ses/*.pth"]


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
    # gradio is OpenVoice's web demo UI — not used by tone-color but
    # imported eagerly at package load. ~40 MB transitive deps.
    "gradio>=3.48",
    # openai + python-dotenv are in OpenVoice's requirements.txt (not
    # setup.py) for their demo notebooks. Included for completeness so
    # 'import openvoice' never trips on a missing module.
    "openai",
    "python-dotenv",
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


# ── Standalone voice session runner (shared by CLI + GUI) ─────────────────


def find_voice_by_name_or_id(name_or_id: str) -> Voice | None:
    """Resolve a voice by id, slug, or display name across library + user."""
    from .voice_library import find_voice, load_all_voices

    direct = find_voice(name_or_id)
    if direct is not None:
        return direct
    needle = name_or_id.lower().strip()
    for v in load_all_voices():
        if v.name.lower() == needle or v.id.lower() == needle:
            return v
    return None


def resolve_voice_devices(
    mic: int | None = None,
    output: int | None = None,
) -> tuple[int, int | None]:
    """Pick mic + virtual cable indices, falling back to auto-detect.

    Returns (mic_index, output_index_or_None). Output is None when no
    virtual audio cable is detected — voice still runs but is silent.
    """
    from . import voice_router

    mic_idx = mic
    if mic_idx is None:
        picked = voice_router.pick_input_device(None)
        mic_idx = int(picked["index"]) if picked else 0
    out_idx = output
    if out_idx is None:
        picked = voice_router.pick_output_device(None)
        out_idx = int(picked["index"]) if picked else None
    return mic_idx, out_idx


# ── RVC voice management (Sprint 14b.2.b) ──────────────────────────────────


def add_rvc_voice(
    pth_path: Path,
    name: str | None = None,
    index_path: Path | None = None,
) -> Voice:
    """Register an RVC voice model with swap-cli.

    Copies the .pth (and optional .index) into the per-voice subdirectory
    inside the RVC models dir, then writes a Voice record so the
    GUI/CLI dropdown surfaces it. The .index file is a Faiss retrieval
    index that improves quality — recommended but optional.

    The Voice record uses:
      - id: 'rvc-<slug>'  (the 'rvc-' prefix is how RVCEngine recognises
        it during make_converter)
      - source: 'library'  (shipped + user-added live in the same list)
      - embedding: [] (RVC's voice identity is the .pth file, not a
        speaker SE; the embedding field stays empty for RVC voices and
        the engine reads voice.id to find the model)
    """
    from .voice_prereq import rvc_models_dir

    if not pth_path.exists():
        raise FileNotFoundError(f"RVC model not found: {pth_path}")
    if not pth_path.suffix == ".pth":
        raise ValueError(f"Expected .pth model file, got: {pth_path.suffix}")

    display_name = name or pth_path.stem.replace("_", " ").replace("-", " ").strip()
    voice_id = f"rvc-{slugify(display_name)}"

    # Copy the .pth (and .index if given) into rvc_models_dir/<voice_id>/.
    target_dir = rvc_models_dir() / voice_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_pth = target_dir / pth_path.name
    if not target_pth.exists():
        target_pth.write_bytes(pth_path.read_bytes())
    if index_path is not None:
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        target_index = target_dir / index_path.name
        if not target_index.exists():
            target_index.write_bytes(index_path.read_bytes())

    voice = Voice(
        id=voice_id,
        name=display_name,
        description=f"RVC voice from {pth_path.name}",
        source="library",
        embedding=[],  # RVC uses the .pth file directly, not an SE vector
        sample_rate=16_000,
        created_at=int(time.time()),
    )
    save_user_voice(voice)
    return voice


def remove_rvc_voice(name_or_id: str) -> bool:
    """Remove an RVC voice — both the JSON record and the model files."""
    import shutil

    from .voice_prereq import rvc_models_dir

    target_id = (
        name_or_id if name_or_id.startswith("rvc-") else f"rvc-{slugify(name_or_id)}"
    )
    voice_path = user_voices_dir() / f"{target_id}.json"
    model_dir = rvc_models_dir() / target_id

    removed_any = False
    if voice_path.exists():
        voice_path.unlink()
        removed_any = True
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
        removed_any = True
    return removed_any


def rvc_model_path_for(voice: Voice) -> Path | None:
    """Return the .pth file for an RVC voice, or None if not found."""
    from .voice_prereq import rvc_models_dir

    if not voice.id.startswith("rvc-"):
        return None
    model_dir = rvc_models_dir() / voice.id
    if not model_dir.exists():
        return None
    pth_files = list(model_dir.glob("*.pth"))
    return pth_files[0] if pth_files else None


def rvc_index_path_for(voice: Voice) -> Path | None:
    """Return the .index file for an RVC voice, or None if not present.

    The .index file enables retrieval-based feature mixing — improves
    quality but is optional.
    """
    from .voice_prereq import rvc_models_dir

    if not voice.id.startswith("rvc-"):
        return None
    model_dir = rvc_models_dir() / voice.id
    if not model_dir.exists():
        return None
    index_files = list(model_dir.glob("*.index"))
    return index_files[0] if index_files else None
