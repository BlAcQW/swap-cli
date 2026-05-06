"""Implementations for the `swap voices` subcommand group.

Sprint 14e: OpenVoice removed. Voice path is RVC-only.
- `install_voice_deps` installs CUDA-matched PyTorch first (Win/Linux NVIDIA),
  then RVC's runtime deps + rvc-python + fairseq via --no-deps.
- `add_rvc_voice` / `remove_rvc_voice` register RVC .pth models.
- No more OpenVoice tone-color extraction; users either download community
  RVC models (e.g. weights.gg, lj1995/VoiceConversionWebUI on HF) or
  train their own with Applio.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .rvc_catalog import CatalogEntry
from .voice_library import (
    Voice,
    delete_user_voice,
    load_user_voices,
    save_user_voice,
    slugify,
    user_voices_dir,
)
from .voice_prereq import models_dir, rvc_models_dir


# ── Install: pip deps for the RVC streaming engine ────────────────────────


@dataclass(frozen=True)
class InstallResult:
    deps_installed: bool
    error: str | None = None


# RVC's runtime dep stack at modern resolved versions. rvc-python itself
# pins fairseq==0.12.2 + numpy<=1.23.5 — both unresolvable on modern
# stacks — so we install rvc-python with --no-deps after seeding these.
VOICE_PACKAGES = (
    "torch>=2.2",
    "torchaudio>=2.2",
    "sounddevice>=0.4",
    "librosa>=0.10",
    "soundfile>=0.12",
    "huggingface-hub>=0.20",
    # rvc-python's actual runtime deps:
    "praat-parselmouth>=0.4.3",
    "faiss-cpu>=1.8",
    "pyworld>=0.3.4",
    "torchcrepe>=0.0.20",
    "omegaconf>=2.3",
    "ffmpeg-python>=0.2",
    "loguru>=0.7",
    # Sprint 14g.3: fairseq runtime deps (we install fairseq itself with
    # --no-deps because of its archived omegaconf<2.1 / hydra-core<1.1
    # pins; modern resolved versions of the same packages work fine for
    # the basic OmegaConf.merge / hydra compose API fairseq uses).
    "hydra-core>=1.3",
    "bitarray>=2.0",
    "regex>=2023.0",
    "sacrebleu>=2.0",
    "scikit-learn>=1.0",
    "cffi>=1.15",
    "Cython>=0.29",
    "packaging>=21.0",
)

# Subset of VOICE_PACKAGES that fairseq specifically needs at runtime.
# `swap voices repair` reinstalls these idempotently so users on older
# installs don't have to re-run the full voice install.
FAIRSEQ_RUNTIME_DEPS = (
    "hydra-core>=1.3",
    "bitarray>=2.0",
    "regex>=2023.0",
    "sacrebleu>=2.0",
    "scikit-learn>=1.0",
    "cffi>=1.15",
    "Cython>=0.29",
    "packaging>=21.0",
    "omegaconf>=2.3",
)

# rvc-python on PyPI peaks at 0.1.5 and pins fairseq + numpy at
# unresolvable versions. Install with --no-deps after seeding runtime deps.
RVC_PYTHON_SPEC = "rvc-python>=0.1.5"

# fairseq 0.12.2 has no wheels on py3.11+, source build is broken
# (omegaconf<2.1 + hydra-core resolver loop, dataclass mutable default
# on py3.12). Install from the archived repo's main branch.
FAIRSEQ_GIT_URL = "git+https://github.com/facebookresearch/fairseq.git"

# CUDA torch index for Windows/Linux NVIDIA. RVC inference quality
# depends on running on the GPU — without --index-url, pip's resolver
# often picks the CPU torch wheel and the user wonders why their 4070
# is idle. cu121 is a stable LTS window for RTX 30/40 drivers.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH_CUDA_PACKAGES = ("torch>=2.2", "torchaudio>=2.2")


def install_voice_deps() -> bool:
    """Install voice deps without touching swap-cli itself.

    Order matters:
      1. Install CUDA-matched torch first (Windows/Linux NVIDIA only)
         so subsequent `pip install` calls don't pull the CPU wheel.
      2. Install VOICE_PACKAGES — modern resolved versions, rvc-python's
         actual runtime deps (faiss, pyworld, omegaconf, etc.).
      3. Install rvc-python with --no-deps.
      4. Install fairseq from git with --no-deps — the PyPI 0.12.2 has
         no wheels on py3.11+; the archived repo's main branch has the
         fixes that didn't make it into the last release.

    Returns True iff every step succeeds.
    """
    # macOS: OpenVoice is gone, but rvc-python's transitive build chain
    # (pyworld, fairseq from git) still wants git + a C compiler — both
    # ship with Xcode CLT. Note: macOS isn't a supported voice platform
    # post-14e (Apple Silicon CPU is too slow for RVC live), but we keep
    # the preflight so CI / curious users get a clear message.
    if sys.platform == "darwin" and shutil.which("git") is None:
        raise RuntimeError(
            "git not found. Install Xcode Command Line Tools first:\n"
            "    xcode-select --install\n"
            "Then re-run `swap voices install`."
        )

    pip = [sys.executable, "-m", "pip", "install"]
    try:
        # CUDA torch first on NVIDIA platforms — keeps CPU torch from
        # winning the resolver when later steps install other torch deps.
        if _is_nvidia_platform():
            subprocess.check_call(
                pip
                + ["--index-url", TORCH_CUDA_INDEX]
                + list(TORCH_CUDA_PACKAGES)
            )

        subprocess.check_call(pip + list(VOICE_PACKAGES))
        subprocess.check_call(pip + ["--no-deps", RVC_PYTHON_SPEC])
        subprocess.check_call(pip + ["--no-deps", FAIRSEQ_GIT_URL])
        # Post-install: patch fairseq for Python 3.11+ dataclass strict mode.
        # Returns False on patch failure so the user sees the install as failed
        # rather than getting a runtime error later.
        if not patch_fairseq_dataclass_defaults():
            print(
                "[voice_ops] fairseq dataclass patch failed — "
                "voice will fail at runtime. See `swap doctor`.",
                flush=True,
            )
            return False
        return True
    except subprocess.CalledProcessError:
        return False


def _is_nvidia_platform() -> bool:
    """True iff we're on Windows or Linux with `nvidia-smi` reachable.
    macOS never has CUDA; Linux without NVIDIA falls through to the
    plain pypi torch (which is fine — CPU)."""
    if sys.platform == "darwin":
        return False
    return shutil.which("nvidia-smi") is not None


def install_fairseq_runtime_deps() -> bool:
    """Idempotently pip-install fairseq's actual runtime deps.

    Used by `swap voices repair` to fix installs that ran before Sprint
    14g.3 (when these packages weren't seeded). Pip skips already-
    satisfied requirements, so re-running is cheap.
    """
    pip = [sys.executable, "-m", "pip", "install"]
    try:
        subprocess.check_call(pip + list(FAIRSEQ_RUNTIME_DEPS))
        return True
    except subprocess.CalledProcessError:
        return False


def patch_fairseq_dataclass_defaults() -> bool:
    """Patch fairseq's `dataclass/configs.py` for Python 3.11+ compat.

    Why: fairseq#5634 — Python 3.11+ rejects mutable dataclass defaults.
    The archived fairseq main branch defines `FairseqConfig` like:

        @dataclass
        class FairseqConfig(FairseqDataclass):
            common: CommonConfig = CommonConfig()       # ❌ rejected
            common_eval: CommonEvalConfig = CommonEvalConfig()
            ... (11 such lines)

    Running `import fairseq` on 3.11+ raises:
        ValueError: mutable default <class 'fairseq.dataclass.configs.CommonConfig'>
        for field common is not allowed: use default_factory

    Repo is archived (March 2026) — no upstream fix coming. We rewrite
    those 11 lines in-place to use `field(default_factory=...)`. The
    `field` import is already at the top of the file, so no other edits
    are needed.

    Idempotent: re-running detects the already-patched form and no-ops.
    Returns True if the patch was applied, was already applied, or
    fairseq isn't installed (nothing to do).
    """
    import re
    import site
    import sysconfig

    # CRITICAL: do NOT use importlib.find_spec("fairseq.*") here. find_spec
    # imports the parent fairseq package, whose __init__.py transitively
    # imports the very broken file we're trying to patch — meaning find_spec
    # crashes with the same ValueError before it can return. Locate
    # configs.py purely by filesystem walk instead.
    raw_paths: list[str] = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            raw_paths.append(paths[key])
    if hasattr(site, "getsitepackages"):
        try:
            raw_paths.extend(site.getsitepackages())
        except (AttributeError, TypeError):
            pass
    if hasattr(site, "getusersitepackages"):
        try:
            user_sp = site.getusersitepackages()
            if user_sp:
                raw_paths.append(user_sp)
        except (AttributeError, TypeError):
            pass

    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        sp = Path(raw)
        if sp in seen:
            continue
        seen.add(sp)
        configs_py = sp / "fairseq" / "dataclass" / "configs.py"
        if configs_py.is_file():
            candidates.append(configs_py)

    if not candidates:
        # fairseq isn't installed in any reachable site-packages.
        return True

    configs_path = candidates[0]
    try:
        src = configs_path.read_text(encoding="utf-8")
    except OSError as err:
        print(f"[voice_ops] cannot read {configs_path}: {err}", flush=True)
        return False

    # Pattern matches:  <whitespace><field>: <Type>Config = <Type>Config()
    # Captures the type name twice, asserts they match. Idempotent because
    # already-patched lines use `field(default_factory=...)` which won't
    # match `= <Type>()` exactly.
    pattern = re.compile(
        r"^(\s*\w+: ([A-Z]\w*Config)) = \2\(\)$",
        re.MULTILINE,
    )
    new_src, count = pattern.subn(
        lambda m: f"{m.group(1)} = field(default_factory={m.group(2)})",
        src,
    )
    if count == 0:
        # Already patched, or upstream changed shape. Either way, no-op.
        return True

    try:
        configs_path.write_text(new_src, encoding="utf-8")
    except OSError as err:
        print(f"[voice_ops] cannot write {configs_path}: {err}", flush=True)
        return False
    print(
        f"[voice_ops] patched {count} mutable defaults in fairseq configs.py "
        "(Python 3.11+ compat)",
        flush=True,
    )
    return True


# ── Library/user voice operations ─────────────────────────────────────────


def list_all() -> tuple[list[Voice], list[Voice]]:
    """Return (library_voices, user_voices) for display.

    Sprint 14e: library is intentionally empty — the bundled OpenVoice
    embeddings were removed. User voices are RVC voices added via
    `swap voices add-rvc`. Returning the tuple shape preserves the
    `voices list` UI: 'Library voices: empty', 'Your voices: …'.
    """
    return [], load_user_voices()


# ── Standalone voice session helpers ──────────────────────────────────────


def find_voice_by_name_or_id(name_or_id: str) -> Voice | None:
    """Resolve a voice by id, slug, or display name across the user library."""
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


# ── RVC voice management ──────────────────────────────────────────────────


def add_rvc_voice(
    pth_path: Path,
    name: str | None = None,
    index_path: Path | None = None,
) -> Voice:
    """Register an RVC voice model with swap-cli.

    Copies the .pth (and optional .index) into the per-voice subdirectory
    inside the RVC models dir, then writes a Voice record so the
    GUI/CLI dropdown surfaces it. The .index file enables retrieval-
    based feature mixing — recommended quality boost when present.

    Voice record convention:
      - id: 'rvc-<slug>'  (the 'rvc-' prefix is how RVCEngine recognises
        it during make_converter)
      - source: 'library'  (shipped + user-added live in the same list)
      - embedding: [] (RVC's voice identity is the .pth file, not a
        speaker SE; the engine reads voice.id to find the model)
    """
    if not pth_path.exists():
        raise FileNotFoundError(f"RVC model not found: {pth_path}")
    if not pth_path.suffix == ".pth":
        raise ValueError(f"Expected .pth model file, got: {pth_path.suffix}")

    display_name = name or pth_path.stem.replace("_", " ").replace("-", " ").strip()
    voice_id = f"rvc-{slugify(display_name)}"

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
        embedding=[],
        sample_rate=16_000,
        created_at=int(time.time()),
    )
    save_user_voice(voice)
    return voice


def remove_rvc_voice(name_or_id: str) -> bool:
    """Remove an RVC voice — both the JSON record and the model files."""
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
    if not voice.id.startswith("rvc-"):
        return None
    model_dir = rvc_models_dir() / voice.id
    if not model_dir.exists():
        return None
    index_files = list(model_dir.glob("*.index"))
    return index_files[0] if index_files else None


# ── Catalog download (Sprint 14g) ─────────────────────────────────────────


def download_catalog_voice(
    entry: CatalogEntry,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Voice:
    """Download a catalog voice from our GH release, verify SHA256, register.

    on_progress(filename, bytes_so_far, total_bytes) fires periodically
    while downloading. Pass None to skip progress reporting.

    Reuses add_rvc_voice() for the final copy + Voice JSON write so the
    on-disk layout is identical whether the user came in via add-rvc or
    download <slug>.

    Raises:
        RuntimeError: if SHA256 doesn't match (download corrupted or the
            catalog entry is wrong — both worth surfacing loudly).
        httpx.HTTPError: on network/HTTP failures (caller wraps for UX).
    """
    import httpx  # base dep; importing inline keeps it out of the module load path

    with tempfile.TemporaryDirectory(prefix=f"swap-catalog-{entry.slug}-") as tmp:
        tmp_dir = Path(tmp)
        pth_path = _stream_with_sha(
            url=entry.pth_url,
            dest=tmp_dir / f"{entry.slug}.pth",
            expected_sha256=entry.pth_sha256,
            on_progress=on_progress,
        )
        index_path: Path | None = None
        if entry.index_url is not None and entry.index_sha256 is not None:
            index_path = _stream_with_sha(
                url=entry.index_url,
                dest=tmp_dir / f"{entry.slug}.index",
                expected_sha256=entry.index_sha256,
                on_progress=on_progress,
            )

        # Hand off to the existing add-rvc path; copies into the user
        # data dir and writes the Voice JSON.
        return add_rvc_voice(pth_path, name=entry.name, index_path=index_path)


def _stream_with_sha(
    url: str,
    dest: Path,
    expected_sha256: str,
    on_progress: Callable[[str, int, int], None] | None,
) -> Path:
    """Stream a URL to dest, computing SHA256 as we go. Raises on mismatch."""
    import httpx

    digest = hashlib.sha256()
    bytes_read = 0
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):  # 1 MB
                fh.write(chunk)
                digest.update(chunk)
                bytes_read += len(chunk)
                if on_progress is not None:
                    on_progress(dest.name, bytes_read, total)

    actual = digest.hexdigest()
    if actual != expected_sha256:
        # Wipe the bad file so a retry starts clean.
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {dest.name}: "
            f"expected {expected_sha256[:12]}…, got {actual[:12]}…. "
            "Download corrupted; re-run `swap voices download` to retry."
        )
    return dest
