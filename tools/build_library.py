#!/usr/bin/env python
"""Build the bundled voice library from tools/personas.yaml.

Runs offline on a machine that has:
  - the [voice] extra installed (`pip install -e .[voice]`)
  - OpenVoice converter weights downloaded (`swap voices install`)
  - coqui-tts (XTTS-v2) — synth path; install separately:
      `pip install coqui-tts`

For each persona in personas.yaml:
  1. XTTS-v2 synthesizes ~30 s of `synthesis_script` with the persona's
     built-in speaker latent.
  2. OpenVoice's tone-color converter extracts a 256-d speaker embedding
     from the synthesized sample.
  3. Saves <id>.json (embedding) and <id>_preview.wav (5 s preview) into
     src/swap_cli/voices/library/.

CPU is fine — XTTS synthesis is the slow step (~10 s per voice). On GPU
the whole library builds in ~1 minute.

Re-running is idempotent: voices already present are skipped unless you
pass --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "src" / "swap_cli" / "voices" / "library"
PERSONAS_YAML = ROOT / "tools" / "personas.yaml"
PREVIEW_SECONDS = 5.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild voices even if their JSON already exists.",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Skip XTTS+OpenVoice and write deterministic placeholders. "
        "Used for testing the surface without a working voice install.",
    )
    args = parser.parse_args()

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        print("pyyaml not installed. `pip install pyyaml`", file=sys.stderr)
        return 1

    if not PERSONAS_YAML.exists():
        print(f"missing: {PERSONAS_YAML}", file=sys.stderr)
        return 1

    config = yaml.safe_load(PERSONAS_YAML.read_text(encoding="utf-8"))
    personas = config["voices"]
    script = config["synthesis_script"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.placeholder:
        return _build_placeholders(personas)

    return _build_real(personas, script, args.force)


# ── Real synthesis path ───────────────────────────────────────────────────


def _build_real(personas: list[dict], script: str, force: bool) -> int:
    """XTTS-v2 → OpenVoice extraction. Requires GPU-friendly hardware."""

    try:
        from TTS.api import TTS  # type: ignore[import-not-found]
    except ImportError:
        print(
            "coqui-tts not installed. `pip install coqui-tts` first.",
            file=sys.stderr,
        )
        return 1

    try:
        from swap_cli.voice_model import extract_embedding, select_device
    except ImportError as err:
        print(f"swap_cli voice deps not installed: {err}", file=sys.stderr)
        return 1

    import soundfile as sf

    device = select_device()
    print(f"[build] device: {device}")
    print(f"[build] loading XTTS-v2 …")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    sample_rate = int(getattr(tts.synthesizer.output_sample_rate, "item", lambda: 24_000)())

    print(f"[build] building {len(personas)} voices into {OUTPUT_DIR}")
    for persona in personas:
        out_path = OUTPUT_DIR / f"{persona['id']}.json"
        preview_path = OUTPUT_DIR / f"{persona['id']}_preview.wav"

        if out_path.exists() and not force:
            print(f"  · {persona['id']:20s}  (cached, skip)")
            continue

        # 1) Synthesize ~30s with the persona's XTTS speaker.
        synth_path = OUTPUT_DIR / f".tmp_{persona['id']}.wav"
        try:
            tts.tts_to_file(
                text=script,
                speaker=persona["xtts_speaker"],
                language="en",
                file_path=str(synth_path),
            )
        except Exception as err:
            print(f"  ✗ {persona['id']}: synth failed — {err}")
            continue

        # 2) Extract OpenVoice tone-color embedding.
        try:
            embedding = extract_embedding(synth_path, device=device)
        except Exception as err:
            print(f"  ✗ {persona['id']}: embedding failed — {err}")
            synth_path.unlink(missing_ok=True)
            continue

        # 3) Save embedding JSON.
        out_path.write_text(
            json.dumps(
                {
                    "id": persona["id"],
                    "name": persona["name"],
                    "description": persona["description"],
                    "source": "library",
                    "embedding": embedding,
                    "sample_rate": 16_000,
                    "created_at": int(time.time()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # 4) Save 5s preview WAV (truncate the synth sample).
        try:
            audio, sr = sf.read(str(synth_path))
            preview_samples = int(PREVIEW_SECONDS * sr)
            sf.write(
                str(preview_path),
                audio[:preview_samples],
                sr,
                subtype="PCM_16",
            )
        except Exception as err:
            print(f"  ⚠ {persona['id']}: preview write failed — {err}")

        synth_path.unlink(missing_ok=True)

        print(
            f"  ✓ {persona['id']:20s}  {persona['name']}  "
            f"({persona['description']})"
        )

    print(f"\n→ Wrote {len(personas)} embeddings + previews to {OUTPUT_DIR}.")
    print("  Commit src/swap_cli/voices/library/*.{json,wav} with your release.")
    return 0


# ── Placeholder path (no GPU / no model — for surface testing) ────────────


def _build_placeholders(personas: list[dict]) -> int:
    import hashlib

    for persona in personas:
        out_path = OUTPUT_DIR / f"{persona['id']}.json"
        digest = hashlib.sha256(persona["id"].encode("utf-8")).digest()
        embedding = [((b - 128) / 128.0) for _ in range(8) for b in digest]
        out_path.write_text(
            json.dumps(
                {
                    "id": persona["id"],
                    "name": persona["name"],
                    "description": (
                        persona["description"]
                        + " (placeholder · regenerate with `python tools/build_library.py`)"
                    ),
                    "source": "library",
                    "embedding": embedding,
                    "sample_rate": 16_000,
                    "created_at": int(time.time()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  · {persona['id']:20s}  {persona['name']}  (placeholder)")
    print(f"\n→ Wrote {len(personas)} placeholder embeddings to {OUTPUT_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
