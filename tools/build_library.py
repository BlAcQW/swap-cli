#!/usr/bin/env python
"""Build the bundled voice library from tools/personas.yaml.

Run this on any machine that has the voice deps installed
(`pip install -e .[voice]`) plus a working OpenVoice v2 checkpoint and
XTTS-v2. CPU is fine — the limiting factor is XTTS synthesis speed,
~5–10s per voice on CPU, ~1s per voice on GPU.

Outputs to src/swap_cli/voices/library/:
  - <id>.json — the OpenVoice tone-color embedding
  - <id>_preview.wav — 5s sample for the GUI ▶ Preview button

Re-running is idempotent. Add a new persona to personas.yaml, rerun this
script, commit the resulting JSON + WAV, and ship a new version.

Sprint 13b.1 STATUS: this file is complete but cannot run yet because
voice_model.extract_embedding() is still a placeholder. 13b.2 wires the
real OpenVoice extraction; this script then becomes runnable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add src/ to the path so we can import swap_cli without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "src" / "swap_cli" / "voices" / "library"
PERSONAS_YAML = ROOT / "tools" / "personas.yaml"


def main() -> int:
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

    print(f"building {len(personas)} voices into {OUTPUT_DIR}")
    for persona in personas:
        out_path = OUTPUT_DIR / f"{persona['id']}.json"
        preview_path = OUTPUT_DIR / f"{persona['id']}_preview.wav"

        # 13b.2: synthesize a sample with XTTS-v2, then run through OpenVoice
        # to extract the embedding. 13b.1 just writes the placeholder so the
        # discovery surface works.
        embedding = _placeholder_embedding(persona["id"])

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

        # 13b.2: preview WAV — synthesize 5s of `script` with the persona's
        # XTTS speaker, save as 16kHz mono.
        if not preview_path.exists():
            preview_path.write_bytes(b"")  # 0-byte stub so the loader knows the file exists

        print(f"  ✓ {persona['id']:20s}  {persona['name']}  ({persona['description']})")

    print(f"\n→ Wrote {len(personas)} embeddings + {len(personas)} preview stubs.")
    print("  Commit src/swap_cli/voices/library/*.json with your release.")
    return 0


def _placeholder_embedding(seed: str) -> list[float]:
    """Deterministic 256-d vector keyed on the persona id."""
    import hashlib

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return [((b - 128) / 128.0) for _ in range(8) for b in digest]


if __name__ == "__main__":
    raise SystemExit(main())
