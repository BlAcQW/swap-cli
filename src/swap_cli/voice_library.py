"""Voice library: bundled presets + user-added voices.

Each voice is stored as a small JSON file containing:
  - id: stable slug
  - name: display name
  - description: shown in the dropdown
  - source: 'library' | 'user'
  - embedding: list[float] — OpenVoice tone-color embedding (256 dims)
  - sample_rate: int — sample rate the embedding was extracted at
  - created_at: unix seconds

Bundled voices live inside the package at swap_cli/voices/library/*.json
(read-only). User voices live in ~/.swap/voices/*.json (read/write).
"""

from __future__ import annotations

import importlib.resources
import json
import time
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "swap-cli"


def user_voices_dir() -> Path:
    """Where custom user voices live. Created on first save."""
    return Path(user_data_dir(APP_NAME)) / "voices"


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    description: str
    source: str  # 'library' | 'user'
    embedding: list[float]
    sample_rate: int
    created_at: int

    @property
    def is_library(self) -> bool:
        return self.source == "library"

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "source": self.source,
                "embedding": self.embedding,
                "sample_rate": self.sample_rate,
                "created_at": self.created_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> Voice:
        d = json.loads(raw)
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            description=str(d.get("description", "")),
            source=str(d.get("source", "user")),
            embedding=[float(x) for x in d["embedding"]],
            sample_rate=int(d.get("sample_rate", 16000)),
            created_at=int(d.get("created_at", time.time())),
        )


def load_library_voices() -> list[Voice]:
    """Return voices bundled with the package."""
    voices: list[Voice] = []
    try:
        library = importlib.resources.files("swap_cli.voices.library")
    except (ModuleNotFoundError, FileNotFoundError, ImportError):
        return voices

    for entry in library.iterdir():
        if not entry.name.endswith(".json"):
            continue
        try:
            voices.append(Voice.from_json(entry.read_text(encoding="utf-8")))
        except Exception as err:  # noqa: BLE001
            print(f"[voice_library] skipping {entry.name}: {err}", flush=True)
    voices.sort(key=lambda v: v.name.lower())
    return voices


def load_user_voices() -> list[Voice]:
    """Return voices the user added via `swap voices add`."""
    voices: list[Voice] = []
    base = user_voices_dir()
    if not base.exists():
        return voices

    for entry in base.glob("*.json"):
        try:
            voices.append(Voice.from_json(entry.read_text(encoding="utf-8")))
        except Exception as err:  # noqa: BLE001
            print(f"[voice_library] skipping {entry.name}: {err}", flush=True)
    voices.sort(key=lambda v: v.name.lower())
    return voices


def load_all_voices() -> list[Voice]:
    """Return library voices first, then user voices."""
    return load_library_voices() + load_user_voices()


def find_voice(voice_id: str) -> Voice | None:
    """Look up a voice by id across both sources."""
    for v in load_all_voices():
        if v.id == voice_id:
            return v
    return None


def save_user_voice(voice: Voice) -> Path:
    """Persist a user-added voice to ~/.swap/voices/<id>.json."""
    base = user_voices_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{voice.id}.json"
    path.write_text(voice.to_json(), encoding="utf-8")
    return path


def delete_user_voice(voice_id: str) -> bool:
    """Remove a user voice. Returns True if the file existed."""
    path = user_voices_dir() / f"{voice_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def slugify(name: str) -> str:
    """Turn 'Morgan Voice!' into 'morgan-voice' for use as an id."""
    out = []
    last_dash = False
    for ch in name.lower().strip():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "voice"
