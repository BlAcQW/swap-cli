"""Curated RVC voice catalog — mirrored to our GitHub Releases.

Why we mirror to our own GH release instead of fetching from HuggingFace
at runtime: HF repos can disappear (account deletion, takedowns, repo
renames). Once we've vetted a voice for license + quality, we want a
stable URL we control. The `original_source` field on each entry
documents where we got the model — for transparency, not for the
runtime download path.

The runtime path: `voice_ops.download_catalog_voice(entry)` fetches
from `pth_url` / `index_url`, verifies SHA256 against the baked-in
hashes, and registers the voice via the existing `add_rvc_voice()`.

Maintainer note: to add a new voice, run `scripts/mirror_voices.py`
with HF_TOKEN set, upload the resulting files to a new GH release
(or extend voices-v1), and append a CatalogEntry here with the new
URL + SHA256.
"""

from __future__ import annotations

from dataclasses import dataclass


GH_RELEASE_BASE = "https://github.com/BlAcQW/swap-cli/releases/download/voices-v1"


@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    name: str
    description: str
    pth_url: str
    pth_sha256: str
    pth_size_mb: int
    index_url: str | None
    index_sha256: str | None
    index_size_mb: int
    original_source: str  # documents the upstream source for transparency

    @property
    def total_size_mb(self) -> int:
        return self.pth_size_mb + (self.index_size_mb if self.index_url else 0)


# Smallest total download → fastest "press Enter and try" experience.
STARTER_SLUG = "soft-asmr"


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        slug="soft-asmr",
        name="Soft ASMR (female)",
        description="Soft, breathy female ASMR voice. Smallest catalog entry — good for first try.",
        pth_url=f"{GH_RELEASE_BASE}/soft-asmr.pth",
        pth_sha256="0b88fd4ab585af6a0bb1046c954938dd1cd8a7a6cdfe7cfa40746badbd7a544f",
        pth_size_mb=54,
        index_url=f"{GH_RELEASE_BASE}/soft-asmr.index",
        index_sha256="cb929eb2fb6a2fe14cfedbdbb850fcd5ca609845523c057b25a44f607f89b76c",
        index_size_mb=55,
        original_source="huggingface.co/binant/soft_asmr_female",
    ),
    CatalogEntry(
        slug="calm-man",
        name="Calm man",
        description="Calm conversational male voice. Larger index → richer reference set.",
        pth_url=f"{GH_RELEASE_BASE}/calm-man.pth",
        pth_sha256="71eedb8a805ee771eca87c2a9a9443e2ff4559d2ebdeb87129cd7cb8193f2b1b",
        pth_size_mb=55,
        index_url=f"{GH_RELEASE_BASE}/calm-man.index",
        index_sha256="3528d5dccecb21ff288301d44c338cf970041a0f8985ca01f4f6a5b3dc639010",
        index_size_mb=579,
        original_source="huggingface.co/binant/calm_man-male",
    ),
    CatalogEntry(
        slug="egirl",
        name="E-girl streamer",
        description="Energetic feminine streamer-style voice.",
        pth_url=f"{GH_RELEASE_BASE}/egirl.pth",
        pth_sha256="fa099355523b09dcde45489a567f7b5e1f843df64bb6f532d2cb6c93606880a1",
        pth_size_mb=55,
        index_url=f"{GH_RELEASE_BASE}/egirl.index",
        index_sha256="29f2731238f14b6c509b9b7bccc75a21ae7414737c6c101c739fd95cc8f37839",
        index_size_mb=256,
        original_source="huggingface.co/binant/egirl_female",
    ),
)


def find(slug: str) -> CatalogEntry | None:
    """Look up a catalog entry by slug. Returns None if unknown."""
    for entry in CATALOG:
        if entry.slug == slug:
            return entry
    return None


def starter() -> CatalogEntry:
    """The default starter — smallest total download for fastest first-run."""
    entry = find(STARTER_SLUG)
    if entry is None:
        raise RuntimeError(
            f"STARTER_SLUG={STARTER_SLUG!r} not found in CATALOG — invariant broken"
        )
    return entry
