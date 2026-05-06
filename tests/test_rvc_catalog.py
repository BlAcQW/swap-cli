"""Tests for the RVC voice catalog (Sprint 14g).

Verifies:
  - The starter slug exists in the catalog (invariant the install
    command depends on).
  - Every entry has the required shape (URLs, SHA256 hashes, sizes).
  - All catalog URLs point at our own GitHub release — never HuggingFace
    or other third-party. Catches accidental drift in PRs.
  - find() returns the right entry or None.

Pure Python — no network. The actual download path is exercised by
manual end-to-end on the user's machine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swap_cli import rvc_catalog  # noqa: E402


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_catalog_has_starter() -> None:
    """STARTER_SLUG must exist in CATALOG — the install starter prompt
    relies on rvc_catalog.starter() returning a real entry."""
    assert rvc_catalog.STARTER_SLUG in {e.slug for e in rvc_catalog.CATALOG}
    assert rvc_catalog.starter().slug == rvc_catalog.STARTER_SLUG


def test_catalog_entries_complete() -> None:
    """Every entry has all required fields, valid SHA256 (64 hex)."""
    assert len(rvc_catalog.CATALOG) >= 1
    for entry in rvc_catalog.CATALOG:
        assert entry.slug
        assert entry.name
        assert entry.description
        assert entry.pth_url
        assert SHA256_RE.match(entry.pth_sha256), f"bad pth_sha256 for {entry.slug}"
        assert entry.pth_size_mb > 0
        # Index is optional but if URL is set, hash must be too.
        if entry.index_url is not None:
            assert entry.index_sha256 is not None
            assert SHA256_RE.match(entry.index_sha256), f"bad index_sha256 for {entry.slug}"
            assert entry.index_size_mb > 0
        assert entry.original_source  # transparency: where did we get this?


def test_catalog_entries_unique_slugs() -> None:
    """Slugs must be unique — they're the user-facing identifier."""
    slugs = [e.slug for e in rvc_catalog.CATALOG]
    assert len(slugs) == len(set(slugs))


def test_catalog_urls_use_our_github_release() -> None:
    """Every catalog URL must point at github.com/BlAcQW/swap-cli/releases.

    This is a regression guard: if a maintainer adds a new entry and
    accidentally pastes the original HF URL, this test fails the PR.
    HF URLs are documented in `original_source` for transparency, never
    used at runtime.
    """
    expected_prefix = "https://github.com/BlAcQW/swap-cli/releases/download/"
    for entry in rvc_catalog.CATALOG:
        assert entry.pth_url.startswith(expected_prefix), (
            f"{entry.slug}: pth_url must use our GH release, got {entry.pth_url}"
        )
        if entry.index_url is not None:
            assert entry.index_url.startswith(expected_prefix), (
                f"{entry.slug}: index_url must use our GH release, got {entry.index_url}"
            )


def test_catalog_find_returns_entry_or_none() -> None:
    starter = rvc_catalog.find(rvc_catalog.STARTER_SLUG)
    assert starter is not None
    assert starter.slug == rvc_catalog.STARTER_SLUG

    assert rvc_catalog.find("definitely-not-a-real-slug") is None


def test_catalog_total_size_includes_index() -> None:
    """total_size_mb is what we show users in the prompt."""
    for entry in rvc_catalog.CATALOG:
        if entry.index_url is None:
            assert entry.total_size_mb == entry.pth_size_mb
        else:
            assert entry.total_size_mb == entry.pth_size_mb + entry.index_size_mb


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
