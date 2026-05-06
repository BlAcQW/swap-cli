"""Maintainer tool: download HuggingFace RVC voice models, hash them,
print catalog entries ready for paste into rvc_catalog.py.

Usage:
    HF_TOKEN=hf_xxx python3 scripts/mirror_voices.py \\
        --slug=cool-voice --hf-repo=user/cool_voice_RVC

Outputs:
  1. Files at ./mirror_out/<slug>.pth and <slug>.index ready for
     `gh release upload voices-vN`.
  2. A copy-pasteable CatalogEntry(...) Python block.

This script is NOT shipped in the wheel — it lives at repo root.
HF token is read from env, NEVER stored in the script or committed.

After running:
    gh release upload voices-v1 mirror_out/cool-voice.pth mirror_out/cool-voice.index
    # ... then paste the CatalogEntry into src/swap_cli/rvc_catalog.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


def fetch(url: str, dest: Path, token: str | None) -> None:
    """Stream a URL to a file. Authenticated only if token is set."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    print(f"  → {url}")
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("content-length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = read / total * 100
                print(f"    {read / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)", end="\r")
    print()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True, help="Catalog slug (e.g. cool-voice)")
    ap.add_argument("--hf-repo", required=True, help="HF repo, e.g. user/repo_name")
    ap.add_argument(
        "--name", default=None, help="Display name (defaults to slug, title-cased)"
    )
    ap.add_argument("--description", default="", help="One-line description")
    ap.add_argument(
        "--pth-filename", default="model.pth", help="Filename of the .pth in the HF repo"
    )
    ap.add_argument(
        "--index-filename",
        default="model.index",
        help="Filename of the .index in the HF repo (or 'none' to skip)",
    )
    ap.add_argument(
        "--out-dir", default="mirror_out", help="Where to save mirrored files"
    )
    ap.add_argument(
        "--release-base",
        default="https://github.com/BlAcQW/swap-cli/releases/download/voices-v1",
        help="GH release base URL — bumps with each catalog version",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("warning: HF_TOKEN not set — public repos still work, but rate-limited", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_dest = out_dir / f"{args.slug}.pth"
    pth_url = f"https://huggingface.co/{args.hf_repo}/resolve/main/{args.pth_filename}"
    print(f"[1/2] downloading .pth")
    fetch(pth_url, pth_dest, token)
    pth_sha = sha256_of(pth_dest)
    pth_mb = pth_dest.stat().st_size // (1 << 20)
    print(f"      sha256: {pth_sha}")
    print(f"      size:   {pth_mb} MB")

    index_dest: Path | None = None
    index_sha: str | None = None
    index_mb = 0
    if args.index_filename and args.index_filename.lower() != "none":
        index_dest = out_dir / f"{args.slug}.index"
        index_url = f"https://huggingface.co/{args.hf_repo}/resolve/main/{args.index_filename}"
        print(f"[2/2] downloading .index")
        fetch(index_url, index_dest, token)
        index_sha = sha256_of(index_dest)
        index_mb = index_dest.stat().st_size // (1 << 20)
        print(f"      sha256: {index_sha}")
        print(f"      size:   {index_mb} MB")

    name = args.name or args.slug.replace("-", " ").title()

    print()
    print("=" * 60)
    print("Files ready in:", out_dir.resolve())
    print()
    print(f"  gh release upload voices-v1 \\")
    print(f"    {pth_dest}{' \\\\' if index_dest else ''}")
    if index_dest:
        print(f"    {index_dest}")
    print()
    print("=" * 60)
    print("Then paste this into src/swap_cli/rvc_catalog.py CATALOG tuple:")
    print()
    print(f"    CatalogEntry(")
    print(f"        slug={args.slug!r},")
    print(f"        name={name!r},")
    print(f"        description={args.description!r},")
    print(f"        pth_url=f\"{{GH_RELEASE_BASE}}/{args.slug}.pth\",")
    print(f"        pth_sha256={pth_sha!r},")
    print(f"        pth_size_mb={pth_mb},")
    if index_dest is not None:
        print(f"        index_url=f\"{{GH_RELEASE_BASE}}/{args.slug}.index\",")
        print(f"        index_sha256={index_sha!r},")
        print(f"        index_size_mb={index_mb},")
    else:
        print(f"        index_url=None,")
        print(f"        index_sha256=None,")
        print(f"        index_size_mb=0,")
    print(f"        original_source={f'huggingface.co/{args.hf_repo}'!r},")
    print(f"    ),")
    return 0


if __name__ == "__main__":
    sys.exit(main())
