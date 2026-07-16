#!/usr/bin/env python3
"""Settle the two fal questions with a real API key. NOT shipped, NOT in CI.

Decart stamps the "✦ AI Generated" badge and refuses to lift it for our account.
fal.ai resells the SAME Lucy realtime model on fal's own Decart account, so its
watermark policy MAY differ. Everything else about switching to fal hinges on
two unknowns this script answers for ~pennies:

  1. WATERMARK — does fal's output carry the roaming badge?
  2. IDENTITY  — fal's only realtime Lucy endpoint is VTON/outfit-branded, but its
     schema documents `reference_image_url` as a *character reference*. Does it
     actually substitute the identity (what swap-cli needs), or only clothing?

Usage:
    set FAL_KEY=fal_sk_...            (Windows)   /   export FAL_KEY=...  (unix)
    python scripts/fal_probe.py --reference face.jpg --source video.mp4 --frames 40

Then LOOK at probe_out/ :
  * badge visible in any frame  -> fal is out; stop here.
  * face replaced by the reference -> identity swap works; the provider is viable.
  * only clothes changed          -> VTON only; fal can't do swap-cli's job.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from swap_cli.providers.fal import (  # noqa: E402
    FAL_REALTIME_APP,
    _decode_result,
    _encode_frame_jpeg,
    _encode_reference,
)

# swap-cli's real prompt — probe what the product actually asks Lucy to do.
IDENTITY_PROMPT = (
    "Substitute the person in the video with the person in the reference image. "
    "Match their face, body, and identity while preserving the original pose, "
    "motion, and lighting."
)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="reference identity image")
    ap.add_argument("--source", help="source video (default: webcam 0)")
    ap.add_argument("--frames", type=int, default=40, help="frames to send (~$0.02/s)")
    ap.add_argument("--prompt", default=IDENTITY_PROMPT)
    ap.add_argument("--out", default="probe_out", help="where to dump frames")
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set — get a key at https://fal.ai/dashboard/keys")
        return 2

    try:
        import fal_client
    except ImportError:
        print("pip install fal-client")
        return 2

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source if args.source else 0)
    if not cap.isOpened():
        print(f"could not open source: {args.source or 'webcam 0'}")
        return 1

    reference = _encode_reference(args.reference)
    print(f"app={FAL_REALTIME_APP}\nprompt={args.prompt!r}\nsending {args.frames} frames…")

    sent = 0
    async with fal_client.realtime_async(FAL_REALTIME_APP) as conn:
        while sent < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            await conn.send(
                {
                    "prompt": args.prompt,
                    "image_url": _encode_frame_jpeg(frame),
                    "reference_image_url": reference,
                }
            )
            result = await conn.recv()
            if result is None:
                print("connection closed by fal")
                break
            if sent == 0:  # dump the raw reply once — reveals the payload shape
                keys = sorted(result)
                print(f"first reply keys: {keys}")
                (outdir / "first_reply_keys.txt").write_text(repr(keys))
            try:
                out = _decode_result(result)
            except Exception as err:  # noqa: BLE001 — probe: show, don't crash
                print(f"decode failed: {err}")
                (outdir / "raw_reply.txt").write_text(repr(result)[:4000])
                break
            cv2.imwrite(str(outdir / f"in_{sent:03d}.png"), frame)
            cv2.imwrite(str(outdir / f"out_{sent:03d}.png"), out)
            sent += 1
            print(f"\r  {sent}/{args.frames}", end="", flush=True)
    cap.release()

    print(f"\n\nwrote {sent} in_/out_ pairs to {outdir}/")
    print("NOW LOOK AT THEM:")
    print("  1. WATERMARK — any '✦ AI Generated' badge in out_*.png? (it roams)")
    print("  2. IDENTITY  — is the FACE replaced by the reference (good),")
    print("     or only the clothing changed (VTON only → fal can't do our job)?")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
