#!/usr/bin/env python3
"""Offline eval harness for watermark removal — NOT shipped, NOT in CI.

Runs WatermarkRemover.process() over a video and reports how often the badge
was covered (the real signal for "are we missing it"), the longest stretch the
badge stayed visible, and raw-confidence stats. Optionally writes a cleaned
video and/or an annotated video (detection box drawn) so the result can be
eyeballed.

    python scripts/wm_eval.py video.mp4
    python scripts/wm_eval.py video.mp4 --out video_clean.mp4 --annotate boxes.mp4
    python scripts/wm_eval.py video.mp4 --removal blur --maintain 0.42 --hold 12

Coverage is measured as the fraction of frames where the remover held a box
(i.e. actively removed the badge). A "visible gap" is a run of consecutive
uncovered frames.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from swap_cli.watermark import (  # noqa: E402
    BUNDLED_TEMPLATE_REF_WIDTH,
    WatermarkParams,
    WatermarkRemover,
    bundled_template_path,
)


def build_params(args: argparse.Namespace, ref_width: int) -> WatermarkParams:
    kw: dict = dict(
        template_path=Path(args.template) if args.template else bundled_template_path(),
        template_ref_width=ref_width,
        removal=args.removal,
    )
    if args.maintain is not None:
        kw["maintain_threshold"] = args.maintain
    if args.hold is not None:
        kw["hold_frames"] = args.hold
    if args.no_signature:
        kw["signature_fallback"] = False
    return WatermarkParams(**kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, help="write the cleaned video here")
    ap.add_argument("--annotate", type=Path, help="write a box-annotated video here")
    ap.add_argument("--removal", choices=["reconstruct", "blur"], default="reconstruct")
    ap.add_argument("--template", help="template PNG (default: bundled)")
    ap.add_argument(
        "--ref-width",
        type=int,
        default=None,
        help=f"template ref width (default: bundled {BUNDLED_TEMPLATE_REF_WIDTH}, "
        "or frame width for a custom one)",
    )
    ap.add_argument("--maintain", type=float, default=None, help="override maintain_threshold")
    ap.add_argument("--hold", type=int, default=None, help="override hold_frames")
    ap.add_argument("--no-signature", action="store_true", help="disable the signature net")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress per-frame [watermark] logs")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_width = args.ref_width or (BUNDLED_TEMPLATE_REF_WIDTH if not args.template else w)
    print(f"video: {w}x{h} {fps:.1f}fps {n} frames | ref_width={ref_width} removal={args.removal}")

    rem = WatermarkRemover(build_params(args, ref_width))

    if args.quiet:
        # Silence the module's per-frame diagnostics for a clean report.
        import builtins

        _orig_print = builtins.print

        def _filtered(*a, **k):
            if a and isinstance(a[0], str) and a[0].startswith("[watermark]"):
                return None
            return _orig_print(*a, **k)

        builtins.print = _filtered  # type: ignore[assignment]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h)) if args.out else None
    ann = cv2.VideoWriter(str(args.annotate), fourcc, fps, (w, h)) if args.annotate else None

    covered = total = 0
    confs: list[float] = []
    gaps: list[int] = []
    cur_gap = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        out = rem.process(frame)
        confs.append(rem._last_conf)
        box = rem._held_box
        if box is not None:
            covered += 1
            if cur_gap:
                gaps.append(cur_gap)
                cur_gap = 0
        else:
            cur_gap += 1
        if writer is not None:
            writer.write(out)
        if ann is not None:
            a = out.copy()
            if box is not None:
                x, y, bw, bh = box
                cv2.rectangle(a, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            ann.write(a)
        if args.max_frames and total >= args.max_frames:
            break
    if cur_gap:
        gaps.append(cur_gap)
    cap.release()
    if writer is not None:
        writer.release()
    if ann is not None:
        ann.release()

    if args.quiet:
        builtins.print = _orig_print  # type: ignore[assignment]

    pct = 100 * covered / max(1, total)
    longest = max(gaps) if gaps else 0
    print(f"\ncoverage: {covered}/{total} = {pct:.1f}% frames had the badge covered")
    print(f"visible (uncovered): {total - covered} = {100 - pct:.1f}%")
    if confs:
        print(
            f"raw conf: min={min(confs):.2f} med={statistics.median(confs):.2f} "
            f"max={max(confs):.2f}"
        )
    if gaps:
        gaps_sorted = sorted(gaps)
        print(
            f"visible gaps: {len(gaps)} runs, longest={longest}f (~{longest / fps:.1f}s), "
            f"median={gaps_sorted[len(gaps_sorted) // 2]}f"
        )
    if args.out:
        print(f"wrote cleaned video: {args.out}")
    if args.annotate:
        print(f"wrote annotated video: {args.annotate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
