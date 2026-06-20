"""Per-frame watermark removal for Decart Lucy-2 output (Sprint 15).

Decart stamps a semi-transparent "✦ AI Generated" pill on every output
frame and roams its position frame-to-frame, so a fixed mask won't do —
we detect it fresh each frame and cv2.inpaint it away.

Detection isolates the badge's bright text before matching: the pill is
translucent, so its *pixels* and *edges* shift with the background, but the
white "AI Generated" text + sparkle are near-opaque and constant. A white
morphological top-hat keeps those thin bright strokes and erases the pill
and the slowly-varying background, so matchTemplate becomes
background-invariant (measured: 0.9+ on the badge across slats/face/noise/
bright backgrounds, vs ~0.3–0.7 and flickering for plain edge matching,
while staying ~0.1 on bright no-badge regions — no false positives). A
brightness threshold method is offered as a cheap fallback.

The module is pure and testable: BGR ndarray in, BGR ndarray out. It
NEVER raises into the render loop — on any failure `process()` returns the
input frame unchanged (mirroring display.py's "don't tear the driver down
on a single bad frame" philosophy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config

# (x, y, w, h) in full-resolution pixels.
Box = tuple[int, int, int, int]

DetectionMethod = Literal["template", "threshold"]
# How a located badge is hidden: "reconstruct" rebuilds the real background
# behind it (temporal clean-plate + inpaint — invisible, best quality), "blur"
# just smears the badge into an unreadable soft patch (always works, no
# reconstruction artifacts, but leaves a visible soft blob). Detection is
# orthogonal — both removals still need `method` to find the badge.
RemovalMode = Literal["reconstruct", "blur"]


@dataclass(frozen=True)
class WatermarkParams:
    """Tunables for watermark detection + inpainting.

    Defaults are tuned for the Decart "AI Generated" pill (a ~190x40px
    semi-transparent badge): edge matching on, a conservative confidence
    gate, and a slightly fat dilation to cover the frosted-glass halo.
    """

    method: DetectionMethod = "template"
    template_path: Path | None = None
    # Confidence gate (0..1). 0.50 biases toward removing the badge: a miss
    # leaves the watermark visible (bad), a rare false positive is a small
    # upper-frame smudge (minor). Tuned against the bundled template.
    threshold: float = 0.50  # ACQUIRE gate: confidence to first lock onto the badge
    # MAINTAIN gate: once locked, keep tracking through dips at this bar. Set
    # close to ACQUIRE so we only fill when genuinely confident — a low gate let
    # the lock COAST onto a wrong bright spot over busy/low-contrast scenes
    # (skin, blinds, the bright corner) and reconstruct then painted a misplaced
    # patch while the real roaming badge stayed visible. The real badge scores
    # 0.65-0.94 when detected, so it stays locked; drifted locks (0.30-0.46) are
    # released so we don't fill (badge may briefly show) instead of a wrong patch.
    maintain_threshold: float = 0.45
    # Multi-template bank: match against the primary template PLUS this many
    # softened (Gaussian-blurred) copies of it, taking the best score. The badge
    # appearance varies frame-to-frame (different background bleeds through the
    # translucent pill), so a single sharp template dips below the gate on the
    # faint frames; a blur-tolerant variant lifts those without changing what's
    # matched (same badge structure → same discrimination, measured promiscuity
    # 0.16 == the primary). 0 disables (single-template behaviour).
    template_variants: int = 2
    # Isolate the bright text via white top-hat before matchTemplate, making
    # the match background-invariant for the semi-transparent badge.
    text_isolate: bool = True
    tophat_kernel: int = 13  # structuring-element size (px) for the top-hat
    roi: tuple[float, float, float, float] | None = None  # fractional x,y,w,h
    inpaint_radius: int = 3
    inpaint_method: Literal["telea", "ns"] = "telea"
    dilation: int = 8  # px to grow the footprint — generous so the badge can't
    # leak past it into the clean plate (which would reappear later)
    # The match box hugs the bright text strokes, but the translucent pill
    # extends beyond them — pad the removal footprint by this fraction of the
    # box size so the whole pill is covered. Over-covering is free with
    # temporal recovery (the extra ring is restored from real pixels), so we
    # pad generously to absorb low-confidence location error/jitter — the pill
    # edge must stay inside the SOLID footprint, not in the feathered seam, or
    # a faint sliver peeks out (and leaks into the clean plate).
    footprint_pad_frac: float = 0.30
    feather: int = 3  # gaussian blur radius for the seam blend
    # Downscale for detection. 0.6 keeps enough edge detail for the
    # semi-transparent pill (0.5 lost too much and missed) while staying
    # ~27ms/frame at 1280x720 — within the 20fps (50ms) budget.
    detect_scale: float = 0.6
    redetect_every: int = 1  # run detection every N frames
    # Track-and-hold: on a brief miss, reuse the last confident box for up to
    # hold_frames so removal stays steady through a dip. Kept SHORT on purpose:
    # the badge roams to RANDOM positions and can jump anywhere at any time, so
    # coasting a stale box for long just holds the WRONG spot while the real
    # badge shows elsewhere. The multi-template bank (global re-acquire every
    # frame) and the signature net carry coverage now — they don't need a long
    # coast — so a short hold both bridges genuine in-place flickers AND
    # minimises stale-box risk on a teleport. (Swept on real footage: 5 gave the
    # best coverage *and* the lowest stale-coast; longer holds overfit one clip.)
    hold_frames: int = 5
    # While coasting (a held box through misses), grow the removal footprint by
    # this fraction of the box per coasted frame (the badge keeps drifting
    # during the miss), capped at coast_expand_max_frac. 0 disables.
    coast_expand_frac: float = 0.06
    coast_expand_max_frac: float = 0.40
    # When tracking, search the match map within a window of this fraction of
    # frame width around the badge's last position, so a faint badge isn't lost
    # to a far background look-alike (the wrong-location "full badge" misses).
    track_window_frac: float = 0.16
    # Temporal recovery: the badge roams, so a covered pixel was visible
    # recently. Fill the footprint from a "clean plate" of the most-recent
    # uncovered value (real pixels, no inpaint trace). A pixel stuck under
    # the badge longer than temporal_max_stale frames falls back to a tight
    # inpaint. Set temporal=False to use plain inpaint only.
    temporal: bool = True
    # Motion-aware fill: when the badge is over a MOVING region (the subject
    # shaking their head / waving), the clean plate is out of date, so fill by
    # inpainting the current surroundings instead of showing a stale ghost.
    # Over a static background we keep using the (trace-free) clean plate.
    motion_aware: bool = True
    motion_thresh: float = 7.0  # mean abs diff (0–255) in the footprint's ring
    # How long a covered pixel keeps using its clean-plate value before the
    # tight-inpaint fallback. The badge SLIDES, so a pixel is covered for tens
    # of frames while the static background behind it stays valid — 90 frames
    # (~4.5s) keeps slides trace-free; the fallback still catches a badge that
    # truly parks on moving content.
    temporal_max_stale: int = 90
    # Frame width the template was authored for. Decart's negotiated output
    # resolution varies (1088x624 and 1280x720 both seen), and matchTemplate
    # is NOT scale-invariant, so we resize the template by frame_width/ref to
    # center the multi-scale search on the real badge size. The bundled
    # default was cropped from a 1280-wide frame.
    template_ref_width: int = 1280
    # Color + shape safety net (Sprint 17). An INDEPENDENT fallback detector,
    # used only when template matching fails to acquire: it looks for the badge's
    # invariants — a small, horizontal, low-saturation near-white, text-textured
    # band — anywhere in the frame. Strict by design (balanced posture): it only
    # fires on a clear badge signature, so it catches faint badges the template
    # misses without smearing skin (saturated → fails s_max) or solid bright
    # blocks (wrong aspect/too dense). Disable via config if it ever misbehaves.
    signature_fallback: bool = True
    signature_threshold: float = 0.50  # signature-fit gate (0..1) to acquire
    signature_s_max: int = 60  # max HSV saturation (0..255) — "near-white"
    signature_v_min: int = 170  # min HSV value (0..255) — "bright"
    # The net only searches near the badge's last-known centre (it roams
    # smoothly), and only for this many frames after the badge was last seen —
    # past that the badge has likely moved too far for a local search and the
    # template bank handles re-acquisition. Keeps the net a precise gap-bridger.
    signature_max_age: int = 20
    # Removal/fill strategy (see RemovalMode). "reconstruct" is the default,
    # high-quality background rebuild; "blur" smears the badge instead.
    removal: RemovalMode = "reconstruct"
    # Blur strength for removal="blur": the Gaussian kernel is this fraction of
    # the footprint's shorter side (clamped to an odd int >= 9). 0.35 reliably
    # makes the badge text unreadable without smearing far beyond it.
    blur_strength: float = 0.35
    # Blur patch opacity (0..1) for removal="blur". 1.0 fully replaces the
    # footprint with the blurred version (a solid smear); lower lets some of the
    # original show through so the patch looks more transparent/natural. 0.8
    # keeps the badge unreadable while softening the "censored blob" look.
    blur_opacity: float = 0.8


# Threshold-method default region when no ROI is configured: the upper
# band where the badge tends to sit, keeping the subject's face out of
# the brightness search so we never inpaint facial highlights.
_DEFAULT_THRESHOLD_ROI = (0.0, 0.0, 1.0, 0.45)
_BRIGHTNESS_CUTOFF = 225  # white-text/pill detection for the threshold method

# Content-scale multipliers tried around the base scale (frame_w/ref_w)
# while the badge size is unknown. Brackets ±30% so a template authored at
# one resolution still matches frames negotiated at another. Once a scale
# clears the gate it is locked and the search collapses to that one scale.
# Ordered base-first so that, with the strict `>` best-pick, near-ties favor
# the expected size (1.0) over a far/wrong scale.
_SEARCH_MULTIPLIERS = (1.0, 0.85, 1.15, 0.7, 1.3)
_LOG_EVERY = 20  # throttle diagnostics to ~1/sec at 20fps
_UNLOCK_AFTER_MISSES = 15  # re-open the scale search after this many misses

# Signature net: the badge is a horizontal text band. These bound a candidate
# component's aspect ratio (w/h) and fill density (white px / bbox area) so the
# net accepts text-like bands and rejects solid bright blocks (walls, blinds:
# too dense or wrong aspect) and skin (saturated — vetoed earlier by s_max).
_SIG_ASPECT_MIN, _SIG_ASPECT_MAX = 2.5, 9.5
_SIG_DENSITY_MIN, _SIG_DENSITY_MAX = 0.05, 0.80
# Accept a band whose width is within this factor of the expected badge width
# for the stream resolution (keeps the net at the badge's size, not arbitrary).
_SIG_WIDTH_LO, _SIG_WIDTH_HI = 0.5, 1.8

# Default template shipped with the package (a clean crop of the Decart
# "✦ AI Generated" badge). Lets watermark removal work out of the box without
# the user capturing their own — they only recapture if Decart restyles it.
_BUNDLED_TEMPLATE = Path(__file__).resolve().parent / "assets" / "watermark_default.png"
# Frame width the bundled template was cropped from, so the multi-scale search
# centers on the real badge size at any stream resolution. The bundled badge was
# cropped from a 954-wide capture of the Decart output (the badge is the same
# graphic for every user; only the scale varies with output resolution).
BUNDLED_TEMPLATE_REF_WIDTH = 954


def bundled_template_path() -> Path | None:
    """Path to the packaged default watermark template, or None if absent."""
    return _BUNDLED_TEMPLATE if _BUNDLED_TEMPLATE.is_file() else None


class WatermarkRemover:
    """Stateful per-frame watermark remover (one instance per session).

    Not thread-safe — the render loop is a single asyncio task, so the
    small detection cache (`_last_box`, `_frame_idx`) is safe there.
    """

    def __init__(self, params: WatermarkParams) -> None:
        self._params = params
        self._frame_idx = 0
        self._last_box: Box | None = None
        self._warned = False
        self._method: DetectionMethod = params.method

        # Full-resolution grayscale template. We resize it per content-scale
        # at detection time (matchTemplate is not scale-invariant), so we keep
        # the original here rather than a single pre-scaled copy.
        self._tpl_gray: np.ndarray | None = None
        self._tpl_size: tuple[int, int] | None = None  # (w, h) at authored res
        # Multi-template bank: the primary plus softened variants, each
        # (gray, (w,h)) at authored res. templates[0] is always the primary
        # (mirrors _tpl_gray/_tpl_size). Built by _build_bank().
        self._templates: list[tuple[np.ndarray, tuple[int, int]]] = []

        # Multi-scale state: the content scale that last matched (so steady
        # state searches one scale), a miss counter to re-open the search, and
        # a cache of resized+edged search templates keyed by (tpl_idx, width px).
        self._locked_scale: float | None = None
        self._locked_tpl_idx: int | None = None  # which bank template last won
        self._miss_streak = 0
        self._tpl_cache: dict[tuple[int, int], np.ndarray] = {}
        self._last_conf = 0.0
        self._last_scale = 0.0
        # Track-and-hold: last confident box + how many misses we've coasted.
        self._held_box: Box | None = None
        self._hold_count = 0
        self._gate_mode = "acquire"  # "acquire" | "maintain" (hysteresis)
        self._last_center: tuple[int, int] | None = None  # for local-search tracking
        # Signature net: the badge's last-known centre (persists across a lock
        # release so the net can keep bridging) and frames since last seen.
        self._sig_center: tuple[int, int] | None = None
        self._sig_age = 0
        self._prev_bgr: np.ndarray | None = None  # for motion-aware fill

        # Temporal recovery state: a per-pixel "clean plate" (most-recent
        # uncovered value) and an age map (frames since last seen uncovered).
        # Lazily sized to the first frame; rebuilt if the frame size changes.
        self._clean_plate: np.ndarray | None = None
        self._age: np.ndarray | None = None
        self._last_fill = "none"
        self._last_stale_frac = 0.0

        if params.method == "template":
            self._load_template(params.template_path)
            if self._tpl_gray is None:
                # Requested template matching but no usable PNG — fall back
                # to the brightness method so we still do *something*.
                print(
                    "[watermark] no usable template — falling back to "
                    "brightness threshold method.",
                    flush=True,
                )
                self._method = "threshold"
            else:
                self._build_bank()

    # ---- construction helpers -------------------------------------------------

    def _load_template(self, template_path: Path | None) -> None:
        if template_path is None:
            return
        path = Path(template_path)
        if not path.exists():
            print(f"[watermark] template not found: {path}", flush=True)
            return
        tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tpl is None or tpl.size == 0:
            print(f"[watermark] template unreadable: {path}", flush=True)
            return
        h, w = tpl.shape[:2]
        self._tpl_gray = tpl
        self._tpl_size = (w, h)

    def _build_bank(self) -> None:
        """Assemble the template bank: the primary plus softened (blurred)
        copies. The badge's appearance varies (background bleed through the
        translucent pill), so a blur-tolerant variant catches faint frames the
        sharp primary dips below the gate on — without changing WHAT is matched
        (same badge, so the same low score on non-badge content)."""
        if self._tpl_gray is None or self._tpl_size is None:
            return
        bank: list[tuple[np.ndarray, tuple[int, int]]] = [(self._tpl_gray, self._tpl_size)]
        # Odd, increasing kernels; each is a softer rendition of the same badge.
        for k in (3, 5, 7)[: max(0, self._params.template_variants)]:
            soft = cv2.GaussianBlur(self._tpl_gray, (k, k), 0)
            bank.append((soft, self._tpl_size))
        self._templates = bank

    @classmethod
    def from_config(cls, cfg: Config, *, enabled: bool) -> WatermarkRemover | None:
        """Build a remover from user config, or None if it shouldn't run.

        Returns None when removal is disabled, or when template matching is
        requested but no template is configured (nothing to match against
        and the brightness fallback would be too risky to enable silently).
        """
        if not enabled:
            return None
        method: DetectionMethod = (
            "threshold" if cfg.watermark_method == "threshold" else "template"
        )
        # User's captured template wins; otherwise fall back to the bundled
        # default so removal works out of the box.
        template = Path(cfg.watermark_template) if cfg.watermark_template else None
        if template is None:
            template = bundled_template_path()
        if method == "template" and template is None:
            print(
                "[watermark] enabled but no template available (none captured "
                "and no bundled default). Run `swap capture-watermark`. "
                "Skipping removal.",
                flush=True,
            )
            return None
        # A user-captured template records the frame width it was grabbed at so
        # the scale search centers exactly; the bundled default uses its own
        # known crop width.
        is_custom = bool(cfg.watermark_template)
        ref_width = (
            (getattr(cfg, "watermark_template_width", None) or 1280)
            if is_custom
            else BUNDLED_TEMPLATE_REF_WIDTH
        )
        params = WatermarkParams(
            method=method,
            template_path=template,
            threshold=cfg.watermark_threshold,
            inpaint_radius=cfg.watermark_inpaint_radius,
            template_ref_width=ref_width,
            signature_fallback=getattr(cfg, "watermark_signature_fallback", True),
        )
        return cls(params)

    # ---- public per-frame entrypoint -----------------------------------------

    def process(self, bgr: np.ndarray) -> np.ndarray:
        """Detect + inpaint the watermark; return the cleaned BGR frame.

        Never raises: on any failure returns `bgr` unchanged and logs once.
        """
        try:
            return self._process(bgr)
        except Exception as err:  # noqa: BLE001 — degrade, never crash the loop
            if not self._warned:
                print(f"[watermark] disabled this frame: {err}", flush=True)
                self._warned = True
            return bgr

    def _process(self, bgr: np.ndarray) -> np.ndarray:
        self._frame_idx += 1
        if self._frame_idx == 1:
            self._log_active(bgr.shape[:2])

        redetect = self._held_box is None or (
            self._frame_idx % max(1, self._params.redetect_every) == 0
        )
        if redetect:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            box, conf = self._detect(gray)
            self._last_conf = conf
            # Hysteresis: high bar to ACQUIRE the badge from cold, low bar to
            # MAINTAIN the lock once we have it (the badge is ever-present and
            # roams smoothly, so coasting at a lower gate is safe and steady).
            tracking = self._held_box is not None
            gate = self._params.maintain_threshold if tracking else self._params.threshold
            self._gate_mode = "maintain" if tracking else "acquire"
            accepted = box is not None and conf >= gate
            # Safety net: when template matching won't pass the gate, fall back to
            # the independent color/shape signature detector — searched LOCALLY,
            # near where the badge was last seen (it roams smoothly), for a short
            # window after the last sighting. Strict bands + the local gate mean
            # it only catches the real (faint) badge, never a far bright look-alike.
            self._sig_age += 1
            if (
                not accepted
                and self._params.signature_fallback
                and self._method == "template"
                and self._sig_center is not None
                and self._sig_age <= self._params.signature_max_age
            ):
                _frame_h, frame_w = gray.shape[:2]
                radius = max(
                    40.0,
                    self._params.track_window_frac * frame_w * (1 + self._sig_age),
                )
                sbox, sconf = self._detect_signature(bgr, near=self._sig_center, radius=radius)
                if sbox is not None and sconf >= self._params.signature_threshold:
                    box, conf, accepted = sbox, max(conf, gate), True
                    self._gate_mode = "signature"
            if accepted:
                self._last_box = box
                self._held_box = box
                self._hold_count = 0
                self._miss_streak = 0
                self._last_center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
                # Refresh the signature net's anchor so it keeps tracking the
                # roaming badge across consecutive misses (and after a release).
                self._sig_center = self._last_center
                self._sig_age = 0
            else:
                self._miss_streak += 1
                # Coast on the last box through brief dips (track-and-hold).
                if self._held_box is not None and self._hold_count < self._params.hold_frames:
                    self._hold_count += 1
                else:
                    # Fully lost the badge: release, and only NOW re-open the
                    # scale search. While tracking we keep the locked size
                    # constant so the box can't drift to a wrong scale.
                    self._held_box = None
                    self._hold_count = 0
                    self._last_center = None
                    self._locked_scale = None
                    self._locked_tpl_idx = None
            # Drop the signature anchor once the badge hasn't been seen for long
            # enough that a local search would be guessing.
            if self._sig_age > self._params.signature_max_age:
                self._sig_center = None
            self._log_detection(self._held_box, accepted)

        box = self._held_box
        if box is None:
            return bgr  # watermark absent / not confident — pass through

        if self._params.removal == "blur":
            return self._blur_region(bgr, box)
        if self._params.temporal:
            return self._restore(bgr, box)
        # Plain-inpaint path (temporal disabled).
        mask = self._build_mask(bgr.shape[:2], box)
        return self._inpaint(bgr, mask)

    # ---- diagnostics ----------------------------------------------------------

    def _log_active(self, shape: tuple[int, int]) -> None:
        h, w = shape
        if self._method == "threshold":
            print(
                f"[watermark] active · method=threshold · gate={self._params.threshold}"
                f" · frame={w}x{h}",
                flush=True,
            )
            return
        tw, th = self._tpl_size or (0, 0)
        print(
            f"[watermark] active · template {tw}x{th} (ref {self._params.template_ref_width}px)"
            f" · gate={self._params.threshold} · frame={w}x{h}",
            flush=True,
        )

    def _log_detection(self, box: Box | None, accepted: bool = True) -> None:
        # Throttle to ~1/sec so we don't flood stdout at 20fps.
        if self._frame_idx % _LOG_EVERY != 0:
            return
        if box is not None:
            x, y, bw, bh = box
            fill = ""
            if self._params.temporal:
                fill = f" fill={self._last_fill} stale={self._last_stale_frac * 100:.0f}%"
            hold = ""
            if self._hold_count:
                hold = f" hold={self._hold_count}/{self._params.hold_frames}"
            mode = "" if accepted else " [coasting]"
            print(
                f"[watermark] match conf={self._last_conf:.2f} ({self._gate_mode}{mode}) "
                f"scale={self._last_scale:.2f} at ({x},{y}) {bw}x{bh}{fill}{hold}",
                flush=True,
            )
        else:
            print(
                f"[watermark] no match · best conf={self._last_conf:.2f} < "
                f"gate {self._params.threshold} — press W in the preview to "
                "capture the badge.",
                flush=True,
            )

    # ---- detection ------------------------------------------------------------

    def _detect(self, gray: np.ndarray) -> tuple[Box | None, float]:
        if self._method == "threshold":
            return self._detect_threshold(gray)
        return self._detect_template(gray)

    def _detect_template(self, gray: np.ndarray) -> tuple[Box | None, float]:
        if not self._templates:
            return None, 0.0

        _frame_h, frame_w = gray.shape[:2]
        ds = self._params.detect_scale
        # Downscale the frame once; reused across every candidate scale.
        small = self._scale(gray)
        search = self._isolate(small) if self._params.text_isolate else small

        # Center the search on the badge's expected size for this resolution.
        base = frame_w / max(1, self._params.template_ref_width)
        if self._locked_scale is not None:
            candidates = (self._locked_scale,)
        else:
            candidates = tuple(base * m for m in _SEARCH_MULTIPLIERS)

        # While locked, search ONLY the template that won the lock at the locked
        # scale (steady-state cost = 1 template x 1 scale). Cold / re-acquire
        # searches the whole bank x every scale and takes the global best.
        if self._locked_tpl_idx is not None and self._locked_scale is not None:
            tpl_indices: tuple[int, ...] = (self._locked_tpl_idx,)
        else:
            tpl_indices = tuple(range(len(self._templates)))

        best_conf = 0.0
        best_loc: tuple[int, int] | None = None
        best_cs = 0.0
        best_tpl = 0
        # When tracking, also find the best peak in a window around the badge's
        # last position, so a faint badge isn't lost to a far look-alike.
        tracking = self._locked_scale is not None and self._last_center is not None
        local_conf = 0.0
        local_loc: tuple[int, int] | None = None
        local_cs = 0.0
        local_tpl = 0
        for ti in tpl_indices:
            tpl_w, tpl_h = self._templates[ti][1]
            for cs in candidates:
                # Final search-template size = authored x content-scale x detect-scale.
                sw = round(tpl_w * cs * ds)
                sh = round(tpl_h * cs * ds)
                if sw < 8 or sh < 8 or sh > search.shape[0] or sw > search.shape[1]:
                    continue
                tpl = self._scaled_template(ti, sw, sh)
                result = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
                _mn, mx, _ml, ml = cv2.minMaxLoc(result)
                if mx > best_conf:
                    best_conf, best_loc, best_cs, best_tpl = float(mx), ml, cs, ti
                if tracking:
                    lc, ll = self._local_peak(result, sw, sh, ds, frame_w)
                    if ll is not None and lc > local_conf:
                        local_conf, local_loc, local_cs, local_tpl = lc, ll, cs, ti

        # While tracking, prefer the local (near-the-badge) peak ONLY when it's
        # a real match — that avoids a far background look-alike. But when the
        # local window is empty (the badge JUMPED away), fall back to the global
        # peak so we re-acquire it THIS frame instead of coasting at the stale
        # spot (which is what briefly showed the full pill).
        if tracking and local_loc is not None:
            far = best_loc != local_loc
            # A confident, clearly-stronger far peak = a genuine fast move → take it.
            strong_global = (
                far
                and best_conf >= self._params.threshold
                and best_conf > local_conf + 0.10
            )
            if local_conf >= self._params.maintain_threshold and not strong_global:
                best_conf, best_loc, best_cs, best_tpl = (
                    local_conf,
                    local_loc,
                    local_cs,
                    local_tpl,
                )
            # else: keep the global peak (re-acquire the jumped badge).

        if best_loc is None:
            return None, best_conf

        # Return the best candidate regardless of gate — _process applies the
        # acquire/maintain hysteresis. Lock the scale AND the winning template
        # only on a CONFIDENT (acquire-level) match: the badge size/appearance is
        # constant within a session, so a marginal match must never shrink/grow
        # the locked box (the 0.70-scale drift) or switch templates mid-track.
        if best_conf >= self._params.threshold:
            self._locked_scale = best_cs
            self._locked_tpl_idx = best_tpl
        self._last_scale = best_cs
        tpl_w, tpl_h = self._templates[best_tpl][1]
        x = round(best_loc[0] / ds)
        y = round(best_loc[1] / ds)
        w = round(tpl_w * best_cs)
        h = round(tpl_h * best_cs)
        return (x, y, w, h), best_conf

    def _local_peak(
        self, result: np.ndarray, sw: int, sh: int, ds: float, frame_w: int
    ) -> tuple[float, tuple[int, int] | None]:
        """Best match within a window around the badge's last center, so a far
        background look-alike can't steal the lock when the badge is faint.
        Returns (conf, top-left loc) in result-map coords, or (0, None)."""
        if self._last_center is None:
            return 0.0, None
        cx, cy = self._last_center
        # Expected template top-left for a badge centered at (cx, cy), in the
        # downscaled result-map coordinate space.
        ex = round(cx * ds - sw / 2)
        ey = round(cy * ds - sh / 2)
        # Window grows while coasting so a moved badge is still found.
        r = max(8, round(self._params.track_window_frac * frame_w * (1 + self._hold_count) * ds))
        x0, y0 = max(0, ex - r), max(0, ey - r)
        x1 = min(result.shape[1], ex + r + 1)
        y1 = min(result.shape[0], ey + r + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0, None
        sub = result[y0:y1, x0:x1]
        _mn, mx, _ml, ml = cv2.minMaxLoc(sub)
        return float(mx), (ml[0] + x0, ml[1] + y0)

    def _scaled_template(self, ti: int, sw: int, sh: int) -> np.ndarray:
        """Resize (and edge) bank template `ti` to a search size, cached by
        (template index, width)."""
        key = (ti, sw)
        cached = self._tpl_cache.get(key)
        if cached is not None and cached.shape[0] == sh:
            return cached
        gray = self._templates[ti][0]
        resized = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        tpl = self._isolate(resized) if self._params.text_isolate else resized
        self._tpl_cache[key] = tpl
        return tpl

    def _detect_threshold(self, gray: np.ndarray) -> tuple[Box | None, float]:
        h, w = gray.shape[:2]
        rx, ry, rw, rh = _roi_to_px(self._params.roi or _DEFAULT_THRESHOLD_ROI, h, w)
        region = gray[ry : ry + rh, rx : rx + rw]
        if region.size == 0:
            return None, 0.0

        _ret, mask = cv2.threshold(region, _BRIGHTNESS_CUTOFF, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        box_area = max(1, bw * bh)
        bright = int(cv2.countNonZero(mask[by : by + bh, bx : bx + bw]))
        confidence = bright / box_area
        # Return the best candidate; _process applies the acquire/maintain gate.
        return (bx + rx, by + ry, bw, bh), confidence

    # ---- signature net (color + shape fallback) -------------------------------

    def _expected_badge_w(self, frame_w: int) -> float:
        """Expected badge width (full-res px) at this stream resolution, from the
        primary template's authored size scaled by frame_w/ref_width."""
        if not self._templates:
            return frame_w * 0.2
        tpl_w = self._templates[0][1][0]
        base = frame_w / max(1, self._params.template_ref_width)
        return tpl_w * base

    def _detect_signature(
        self,
        bgr: np.ndarray,
        near: tuple[int, int] | None = None,
        radius: float | None = None,
    ) -> tuple[Box | None, float]:
        """Fallback used when template matching didn't acquire: find the badge by
        its signature — a small, horizontal, low-saturation near-white,
        text-textured band — NEAR its last-known position `near` (within
        `radius` full-res px). The local gate is essential: real frames have many
        near-white horizontal bands (collars, teeth, bright strips), so a global
        search lands on the wrong one (measured 0% on-badge); restricted to the
        roaming badge's neighbourhood it's precise. Strict aspect/density/size
        bands plus the near-white veto keep it off skin and solid bright blocks.
        Returns (full-res box, fit-confidence 0..1) or (None, 0.0)."""
        ds = self._params.detect_scale
        small = cv2.resize(bgr, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        white = ((s < self._params.signature_s_max) & (v > self._params.signature_v_min)).astype(
            np.uint8
        ) * 255
        if int(cv2.countNonZero(white)) == 0:
            return None, 0.0
        # Merge text strokes into a horizontal band (wide kernel), then drop
        # vertical hairlines so tall non-text shapes don't survive.
        band = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        )

        exp_w_ds = self._expected_badge_w(bgr.shape[1]) * ds
        lo_w, hi_w = exp_w_ds * _SIG_WIDTH_LO, exp_w_ds * _SIG_WIDTH_HI
        # Local gate: only consider bands near the badge's last-known centre.
        nx = ny = nr_ds = None
        if near is not None and radius is not None:
            nx, ny = near[0] * ds, near[1] * ds
            nr_ds = radius * ds

        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(band, connectivity=8)
        best_conf = 0.0
        best_box: Box | None = None
        for i in range(1, n):  # skip background label 0
            x, y, w, h, _area = (int(stats[i, k]) for k in range(5))
            if w < 8 or h < 4:
                continue
            if nx is not None:
                ccx, ccy = x + w / 2, y + h / 2
                if ((ccx - nx) ** 2 + (ccy - ny) ** 2) ** 0.5 > nr_ds:
                    continue  # outside the badge's neighbourhood — skip
            aspect = w / max(1, h)
            # Score density from the RAW near-white mask in the band's bbox (not
            # the morph-closed area): text strokes leave it sparse (~0.3-0.55),
            # while a solid bright block fills it (~1.0), cleanly separable.
            raw = int(cv2.countNonZero(white[y : y + h, x : x + w]))
            density = raw / max(1, w * h)
            if not (_SIG_ASPECT_MIN <= aspect <= _SIG_ASPECT_MAX):
                continue
            if not (_SIG_DENSITY_MIN <= density <= _SIG_DENSITY_MAX):
                continue
            if not (lo_w <= w <= hi_w):
                continue
            conf = min(
                _band_score(aspect, _SIG_ASPECT_MIN, _SIG_ASPECT_MAX),
                _band_score(density, _SIG_DENSITY_MIN, _SIG_DENSITY_MAX),
                _band_score(w, lo_w, hi_w),
            )
            if conf > best_conf:
                best_conf = conf
                best_box = (round(x / ds), round(y / ds), round(w / ds), round(h / ds))
        return best_box, best_conf

    # ---- mask + inpaint -------------------------------------------------------

    def _build_mask(self, shape: tuple[int, int], box: Box) -> np.ndarray:
        h, w = shape
        x, y, bw, bh = box
        mask = np.zeros((h, w), dtype=np.uint8)
        # Clamp the box to frame bounds so a near-edge match can't index OOB.
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            return mask
        mask[y0:y1, x0:x1] = 255

        if self._params.dilation > 0:
            k = 2 * self._params.dilation + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        if self._params.feather > 0:
            k = 2 * self._params.feather + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    def _inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        flag = cv2.INPAINT_NS if self._params.inpaint_method == "ns" else cv2.INPAINT_TELEA
        # cv2.inpaint wants a binary (0/255) single-channel mask.
        binary = (mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(bgr, binary, self._params.inpaint_radius, flag)

    # ---- temporal recovery ----------------------------------------------------

    def _padded_box(self, shape: tuple[int, int], box: Box) -> tuple[int, int, int, int]:
        """Box grown by footprint_pad_frac (min dilation), clamped to frame —
        covers the whole pill, not just the matched text strokes. While coasting
        (a held box through misses), grows further with `_hold_count` so a badge
        drifting during the miss stays under cover."""
        h, w = shape
        x, y, bw, bh = box
        pad_frac = self._params.footprint_pad_frac
        if self._hold_count > 0:
            pad_frac += min(
                self._params.coast_expand_max_frac,
                self._params.coast_expand_frac * self._hold_count,
            )
        pad_x = max(self._params.dilation, round(pad_frac * bw))
        pad_y = max(self._params.dilation, round(pad_frac * bh))
        return (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(w, x + bw + pad_x),
            min(h, y + bh + pad_y),
        )

    def _blur_region(self, bgr: np.ndarray, box: Box) -> np.ndarray:
        """Hide the badge by Gaussian-blurring its footprint into an unreadable
        soft patch (removal="blur").

        Unlike reconstruction this doesn't try to rebuild the background — it
        just smears the badge, so it always works (no template-reconstruction
        artifacts) at the cost of a visible soft blob. The blur kernel scales
        with the footprint so the text becomes illegible, and the same feathered
        seam as `_restore` blends the patch edge."""
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self._padded_box((h, w), box)
        self._last_fill = "blur"
        self._last_stale_frac = 0.0
        if x1 <= x0 or y1 <= y0:
            return bgr

        sub = (slice(y0, y1), slice(x0, x1))
        bgr_sub = bgr[sub]
        short_side = min(y1 - y0, x1 - x0)
        ksize = max(9, round(self._params.blur_strength * short_side))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(bgr_sub, (ksize, ksize), 0)

        footprint = np.zeros((h, w), dtype=np.uint8)
        footprint[y0:y1, x0:x1] = 255
        fp_sub = footprint[sub]
        if self._params.feather > 0:
            k = 2 * self._params.feather + 1
            alpha = cv2.GaussianBlur(fp_sub, (k, k), 0).astype(np.float32) / 255.0
        else:
            alpha = (fp_sub > 0).astype(np.float32)
        # Scale by blur_opacity so the patch can stay a little transparent (some
        # of the original shows through) instead of a solid smear.
        alpha = alpha * float(np.clip(self._params.blur_opacity, 0.0, 1.0))
        alpha = alpha[:, :, None]
        blended = bgr_sub.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha

        out = bgr.copy()
        out[sub] = blended.astype(np.uint8)
        return out

    def _scene_moving(
        self, bgr_sub: np.ndarray, cov_sub: np.ndarray, sub: tuple
    ) -> bool:
        """True if the visible surroundings of the badge changed vs the previous
        frame (the subject is moving there) — so the clean plate is stale and we
        should inpaint instead of showing a ghost."""
        if not self._params.motion_aware or self._prev_bgr is None:
            return False
        if self._prev_bgr.shape[:2] != self._clean_plate.shape[:2]:
            return False
        prev_sub = self._prev_bgr[sub]
        if prev_sub.shape != bgr_sub.shape:
            return False
        ring = ~cov_sub  # uncovered surroundings within the sub-window
        if not bool(ring.any()):
            return False
        diff = np.abs(bgr_sub[ring].astype(np.int16) - prev_sub[ring].astype(np.int16))
        return float(diff.mean()) > self._params.motion_thresh

    def _restore(self, bgr: np.ndarray, box: Box) -> np.ndarray:
        """Fill the badge footprint with real pixels from a per-pixel "clean
        plate" (the most-recent uncovered value), since the roaming badge
        reveals every location moments before/after it covers it. Pixels stuck
        under the badge longer than `temporal_max_stale` fall back to a tight
        inpaint. A feathered seam blends the boundary."""
        h, w = bgr.shape[:2]
        max_stale = self._params.temporal_max_stale

        # (Re)initialise buffers on first frame or a resolution change. Age
        # starts "stale" everywhere so a pixel is only trusted once we've
        # actually seen it uncovered (avoids baking in the frame-1 badge).
        if (
            self._clean_plate is None
            or self._age is None
            or self._clean_plate.shape[:2] != (h, w)
        ):
            self._clean_plate = bgr.copy()
            self._age = np.full((h, w), max_stale + 1, dtype=np.int32)

        # Footprint = the box, padded generously so the translucent pill
        # (which extends past the matched text strokes) is fully covered.
        # Over-covering is free with temporal recovery.
        x0, y0, x1, y1 = self._padded_box((h, w), box)
        footprint = np.zeros((h, w), dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            footprint[y0:y1, x0:x1] = 255

        covered = footprint > 0
        # Refresh the plate everywhere the badge ISN'T (fast: copy the whole
        # frame, then revert the few covered pixels to their last-clean value)
        # — far cheaper than fancy-indexing ~99% of the frame.
        prev_plate = self._clean_plate
        self._clean_plate = bgr.copy()
        self._clean_plate[covered] = prev_plate[covered]
        self._age[~covered] = 0
        self._age[covered] += 1

        # All remaining work is localised to the footprint's bounding box (+a
        # margin for feather/inpaint), so cost is O(badge area), not O(frame).
        if not covered.any():
            self._prev_bgr = bgr
            return bgr
        margin = self._params.feather + self._params.inpaint_radius + 2
        sx0, sy0 = max(0, x0 - margin), max(0, y0 - margin)
        sx1, sy1 = min(w, x1 + margin), min(h, y1 + margin)
        sub = (slice(sy0, sy1), slice(sx0, sx1))

        fp_sub = footprint[sub]
        cov_sub = fp_sub > 0
        age_sub = self._age[sub]
        bgr_sub = bgr[sub]
        flag = cv2.INPAINT_NS if self._params.inpaint_method == "ns" else cv2.INPAINT_TELEA

        # Motion-aware: is the badge over a MOVING region (the subject moving)?
        # Compare the visible surroundings (uncovered part of the sub-window) to
        # the previous frame. If moving, the clean plate is out of date.
        moving = self._scene_moving(bgr_sub, cov_sub, sub)

        filled = bgr_sub.copy()
        if moving:
            # Over a moving subject: inpaint the whole footprint from the
            # current surroundings (a neutral patch, not a stale face-ghost).
            inpaint_mask = (cov_sub.astype(np.uint8)) * 255
            patched = cv2.inpaint(filled, inpaint_mask, self._params.inpaint_radius, flag)
            filled[cov_sub] = patched[cov_sub]
            self._last_fill = "inpaint(motion)"
            self._last_stale_frac = 1.0
        else:
            fresh_sub = cov_sub & (age_sub <= max_stale)
            stale_sub = cov_sub & (age_sub > max_stale)
            filled[fresh_sub] = self._clean_plate[sub][fresh_sub]  # real recent pixels
            stale_frac = 0.0
            if stale_sub.any():
                # Tight fallback: inpaint only the actual badge strokes in the
                # stale area, so the rare hallucination stays minimal.
                shape = self._badge_shape(bgr_sub)
                inpaint_mask = ((stale_sub & (shape > 0)).astype(np.uint8)) * 255
                if int(cv2.countNonZero(inpaint_mask)) == 0:
                    inpaint_mask = (stale_sub.astype(np.uint8)) * 255
                patched = cv2.inpaint(filled, inpaint_mask, self._params.inpaint_radius, flag)
                sel = inpaint_mask > 0
                filled[sel] = patched[sel]
                stale_frac = float(stale_sub.sum()) / max(1, int(cov_sub.sum()))
            self._last_fill = "temporal+inpaint" if stale_frac > 0 else "temporal"
            self._last_stale_frac = stale_frac

        # Feathered seam (local) so a plate/exposure mismatch shows no hard edge.
        if self._params.feather > 0:
            k = 2 * self._params.feather + 1
            alpha = cv2.GaussianBlur(fp_sub, (k, k), 0).astype(np.float32) / 255.0
        else:
            alpha = cov_sub.astype(np.float32)
        alpha = alpha[:, :, None]
        blended = bgr_sub.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha

        out = bgr.copy()
        out[sub] = blended.astype(np.uint8)
        self._prev_bgr = bgr  # for next frame's motion check
        return out

    def _badge_shape(self, bgr_sub: np.ndarray) -> np.ndarray:
        """Mask of the badge's bright strokes in a frame sub-image (which
        still shows the badge), via a full-res white top-hat. Kernel is scaled
        up from the downscaled detector's so it isolates the same physical
        strokes at full resolution. Used to keep the inpaint fallback tight."""
        if bgr_sub.size == 0:
            return np.zeros(bgr_sub.shape[:2], dtype=np.uint8)
        region = cv2.cvtColor(bgr_sub, cv2.COLOR_BGR2GRAY)
        ksz = max(3, round(self._params.tophat_kernel / max(0.1, self._params.detect_scale)))
        if ksz % 2 == 0:
            ksz += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        top = cv2.morphologyEx(region, cv2.MORPH_TOPHAT, kernel)
        _ret, strokes = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.dilate(strokes, np.ones((3, 3), np.uint8))

    # ---- small utilities ------------------------------------------------------

    def _scale(self, gray: np.ndarray) -> np.ndarray:
        s = self._params.detect_scale
        if s >= 0.999:
            return gray
        return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def _isolate(self, gray: np.ndarray) -> np.ndarray:
        """White top-hat: keep bright strokes smaller than the kernel (the
        text/sparkle), erase the pill and slowly-varying background. This is
        what makes matching invariant to whatever is behind the translucent
        badge."""
        k = max(3, self._params.tophat_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def _band_score(x: float, lo: float, hi: float, ramp: float = 0.2) -> float:
    """Trapezoidal membership 0..1: 1.0 across the central plateau, ramping to 0
    at the edges lo/hi over `ramp` of the band width. Comfortably-inside values
    score ~1.0; borderline ones score lower (a soft margin for ranking)."""
    if x <= lo or x >= hi:
        return 0.0
    span = hi - lo
    edge = span * ramp
    if x < lo + edge:
        return (x - lo) / edge
    if x > hi - edge:
        return (hi - x) / edge
    return 1.0


def _roi_to_px(roi: tuple[float, float, float, float], h: int, w: int) -> Box:
    """Convert a fractional (x, y, w, h) ROI in 0..1 into pixel coords."""
    fx, fy, fw, fh = roi
    x = round(fx * w)
    y = round(fy * h)
    rw = round(fw * w)
    rh = round(fh * h)
    return x, y, max(1, rw), max(1, rh)
