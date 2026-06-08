"""Per-frame watermark removal for Decart Lucy-2 output (Sprint 15).

Decart stamps a semi-transparent "✦ AI Generated" pill on every output
frame and roams its position frame-to-frame, so a fixed mask won't do —
we detect it fresh each frame and cv2.inpaint it away.

Detection is edge-based template matching by default: the pill is
translucent, so the *pixels* under it shift with the background while the
pill outline / sparkle / glyph *edges* stay constant. Matching a Canny
edge map is therefore far more robust than matching raw BGR. A brightness
threshold method is offered as a cheap fallback.

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
    threshold: float = 0.50
    edge_match: bool = True  # Canny edges before matchTemplate (semi-transparent)
    roi: tuple[float, float, float, float] | None = None  # fractional x,y,w,h
    inpaint_radius: int = 3
    inpaint_method: Literal["telea", "ns"] = "telea"
    dilation: int = 6  # px to grow the mask before inpaint
    feather: int = 2  # gaussian blur radius on mask edges
    # Downscale for detection. 0.6 keeps enough edge detail for the
    # semi-transparent pill (0.5 lost too much and missed) while staying
    # ~27ms/frame at 1280x720 — within the 20fps (50ms) budget.
    detect_scale: float = 0.6
    redetect_every: int = 1  # run detection every N frames


# Threshold-method default region when no ROI is configured: the upper
# band where the badge tends to sit, keeping the subject's face out of
# the brightness search so we never inpaint facial highlights.
_DEFAULT_THRESHOLD_ROI = (0.0, 0.0, 1.0, 0.45)
_BRIGHTNESS_CUTOFF = 225  # white-text/pill detection for the threshold method

# Default template shipped with the package (a crop of the Decart "AI
# Generated" pill). Lets watermark removal work out of the box without the
# user capturing their own — they only recapture if Decart restyles it.
_BUNDLED_TEMPLATE = Path(__file__).resolve().parent / "assets" / "watermark_default.png"


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

        # Pre-compute the scaled edge template once. `_tpl_size` holds the
        # FULL-resolution (w, h) used to size the inpaint mask.
        self._tpl_edge: np.ndarray | None = None
        self._tpl_size: tuple[int, int] | None = None

        if params.method == "template":
            self._load_template(params.template_path)
            if self._tpl_edge is None:
                # Requested template matching but no usable PNG — fall back
                # to the brightness method so we still do *something*.
                print(
                    "[watermark] no usable template — falling back to "
                    "brightness threshold method.",
                    flush=True,
                )
                self._method = "threshold"

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
        self._tpl_size = (w, h)
        scaled = self._scale(tpl)
        self._tpl_edge = self._edge(scaled) if self._params.edge_match else scaled

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
        params = WatermarkParams(
            method=method,
            template_path=template,
            threshold=cfg.watermark_threshold,
            inpaint_radius=cfg.watermark_inpaint_radius,
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
        redetect = self._last_box is None or (
            self._frame_idx % max(1, self._params.redetect_every) == 0
        )
        if redetect:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            box, _conf = self._detect(gray)
            self._last_box = box

        box = self._last_box
        if box is None:
            return bgr  # watermark absent / not confident — pass through

        mask = self._build_mask(bgr.shape[:2], box)
        return self._inpaint(bgr, mask)

    # ---- detection ------------------------------------------------------------

    def _detect(self, gray: np.ndarray) -> tuple[Box | None, float]:
        if self._method == "threshold":
            return self._detect_threshold(gray)
        return self._detect_template(gray)

    def _detect_template(self, gray: np.ndarray) -> tuple[Box | None, float]:
        if self._tpl_edge is None or self._tpl_size is None:
            return None, 0.0
        small = self._scale(gray)
        search = self._edge(small) if self._params.edge_match else small
        th, tw = self._tpl_edge.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return None, 0.0

        result = cv2.matchTemplate(search, self._tpl_edge, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
        if max_v < self._params.threshold:
            return None, float(max_v)

        scale = self._params.detect_scale
        x = round(max_l[0] / scale)
        y = round(max_l[1] / scale)
        w, h = self._tpl_size
        return (x, y, w, h), float(max_v)

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
        if confidence < self._params.threshold:
            return None, confidence
        # Offset back into full-frame coordinates.
        return (bx + rx, by + ry, bw, bh), confidence

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

    # ---- small utilities ------------------------------------------------------

    def _scale(self, gray: np.ndarray) -> np.ndarray:
        s = self._params.detect_scale
        if s >= 0.999:
            return gray
        return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _edge(gray: np.ndarray) -> np.ndarray:
        return cv2.Canny(gray, 60, 180)


def _roi_to_px(roi: tuple[float, float, float, float], h: int, w: int) -> Box:
    """Convert a fractional (x, y, w, h) ROI in 0..1 into pixel coords."""
    fx, fy, fw, fh = roi
    x = round(fx * w)
    y = round(fy * h)
    rw = round(fw * w)
    rh = round(fh * h)
    return x, y, max(1, rw), max(1, rh)
