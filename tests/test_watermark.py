"""Tests for per-frame watermark removal (Sprint 15).

Unit tests that exercise real detection / mask / inpaint need a real cv2,
so they `pytest.importorskip("cv2")` and skip in the bare CI sandbox
(same workaround as test_virtual_camera.py). The wiring tests below run
without cv2 behaviour by constructing params/objects only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Other test modules (e.g. test_virtual_camera) insert a stub `cv2`
# ModuleType into sys.modules. That stub lacks the real OpenCV symbols we
# exercise here, and importorskip would happily hand it back. Evict any
# such stub so we import the real cv2 (or skip cleanly when it's absent).
_stub_cv2 = sys.modules.get("cv2")
if _stub_cv2 is not None and not hasattr(_stub_cv2, "imwrite"):
    del sys.modules["cv2"]
    # Drop swap_cli modules that bound the stub cv2 at import (e.g. via
    # test_virtual_camera) so they re-import against the real cv2 below.
    for _m in ("swap_cli.watermark", "swap_cli.display"):
        sys.modules.pop(_m, None)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from swap_cli.watermark import (  # noqa: E402
    BUNDLED_TEMPLATE_REF_WIDTH,
    WatermarkParams,
    WatermarkRemover,
    _roi_to_px,
    bundled_template_path,
)

# --- helpers ---------------------------------------------------------------

FRAME_W, FRAME_H = 640, 360
WM_W, WM_H = 120, 32


def _make_watermark_template() -> np.ndarray:
    """A small high-contrast badge: white rounded text-like glyph block."""
    tpl = np.zeros((WM_H, WM_W, 3), dtype=np.uint8)
    cv2.rectangle(tpl, (4, 4), (WM_W - 4, WM_H - 4), (200, 200, 200), 2)
    cv2.putText(tpl, "AI", (14, WM_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return tpl


def _textured_frame() -> np.ndarray:
    """A non-uniform background so edge matching has something to reject."""
    frame = np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8)
    for x in range(0, FRAME_W, 24):
        cv2.line(frame, (x, 0), (x, FRAME_H), (130, 120, 110), 3)
    return frame


def _stamp(frame: np.ndarray, tpl: np.ndarray, x: int, y: int,
           alpha: float = 0.65) -> np.ndarray:
    """Alpha-composite the template onto the frame (semi-transparent)."""
    out = frame.copy()
    roi = out[y : y + tpl.shape[0], x : x + tpl.shape[1]].astype(np.float32)
    blended = roi * (1 - alpha) + tpl.astype(np.float32) * alpha
    out[y : y + tpl.shape[0], x : x + tpl.shape[1]] = blended.astype(np.uint8)
    return out


def _write_template(tmp_path: Path) -> Path:
    p = tmp_path / "wm.png"
    cv2.imwrite(str(p), _make_watermark_template())
    return p


def _remover(tmp_path: Path, **overrides) -> WatermarkRemover:
    params = WatermarkParams(
        method="template",
        template_path=_write_template(tmp_path),
        threshold=overrides.pop("threshold", 0.3),
        detect_scale=overrides.pop("detect_scale", 1.0),
        # The synthetic template is authored at full size for the FRAME_W-wide
        # test frames, so center the multi-scale search on that width.
        template_ref_width=overrides.pop("template_ref_width", FRAME_W),
        **overrides,
    )
    return WatermarkRemover(params)


def _scaled_badge(scale: float) -> np.ndarray:
    """The badge template resized by `scale` (mimics a lower-res render)."""
    tpl = _make_watermark_template()
    h, w = tpl.shape[:2]
    return cv2.resize(tpl, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _blank_frame(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    for x in range(0, width, 24):
        cv2.line(frame, (x, 0), (x, height), (130, 120, 110), 3)
    return frame


# --- detection -------------------------------------------------------------

def test_template_detect_finds_known_location(tmp_path: Path) -> None:
    rem = _remover(tmp_path)
    truth_x, truth_y = 300, 60
    frame = _stamp(_textured_frame(), _make_watermark_template(), truth_x, truth_y)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box, conf = rem._detect(gray)

    assert box is not None
    bx, by, _bw, _bh = box
    assert abs(bx - truth_x) <= 12
    assert abs(by - truth_y) <= 12
    assert conf >= rem._params.threshold


def test_multiscale_detects_smaller_badge(tmp_path: Path) -> None:
    """The real bug: output resolution (1088) < template's authored width
    (1280) → badge is smaller than the template. Multi-scale must still find
    it. Here the template is authored for FRAME_W; a 0.85x-wide frame renders
    a 0.85x badge — base_scale = frame_w/ref centers the search on it."""
    rem = _remover(tmp_path, threshold=0.3)
    scale = 0.85
    fw = int(FRAME_W * scale)
    frame = _blank_frame(fw, int(FRAME_H * scale))
    badge = _scaled_badge(scale)
    tx, ty = 120, 40
    frame[ty : ty + badge.shape[0], tx : tx + badge.shape[1]] = badge

    box, conf = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert box is not None, f"multi-scale missed the smaller badge (conf={conf})"
    bx, by, bw, _bh = box
    assert abs(bx - tx) <= 14 and abs(by - ty) <= 14
    # Box width tracks the scaled template (~0.85 x WM_W), not the raw size.
    assert abs(bw - WM_W * scale) <= WM_W * 0.2


def test_scale_lock_after_match(tmp_path: Path) -> None:
    rem = _remover(tmp_path, threshold=0.3)
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert rem._locked_w is None
    box, _conf = rem._detect(gray)
    assert box is not None
    assert rem._locked_w is not None  # winning badge width cached for steady state


def test_template_gate_rejects_absent_watermark(tmp_path: Path) -> None:
    rem = _remover(tmp_path, threshold=0.6)
    clean = _textured_frame()  # no watermark stamped

    out = rem.process(clean)
    assert np.array_equal(out, clean)  # untouched — no inpaint on a clean frame


def test_threshold_detect_in_roi_keeps_lower_frame_safe() -> None:
    params = WatermarkParams(method="threshold", threshold=0.2)
    rem = WatermarkRemover(params)
    frame = _textured_frame()
    # Bright patch in the upper band (inside default ROI).
    cv2.rectangle(frame, (260, 30), (380, 70), (255, 255, 255), -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    box, _conf = rem._detect_threshold(gray)
    assert box is not None
    _bx, by, _bw, bh = box
    # Detection must stay in the upper band, never the lower (face) region.
    assert by + bh <= FRAME_H * 0.45 + 5


# --- multi-template bank (Sprint 17) ---------------------------------------

def test_bank_includes_softened_variants() -> None:
    """The bank is each source plus `template_variants` blurred copies. Using the
    bundled template as primary isolates the variant logic (no extra backstop)."""
    import swap_cli.watermark as wm

    rem = WatermarkRemover(
        WatermarkParams(template_path=wm.bundled_template_path(), template_variants=2,
                        template_ref_width=BUNDLED_TEMPLATE_REF_WIDTH)
    )
    assert len(rem._templates) == 3  # primary + 2 variants (primary IS bundled → no backstop)
    assert all(sz == rem._tpl_size for _g, sz, _r in rem._templates)
    # The variants differ from the primary (they're blurred).
    assert not np.array_equal(rem._templates[0][0], rem._templates[1][0])


def test_bank_disabled_is_single_template() -> None:
    import swap_cli.watermark as wm

    rem = WatermarkRemover(
        WatermarkParams(template_path=wm.bundled_template_path(), template_variants=0,
                        template_ref_width=BUNDLED_TEMPLATE_REF_WIDTH)
    )
    assert len(rem._templates) == 1
    assert np.array_equal(rem._templates[0][0], rem._tpl_gray)


def test_bank_keeps_bundled_backstop_for_custom_template(tmp_path: Path) -> None:
    """A custom capture must NOT drop the known-good bundled default: the bank
    keeps both sources so a poor capture can't tank detection. Each source
    carries its own ref width (custom = capture width, bundled = 954)."""
    rem = _remover(tmp_path, template_variants=0, template_ref_width=640)
    # Two sources (custom + bundled), no blur variants.
    assert len(rem._templates) == 2
    refs = {r for _g, _s, r in rem._templates}
    assert refs == {640, BUNDLED_TEMPLATE_REF_WIDTH}  # each at its own ref width
    # Primary (custom) is first; bundled is the backstop.
    assert rem._templates[0][2] == 640


def test_bundled_primary_not_double_loaded() -> None:
    """When the primary IS the bundled default, it isn't added a second time."""
    import swap_cli.watermark as wm

    rem = WatermarkRemover(
        WatermarkParams(template_path=wm.bundled_template_path(), template_variants=0,
                        template_ref_width=BUNDLED_TEMPLATE_REF_WIDTH)
    )
    assert len(rem._templates) == 1


def test_bundled_backstop_detects_when_custom_useless(tmp_path: Path) -> None:
    """The real failure mode: a poor custom capture can't find the badge, but the
    bundled backstop in the bank still does. (User's live bug — a bad capture had
    overridden the good bundled default.)"""
    import swap_cli.watermark as wm

    # A useless custom template: random noise that won't match the real badge.
    bad = tmp_path / "bad.png"
    cv2.imwrite(str(bad), np.random.RandomState(1).randint(0, 255, (46, 208, 3), np.uint8))

    # A frame with the REAL bundled badge stamped in, at the bundled ref width.
    badge = cv2.imread(str(wm.bundled_template_path()))
    th, tw = badge.shape[:2]
    frame = np.full((568, BUNDLED_TEMPLATE_REF_WIDTH, 3), 60, np.uint8)
    x, y = 300, 120
    frame[y : y + th, x : x + tw] = badge

    rem = WatermarkRemover(
        WatermarkParams(template_path=bad, template_variants=2,
                        template_ref_width=BUNDLED_TEMPLATE_REF_WIDTH)
    )
    box, conf = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert box is not None
    assert conf >= rem._params.threshold  # caught by the bundled backstop
    assert abs(box[0] - x) <= 10 and abs(box[1] - y) <= 10


def test_corrupt_custom_falls_back_to_bundled_not_threshold(tmp_path: Path) -> None:
    """A corrupt/unreadable custom template must fall back to the bundled default
    (template method), NOT silently degrade to the brightness threshold method
    (which would explain a badge that suddenly shows in every frame)."""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a valid PNG file")
    rem = WatermarkRemover(
        WatermarkParams(template_path=corrupt, template_variants=0, template_ref_width=1088)
    )
    assert rem._method == "template"  # did NOT fall back to brightness threshold
    assert rem._templates  # bundled default loaded despite the corrupt custom
    refs = {r for _g, _s, r in rem._templates}
    assert BUNDLED_TEMPLATE_REF_WIDTH in refs


def test_bank_conf_never_below_single(tmp_path: Path) -> None:
    """Matching against the bank takes the max, so a faint badge can only score
    >= the single sharp template — never worse. (The real-clip win: the softened
    variant lifts faint frames the sharp template dips below the gate on.)"""
    # A faint, slightly-blurred badge — the appearance that sinks the sharp match.
    tpl = _make_watermark_template()
    faint = cv2.GaussianBlur(tpl, (5, 5), 0)
    frame = _stamp(_textured_frame(), faint, 300, 60, alpha=0.45)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    single = _remover(tmp_path, template_variants=0)
    bank = _remover(tmp_path, template_variants=3)
    _b1, c_single = single._detect(gray)
    _b2, c_bank = bank._detect(gray)
    assert c_bank >= c_single


def test_locked_width_set_and_reset(tmp_path: Path, monkeypatch) -> None:
    """A confident acquire locks the badge's physical width; full release clears
    it (so the next cold search re-opens the multi-scale search). The lock is a
    SIZE, not a template — every bank template still competes each frame."""
    rem = _remover(tmp_path, threshold=0.3, hold_frames=2, template_variants=2)
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    rem.process(frame)
    assert rem._locked_w is not None  # locked the badge width

    monkeypatch.setattr(rem, "_detect", lambda _g: (None, 0.0))  # force misses
    clean = _textured_frame()
    for _ in range(rem._params.hold_frames + 1):  # exceed hold → release
        rem.process(clean)
    assert rem._held_box is None
    assert rem._locked_w is None  # cleared on release


def test_bank_competes_while_locked_no_monopoly(tmp_path: Path) -> None:
    """Regression for the live bug: a bad template that locks must NOT starve the
    rest of the bank. The lock is a SIZE (`_locked_w`), not a single template
    index — so every bank template still competes each frame and a stronger match
    (e.g. the bundled backstop) can re-take the lock. The old per-template lock
    (`_locked_tpl_idx`) that caused the monopoly is gone."""
    rem = _remover(tmp_path, threshold=0.3, template_variants=2)
    assert not hasattr(rem, "_locked_tpl_idx")  # monopoly mechanism removed
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    box1, _c1 = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # acquire
    assert rem._locked_w is not None and box1 is not None
    # Still tracks the badge on the next frame with the full bank in play.
    rem._last_center = (box1[0] + box1[2] // 2, box1[1] + box1[3] // 2)
    box2, _c2 = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert box2 is not None


# --- signature safety net (Sprint 17) --------------------------------------

def _white_band(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    """Stamp a bright near-white horizontal text band (the badge signature)."""
    out = frame.copy()
    cv2.putText(out, "AI Generated", (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def test_signature_finds_white_text_band(tmp_path: Path) -> None:
    rem = _remover(tmp_path)  # template_ref_width=FRAME_W, detect_scale=1.0
    frame = _white_band(_textured_frame(), 280, 60)
    box, conf = rem._detect_signature(frame)
    assert box is not None and conf > 0
    bx, _by, bw, _bh = box
    assert 240 <= bx <= 320  # near the stamped band
    assert bw >= 60  # a band, not a speck


def test_signature_rejects_saturated_skin(tmp_path: Path) -> None:
    """A saturated (skin-toned) block must NOT trigger the net — near-white veto."""
    rem = _remover(tmp_path)
    frame = _textured_frame()
    cv2.rectangle(frame, (280, 60), (440, 95), (60, 120, 200), -1)  # warm/saturated
    box, _conf = rem._detect_signature(frame)
    assert box is None


def test_signature_rejects_solid_bright_block(tmp_path: Path) -> None:
    """A solid white wall/blind block is too DENSE (≈100% fill) to be text."""
    rem = _remover(tmp_path)
    frame = _textured_frame()
    cv2.rectangle(frame, (280, 60), (440, 95), (255, 255, 255), -1)  # solid fill
    box, _conf = rem._detect_signature(frame)
    assert box is None


def test_signature_rejects_wrong_aspect(tmp_path: Path) -> None:
    """A near-square bright blob fails the horizontal-band aspect check."""
    rem = _remover(tmp_path)
    frame = _textured_frame()
    # A small white square with a hole so density is text-like but aspect ~1.
    cv2.rectangle(frame, (300, 60), (340, 100), (255, 255, 255), 3)
    box, _conf = rem._detect_signature(frame)
    assert box is None


def test_signature_local_gate_ignores_far_band(tmp_path: Path) -> None:
    """The net only accepts bands NEAR the anchor — a real badge's neighbourhood.
    A band far from `near` is skipped (this is what stops it grabbing a collar /
    bright strip across the frame, the 0%-on-badge global failure)."""
    rem = _remover(tmp_path)
    frame = _white_band(_textured_frame(), 480, 60)  # band on the right
    # Anchor on the LEFT, small radius → the right-side band is out of range.
    box, _conf = rem._detect_signature(frame, near=(120, 76), radius=80)
    assert box is None
    # Anchored ON the band → found.
    box2, conf2 = rem._detect_signature(frame, near=(540, 76), radius=120)
    assert box2 is not None and conf2 > 0


def test_signature_net_bridges_when_template_misses(tmp_path: Path, monkeypatch) -> None:
    """Once the badge has been seen (anchor set), the net sustains coverage on a
    badge the template can no longer find — past the coast window, so this is the
    net working, not track-and-hold."""
    rem = _remover(tmp_path, threshold=0.3, signature_fallback=True)
    rem.process(_stamp(_textured_frame(), _make_watermark_template(), 280, 60))
    assert rem._sig_center is not None  # anchor primed by the first acquire

    monkeypatch.setattr(rem, "_detect", lambda _g: (None, 0.0))  # template blind
    frame = _white_band(_textured_frame(), 280, 60)  # badge band still there
    for _ in range(rem._params.hold_frames + 3):  # outlast coast → only the net holds
        rem.process(frame)
    assert rem._held_box is not None
    assert rem._gate_mode == "signature"


def test_signature_net_needs_anchor(tmp_path: Path, monkeypatch) -> None:
    """Cold (no badge ever seen) → no anchor → the net stays off, so it can't
    fire on a random bright band before the template has ever locked."""
    rem = _remover(tmp_path, threshold=0.3, signature_fallback=True)
    monkeypatch.setattr(rem, "_detect", lambda _g: (None, 0.0))
    out = rem.process(_white_band(_textured_frame(), 280, 60))
    assert rem._held_box is None  # no anchor yet → net silent
    assert np.array_equal(out, _white_band(_textured_frame(), 280, 60))


def test_signature_fallback_disabled(tmp_path: Path, monkeypatch) -> None:
    """With the net off, coverage expires once the coast window ends — nothing
    bridges the template miss (contrast with the bridge test above)."""
    rem = _remover(tmp_path, threshold=0.3, signature_fallback=False)
    rem.process(_stamp(_textured_frame(), _make_watermark_template(), 280, 60))
    monkeypatch.setattr(rem, "_detect", lambda _g: (None, 0.0))
    frame = _white_band(_textured_frame(), 280, 60)
    for _ in range(rem._params.hold_frames + 3):  # outlast coast → release
        rem.process(frame)
    assert rem._held_box is None
    assert rem._gate_mode != "signature"


# --- mask + inpaint --------------------------------------------------------

def test_build_mask_shape_and_dilation(tmp_path: Path) -> None:
    rem = _remover(tmp_path, dilation=6, feather=2)
    mask = rem._build_mask((FRAME_H, FRAME_W), (300, 60, WM_W, WM_H))

    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert mask.shape == (FRAME_H, FRAME_W)
    assert mask.max() == 255
    # Dilation grows the white region beyond the raw box area.
    assert int((mask > 0).sum()) > WM_W * WM_H
    # Feather produces non-binary edge values.
    assert np.any((mask > 0) & (mask < 255))


def test_inpaint_reduces_watermark_energy(tmp_path: Path) -> None:
    # Pure-inpaint path (temporal disabled): one frame, badge filled by guess.
    rem = _remover(tmp_path, temporal=False)
    clean = _textured_frame()
    tpl = _make_watermark_template()
    x, y = 300, 60
    dirty = _stamp(clean, tpl, x, y)

    cleaned = rem.process(dirty)

    region = (slice(y, y + WM_H), slice(x, x + WM_W))
    before = np.abs(dirty[region].astype(int) - clean[region].astype(int)).mean()
    after = np.abs(cleaned[region].astype(int) - clean[region].astype(int)).mean()
    assert after < before  # watermark energy materially reduced


# --- temporal recovery -----------------------------------------------------

def test_temporal_recovery_restores_real_pixels(tmp_path: Path) -> None:
    """The badge moves, so a covered location was clean a moment ago →
    restored from real pixels (near-zero error), far better than inpaint."""
    rem = _remover(tmp_path, threshold=0.3)  # temporal on by default
    clean = _textured_frame()
    tpl = _make_watermark_template()

    # Frame 1: badge at A. Frame 2: badge moved to B (A is now revealed and
    # learned into the clean plate; B was clean in frame 1).
    pos_a, pos_b = (120, 60), (400, 60)
    rem.process(_stamp(clean, tpl, *pos_a))
    out = rem.process(_stamp(clean, tpl, *pos_b))

    bx, by = pos_b
    region = (slice(by, by + WM_H), slice(bx, bx + WM_W))
    temporal_err = np.abs(out[region].astype(int) - clean[region].astype(int)).mean()

    # Compare with what pure inpaint would leave on that frame.
    m = np.zeros((FRAME_H, FRAME_W), np.uint8)
    m[by : by + WM_H, bx : bx + WM_W] = 255
    ip = cv2.inpaint(_stamp(clean, tpl, *pos_b), m, 3, cv2.INPAINT_TELEA)
    inpaint_err = np.abs(ip[region].astype(int) - clean[region].astype(int)).mean()

    assert temporal_err < 2.0  # essentially the real background restored
    assert temporal_err < inpaint_err  # and materially cleaner than a guess


def test_temporal_plate_updates_where_uncovered(tmp_path: Path) -> None:
    rem = _remover(tmp_path, threshold=0.3)
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    rem.process(frame)
    assert rem._clean_plate is not None and rem._age is not None
    # A corner far from the badge must be 'fresh' and match the live frame.
    assert rem._age[10, 10] == 0
    assert np.array_equal(rem._clean_plate[10, 10], frame[10, 10])


def test_stale_falls_back_to_inpaint(tmp_path: Path) -> None:
    rem = _remover(tmp_path, threshold=0.3, temporal_max_stale=3)
    clean = _textured_frame()
    dirty = _stamp(clean, _make_watermark_template(), 300, 60)  # badge parked
    for _ in range(6):  # hold past max_stale
        rem.process(dirty)
    assert "inpaint" in rem._last_fill  # fallback engaged when stuck


def test_buffers_reinit_on_frame_size_change(tmp_path: Path) -> None:
    rem = _remover(tmp_path, threshold=0.3)
    rem.process(_stamp(_textured_frame(), _make_watermark_template(), 300, 60))
    assert rem._clean_plate.shape[:2] == (FRAME_H, FRAME_W)
    # A differently-sized frame must rebuild the buffers, not index-error.
    big = np.full((FRAME_H * 2, FRAME_W * 2, 3), 90, np.uint8)
    out = rem._restore(big, (100, 100, WM_W, WM_H))
    assert out.shape == big.shape
    assert rem._clean_plate.shape[:2] == (FRAME_H * 2, FRAME_W * 2)


def test_slow_slide_stays_temporal(tmp_path: Path) -> None:
    """Regression for the live failure: a slowly-sliding badge over a static
    background must keep using the clean plate (temporal), not age out into
    inpaint. Previously stale=100%; now stale≈0 after the plate fills."""
    rem = _remover(tmp_path, threshold=0.3)
    clean = _textured_frame()
    tpl = _make_watermark_template()

    # Warm-up: roam widely so the y=60 band gets revealed into the plate.
    span = FRAME_W - WM_W - 20
    for i in range(60):
        x = 20 + int((span / 2) * (1 - np.cos(i * 0.2)))
        rem.process(_stamp(clean, tpl, x, 60))

    # Now a slow continuous slide over the already-seen band.
    out, x = None, 0
    for i in range(8):
        x = 60 + i * 4
        out = rem.process(_stamp(clean, tpl, x, 60))

    region = (slice(60, 60 + WM_H), slice(x, x + WM_W))
    err = np.abs(out[region].astype(int) - clean[region].astype(int)).mean()
    assert rem._last_fill == "temporal"  # real pixels, no inpaint
    assert rem._last_stale_frac < 0.1
    assert err < 4.0  # background restored, not a smear


def test_track_and_hold_bridges_misses(tmp_path: Path, monkeypatch) -> None:
    """A few no-match frames must NOT flash the badge back — the last
    confident box is held for up to hold_frames."""
    rem = _remover(tmp_path, threshold=0.3, hold_frames=4)
    rem.process(_stamp(_textured_frame(), _make_watermark_template(), 300, 60))
    assert rem._held_box is not None
    held = rem._held_box

    monkeypatch.setattr(rem, "_detect", lambda _g: (None, 0.0))  # force misses
    clean = _textured_frame()
    for _ in range(4):  # within the hold window → still coasting
        rem.process(clean)
        assert rem._held_box == held
    rem.process(clean)  # one miss past hold_frames → give up
    assert rem._held_box is None


def test_hysteresis_maintains_through_dip(tmp_path: Path, monkeypatch) -> None:
    """Once acquired, a confidence dip between maintain and acquire keeps the
    lock (no flicker). This is the fix for 'covers some, doesn't cover some'."""
    rem = _remover(tmp_path, threshold=0.5, maintain_threshold=0.38)
    box = (300, 60, WM_W, WM_H)

    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.80))  # strong → acquire
    rem.process(_textured_frame())
    assert rem._held_box == box
    assert rem._gate_mode == "acquire"

    # Dip to 0.45 (below acquire 0.50, above maintain 0.38) → still tracked.
    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.45))
    rem.process(_textured_frame())
    assert rem._held_box == box
    assert rem._gate_mode == "maintain"


def test_maintain_tracks_low_conf_match(tmp_path: Path, monkeypatch) -> None:
    """Once tracking, follow the best match down to the maintain floor instead
    of coasting a stale box — fixes the occasional 'missed' badge on deep dips."""
    rem = _remover(tmp_path, threshold=0.5, maintain_threshold=0.28)
    acquired = (300, 60, WM_W, WM_H)
    monkeypatch.setattr(rem, "_detect", lambda _g: (acquired, 0.80))  # acquire
    rem.process(_textured_frame())
    assert rem._held_box == acquired

    # Badge moved; low-conf (0.30 ≥ 0.28) match at the NEW spot → follow it.
    moved = (450, 200, WM_W, WM_H)
    monkeypatch.setattr(rem, "_detect", lambda _g: (moved, 0.30))
    rem.process(_textured_frame())
    assert rem._held_box == moved  # tracked, not coasting at the old box

    # Below the floor (0.20 < 0.28) → coast (hold the last box).
    monkeypatch.setattr(rem, "_detect", lambda _g: ((10, 10, WM_W, WM_H), 0.20))
    rem.process(_textured_frame())
    assert rem._held_box == moved  # unchanged — held, not chasing noise


def test_no_false_acquire_when_cold(tmp_path: Path, monkeypatch) -> None:
    """A 0.45 candidate from cold must NOT acquire (below the acquire gate),
    so spurious low matches can't start a false lock."""
    rem = _remover(tmp_path, threshold=0.5, maintain_threshold=0.38)
    box = (300, 60, WM_W, WM_H)
    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.45))
    frame = _textured_frame()
    out = rem.process(frame)
    assert rem._held_box is None
    assert np.array_equal(out, frame)  # nothing removed


def test_footprint_covers_badge_with_margin(tmp_path: Path) -> None:
    rem = _remover(tmp_path, dilation=8)
    mask = rem._build_mask((FRAME_H, FRAME_W), (300, 60, WM_W, WM_H))
    # Dilation 8 covers several px outside the raw box on every side.
    assert mask[60, 300 - 6] > 0  # left of the box
    assert mask[60 + WM_H + 6, 300 + WM_W // 2] > 0  # below the box


def test_coast_expands_footprint(tmp_path: Path) -> None:
    """While coasting (held box through misses) the footprint grows with the
    coast count, so a badge drifting during the miss stays covered."""
    rem = _remover(tmp_path, footprint_pad_frac=0.30, dilation=8,
                   coast_expand_frac=0.06, coast_expand_max_frac=0.40)
    box = (300, 200, 200, 50)
    rem._hold_count = 0
    x0a, y0a, x1a, y1a = rem._padded_box((FRAME_H, FRAME_W), box)
    rem._hold_count = 5  # coasting for 5 frames
    x0b, y0b, x1b, y1b = rem._padded_box((FRAME_H, FRAME_W), box)
    # The coasted footprint is strictly larger on every side.
    assert x0b < x0a and y0b < y0a and x1b > x1a and y1b > y1a


def test_coast_expansion_is_capped(tmp_path: Path) -> None:
    """Coast growth saturates at coast_expand_max_frac — it can't grow forever."""
    rem = _remover(tmp_path, footprint_pad_frac=0.0, dilation=0,
                   coast_expand_frac=0.06, coast_expand_max_frac=0.40)
    box = (300, 200, 200, 50)
    rem._hold_count = 100  # would be 6.0x without the cap
    x0, _y0, x1, _y1 = rem._padded_box((FRAME_H, FRAME_W), box)
    pad_x = round(0.40 * 200)  # capped fraction
    assert (x1 - x0) == 200 + 2 * pad_x


def test_default_hold_frames_is_short(tmp_path: Path) -> None:
    """Coast is kept short on purpose: the badge can jump anywhere at any time,
    so a long hold would cover a stale spot while the real badge shows elsewhere.
    The bank + signature net provide coverage, not a long coast."""
    rem = _remover(tmp_path)
    assert rem._params.hold_frames == 5


def test_padded_box_extends_beyond_match(tmp_path: Path) -> None:
    """The removal footprint must reach well past the matched box so the
    translucent pill (wider than the text strokes) is fully covered."""
    rem = _remover(tmp_path, footprint_pad_frac=0.14, dilation=8)
    box = (300, 60, WM_W, WM_H)  # WM_W=120
    x0, y0, x1, y1 = rem._padded_box((FRAME_H, FRAME_W), box)
    pad_x = max(8, round(0.14 * WM_W))  # ~17px
    assert x0 == 300 - pad_x and x1 == 300 + WM_W + pad_x
    assert pad_x > 8  # proportional pad beat the flat dilation
    assert y0 < 60 and y1 > 60 + WM_H


def test_footprint_absorbs_location_error(tmp_path: Path) -> None:
    """At low confidence the match lands a few px off; the padded footprint
    (0.30) must still cover the whole pill so no edge peeks out / leaks."""
    rem = _remover(tmp_path, footprint_pad_frac=0.30, dilation=8)
    box_w, box_h = 200, 57
    # Matched box (tight, on the text strokes).
    bx, by = 400, 200
    x0, y0, x1, y1 = rem._padded_box((FRAME_H, FRAME_W), (bx, by, box_w, box_h))
    # The real pill is wider than the strokes AND the match is offset ~30px.
    pill_overhang, offset = 16, 30
    pill_x0 = bx - pill_overhang - offset
    pill_x1 = bx + box_w + pill_overhang - offset
    # Footprint must fully contain the offset, oversized pill.
    assert x0 <= pill_x0 and x1 >= pill_x1
    assert y0 < by and y1 > by + box_h


def _text_template(tmp_path: Path) -> Path:
    img = np.zeros((52, 200), np.uint8)
    cv2.putText(img, "* AI Generated", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    p = tmp_path / "txt.png"
    cv2.imwrite(str(p), np.dstack([img] * 3))
    return p


def _wide_scene(badge_xy, badge_alpha, blinds_xy=None):
    """1088-wide frame with slats; a (possibly faint) badge, optional non-text
    'blinds' distractor."""
    h, w = 624, 1088
    f = np.full((h, w), 120, np.uint8)
    for xx in range(0, w, 24):
        cv2.line(f, (xx, 0), (xx, h), 178, 6)
    bw, bh = 200, 52
    txt = np.zeros((bh, bw), np.uint8)
    cv2.putText(txt, "* AI Generated", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    bx, by = badge_xy
    roi = f[by : by + bh, bx : bx + bw]
    m = txt > 0
    roi[m] = (roi[m] * (1 - badge_alpha) + 235 * badge_alpha).astype(np.uint8)
    if blinds_xy is not None:
        dx, dy = blinds_xy
        for yy in range(dy, dy + bh, 7):
            cv2.line(f, (dx, yy), (dx + bw, yy), 235, 3)
    return f


def test_local_search_ignores_far_distractor(tmp_path: Path) -> None:
    """A badge still tracked within the maintain band must win over a far
    background look-alike (the 'full badge showed' wrong-location misses). With
    the safer gate we no longer track a sub-0.45 badge — that's released by
    design — so this exercises a confidently-tracked (but dipped) badge."""
    rem = WatermarkRemover(
        WatermarkParams(template_path=_text_template(tmp_path), template_ref_width=1088,
                        detect_scale=0.6)
    )
    badge = (300, 250)
    rem._detect(_wide_scene(badge, 0.7))  # acquire (strong) → locks scale
    rem._last_center = (badge[0] + 100, badge[1] + 26)  # tracking near the badge

    frame = _wide_scene(badge, 0.5, blinds_xy=(800, 470))  # tracked badge + far distractor
    box, _conf = rem._detect(frame)
    assert box is not None
    cx = box[0] + box[2] // 2
    assert abs(cx - (badge[0] + 100)) < 80  # stayed on the badge, not 500px away


def test_safer_gate_releases_drifted_lock(tmp_path: Path, monkeypatch) -> None:
    """Safer gate: after acquiring, sustained sub-maintain (drifted) matches must
    RELEASE the lock rather than coast on it — so reconstruct stops filling the
    wrong spot. A match of 0.35 (above the old 0.28 gate, below the new 0.45)
    is the drift case that used to paint a misplaced patch."""
    rem = _remover(tmp_path, threshold=0.5)  # default maintain=0.45, hold=5
    box = (300, 60, WM_W, WM_H)
    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.8))  # confident acquire
    rem.process(_textured_frame())
    assert rem._held_box is not None

    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.35))  # drifted/weak
    for _ in range(rem._params.hold_frames + 1):
        rem.process(_textured_frame())
    assert rem._held_box is None  # released — no fill on the weak/drifted lock


def test_reacquires_jumped_badge_immediately(tmp_path: Path) -> None:
    """The badge jumps far to a spot scoring between maintain and acquire, with
    nothing at the old position → re-acquire it THIS frame (don't coast on the
    stale box, which is what briefly showed the full pill)."""
    rem = WatermarkRemover(
        WatermarkParams(template_path=_text_template(tmp_path), template_ref_width=1088,
                        detect_scale=0.6)
    )
    a = (200, 250)
    rem._detect(_wide_scene(a, 0.7))  # acquire + lock at A
    rem._last_center = (a[0] + 100, a[1] + 26)
    # Badge now ONLY at a far B, faint (global conf in (0.28, 0.50)); A is empty.
    box, conf = rem._detect(_wide_scene((820, 470), 0.25))
    assert 0.28 <= conf < 0.5  # the moderate-confidence window that used to coast
    assert abs((box[0] + box[2] // 2) - (820 + 100)) < 90  # followed to B, not stale A


def test_confident_far_match_reacquires(tmp_path: Path) -> None:
    rem = WatermarkRemover(
        WatermarkParams(template_path=_text_template(tmp_path), template_ref_width=1088,
                        detect_scale=0.6)
    )
    rem._detect(_wide_scene((300, 250), 0.7))
    rem._last_center = (400, 276)
    # A strong badge far away (genuine move) must be followed, not suppressed.
    box, conf = rem._detect(_wide_scene((760, 430), 0.7))
    assert conf >= 0.5
    assert abs((box[0] + box[2] // 2) - (760 + 100)) < 80


def test_motion_aware_inpaints_when_scene_moves(tmp_path: Path) -> None:
    """Over a moving region the fill must switch from the (stale) clean plate to
    inpaint, so we don't show a ghost."""
    rem = _remover(tmp_path, threshold=0.3)
    badge = (300, 60, WM_W, WM_H)

    static = _textured_frame()
    rem._restore(static, badge)  # primes _prev_bgr + plate
    rem._restore(static, badge)  # static surroundings → temporal
    assert "temporal" in rem._last_fill

    moved = _textured_frame()
    cv2.rectangle(moved, (340, 40), (520, 140), (240, 230, 220), -1)  # big change nearby
    rem._restore(moved, badge)
    assert rem._last_fill == "inpaint(motion)"


def test_blur_removal_smears_badge(tmp_path: Path) -> None:
    """removal='blur' replaces the badge footprint with a blurred patch — the
    region's high-frequency detail (Laplacian variance) drops sharply and the
    output differs from the input there."""
    rem = _remover(tmp_path, threshold=0.3, removal="blur")
    bx, by = 300, 60
    badge = (bx, by, WM_W, WM_H)
    # Sharp, high-detail content under the badge so blur has something to smear.
    frame = _textured_frame()
    cv2.putText(frame, "AI GENERATED", (bx + 4, by + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    out = rem._blur_region(frame, badge)
    assert rem._last_fill == "blur"

    reg = (slice(by, by + WM_H), slice(bx, bx + WM_W))

    def detail(img: np.ndarray) -> float:
        g = cv2.cvtColor(img[reg], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    assert detail(out) < detail(frame) * 0.5  # markedly blurrier
    # The footprint actually changed (not a pass-through).
    assert int(np.abs(out[reg].astype(int) - frame[reg].astype(int)).sum()) > 0


def test_blur_mode_sets_fill_label_via_process(tmp_path: Path) -> None:
    """A full process() pass in blur mode routes to the blur fill once the badge
    is acquired (no temporal/inpaint)."""
    rem = _remover(tmp_path, threshold=0.3, removal="blur")
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    rem.process(frame)
    assert rem._last_fill == "blur"


def test_scale_locks_only_on_confident_match(tmp_path: Path) -> None:
    """A marginal match must not lock the badge width (the 0.70 drift that left
    the badge undersized); only a confident (>= acquire) match locks it."""
    rem = _remover(tmp_path, threshold=0.5)
    # Clean frame → best conf below the acquire gate → no width lock.
    rem._detect(cv2.cvtColor(_textured_frame(), cv2.COLOR_BGR2GRAY))
    assert rem._locked_w is None
    # Badge frame → confident → width locks.
    frame = _stamp(_textured_frame(), _make_watermark_template(), 200, 60)
    _box, conf = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert conf >= 0.5 and rem._locked_w is not None


def test_scale_persists_through_coast_resets_on_release(
    tmp_path: Path, monkeypatch
) -> None:
    rem = _remover(tmp_path, threshold=0.5, maintain_threshold=0.38, hold_frames=3)
    box = (300, 60, WM_W, WM_H)
    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.8))  # acquire
    rem.process(_textured_frame())
    rem._locked_w = float(WM_W)  # a locked badge width

    monkeypatch.setattr(rem, "_detect", lambda _g: (box, 0.1))  # misses (coast)
    for _ in range(3):  # within hold → width must persist
        rem.process(_textured_frame())
        assert rem._held_box is not None
        assert rem._locked_w == float(WM_W)
    rem.process(_textured_frame())  # exceed hold → full release
    assert rem._held_box is None
    assert rem._locked_w is None  # only now re-open the multi-scale search


def test_capture_autotighten_shrinks_loose_box() -> None:
    from swap_cli.display import _tighten_to_badge

    canvas = np.full((140, 460, 3), 90, np.uint8)
    badge = _make_watermark_template()  # WM_H x WM_W
    canvas[50 : 50 + WM_H, 150 : 150 + WM_W] = badge
    tight = _tighten_to_badge(canvas)
    # Shrunk well below the loose canvas, roughly the badge size.
    assert tight.shape[1] < canvas.shape[1] - 100
    assert tight.shape[0] < canvas.shape[0] - 40
    assert tight.shape[1] >= WM_W - 20  # still covers the strokes


# --- resilience ------------------------------------------------------------

def test_process_never_raises(tmp_path: Path, capsys) -> None:
    rem = _remover(tmp_path)

    def _boom(_gray):
        raise RuntimeError("detector exploded")

    rem._detect = _boom  # type: ignore[assignment]
    frame = _textured_frame()
    out = rem.process(frame)

    assert np.array_equal(out, frame)  # original returned on failure
    # Second failure must NOT print again (warned-once guard).
    rem.process(frame)
    assert capsys.readouterr().out.count("disabled this frame") == 1


def test_diagnostics_active_logged_once_and_throttled(tmp_path: Path, capsys) -> None:
    """No silent failure (the original bug) and no per-frame flood."""
    rem = _remover(tmp_path, threshold=0.5)
    frame = _textured_frame()  # no badge → genuine 'no match' every frame
    for _ in range(45):
        rem.process(frame)
    out = capsys.readouterr().out
    # 'active' banner exactly once (first frame).
    assert out.count("[watermark] active") == 1
    # 'no match' diagnostics throttled to ~1/20 frames, not 45 lines.
    no_match = out.count("no match")
    assert 1 <= no_match <= 4
    assert "press W" in out  # actionable hint surfaced


def test_diagnostics_reports_match(tmp_path: Path, capsys) -> None:
    rem = _remover(tmp_path, threshold=0.3)
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
    for _ in range(20):
        rem.process(frame)
    out = capsys.readouterr().out
    assert "[watermark] match conf=" in out


def test_process_pass_through_when_below_threshold(tmp_path: Path) -> None:
    # No badge in the frame → confidence below gate → frame returned as-is.
    rem = _remover(tmp_path, threshold=0.5)
    frame = _textured_frame()
    out = rem.process(frame)
    assert np.array_equal(out, frame)


# --- capture hot-reload -----------------------------------------------------

def test_capture_hot_reloads_remover(monkeypatch, tmp_path: Path) -> None:
    """Pressing W mid-session must swap in the captured template live, not
    only on the next run (the Sprint 15b bug)."""
    from unittest.mock import MagicMock

    from swap_cli import config as cfgmod
    from swap_cli import display as disp_mod
    from swap_cli.display import Display

    # Isolate config + template path to tmp.
    monkeypatch.setattr(cfgmod, "config_path", lambda: tmp_path / "config.toml")
    dest = tmp_path / "captured.png"
    monkeypatch.setattr(disp_mod, "default_watermark_template_path", lambda: dest)

    # A 1088-wide raw frame with the badge stamped at a known spot.
    raw = _blank_frame(1088, 624)
    badge = _make_watermark_template()
    bx, by = 500, 120
    raw[by : by + badge.shape[0], bx : bx + badge.shape[1]] = badge

    disp = Display(track=MagicMock(), watermark=None)
    disp._latest_raw_bgr = raw

    # User drags a box around the badge.
    monkeypatch.setattr(
        cv2, "selectROI",
        lambda *a, **k: (bx, by, badge.shape[1], badge.shape[0]),
    )
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)

    disp._capture_watermark_template()

    # Hot-reloaded: a live remover now exists, built from the captured crop.
    assert disp._watermark is not None
    assert disp._watermark._tpl_size == (badge.shape[1], badge.shape[0])
    # Config persisted the template + the capture-frame width.
    cfg = cfgmod.load()
    assert cfg.watermark_template == str(dest)
    assert cfg.watermark_template_width == 1088
    # And the remover centers its scale search on that width.
    assert disp._watermark._params.template_ref_width == 1088


def test_capture_rejects_tiny_selection(monkeypatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from swap_cli import config as cfgmod
    from swap_cli import display as disp_mod
    from swap_cli.display import Display

    monkeypatch.setattr(cfgmod, "config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(disp_mod, "default_watermark_template_path",
                        lambda: tmp_path / "captured.png")
    disp = Display(track=MagicMock(), watermark=None)
    disp._latest_raw_bgr = _blank_frame(1088, 624)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (10, 10, 5, 5))  # too small
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)

    disp._capture_watermark_template()
    assert disp._watermark is None  # nothing swapped in
    assert not (tmp_path / "captured.png").exists()  # nothing saved


# --- factory ---------------------------------------------------------------

def test_from_config_returns_none_when_disabled() -> None:
    from swap_cli import config as cfgmod

    cfg = cfgmod.Config(None, None, None, None)
    assert WatermarkRemover.from_config(cfg, enabled=False) is None


def test_from_config_returns_none_when_no_template(monkeypatch) -> None:
    import swap_cli.watermark as wm
    from swap_cli import config as cfgmod

    # Force "no bundled default" so this exercises the truly-empty path.
    monkeypatch.setattr(wm, "bundled_template_path", lambda: None)
    cfg = cfgmod.Config(None, None, None, None, remove_watermark=True)
    assert WatermarkRemover.from_config(cfg, enabled=True) is None


def test_from_config_falls_back_to_bundled_template(monkeypatch, tmp_path: Path) -> None:
    import swap_cli.watermark as wm
    from swap_cli import config as cfgmod

    # No user template, but a bundled default exists → removal works.
    bundled = _write_template(tmp_path)
    monkeypatch.setattr(wm, "bundled_template_path", lambda: bundled)
    cfg = cfgmod.Config(None, None, None, None, remove_watermark=True)
    rem = WatermarkRemover.from_config(cfg, enabled=True)
    assert isinstance(rem, WatermarkRemover)
    assert rem._tpl_gray is not None


def test_bundled_template_path_returns_none_when_absent(monkeypatch) -> None:
    import swap_cli.watermark as wm

    monkeypatch.setattr(wm, "_BUNDLED_TEMPLATE", Path("/nonexistent/wm.png"))
    assert wm.bundled_template_path() is None


def test_from_config_builds_remover_with_template(tmp_path: Path) -> None:
    from swap_cli import config as cfgmod

    tpl = _write_template(tmp_path)
    cfg = cfgmod.Config(
        None, None, None, None,
        remove_watermark=True,
        watermark_template=str(tpl),
    )
    rem = WatermarkRemover.from_config(cfg, enabled=True)
    assert isinstance(rem, WatermarkRemover)
    assert rem._tpl_gray is not None


def test_bundled_default_template_ships_and_detects() -> None:
    """The packaged default template exists, is a valid image, and the
    remover can locate it when placed in a native-size frame."""
    import swap_cli.watermark as wm

    p = wm.bundled_template_path()
    assert p is not None and p.is_file(), "bundled watermark_default.png missing"
    tpl = cv2.imread(str(p))
    assert tpl is not None and tpl.size > 0

    # Drop the badge into a blank 1280x720 frame and confirm detection.
    frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
    th, tw = tpl.shape[:2]
    x, y = 500, 120
    frame[y : y + th, x : x + tw] = tpl
    rem = WatermarkRemover(WatermarkParams(template_path=p))
    box, conf = rem._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert box is not None
    assert abs(box[0] - x) <= 8 and abs(box[1] - y) <= 8
    assert conf >= WatermarkParams().threshold


def test_roi_to_px_converts_fractions() -> None:
    assert _roi_to_px((0.0, 0.0, 1.0, 0.5), 360, 640) == (0, 0, 640, 180)
    assert _roi_to_px((0.5, 0.5, 0.5, 0.5), 360, 640) == (320, 180, 320, 180)


# --- graceful shutdown (Sprint 16) -----------------------------------------

def test_display_ends_session_on_stream_error(monkeypatch) -> None:
    """When the Decart remote track raises (connection drop), Display ends the
    session (calls on_quit) instead of hanging — and doesn't propagate."""
    import asyncio
    from unittest.mock import MagicMock

    from swap_cli import display as disp_mod
    from swap_cli.display import Display

    for fn in ("namedWindow", "resizeWindow", "imshow", "waitKey",
               "destroyAllWindows", "setWindowProperty"):
        monkeypatch.setattr(disp_mod.cv2, fn, lambda *a, **k: 0)

    quit_called = {"v": False}
    track = MagicMock()

    async def _boom():
        raise RuntimeError("")  # empty message, like aiortc's stream-ended

    track.recv = _boom
    disp = Display(track=track, on_quit=lambda: quit_called.__setitem__("v", True))
    asyncio.run(disp._loop())  # must NOT raise

    assert quit_called["v"] is True
    assert disp._stopped.is_set()


def test_suppress_pattern_catches_cancelled_error() -> None:
    """Guards the runtime fix: suppress(Exception) does NOT catch
    asyncio.CancelledError (a BaseException) — the tuple form does."""
    import asyncio
    from contextlib import suppress

    with pytest.raises(asyncio.CancelledError), suppress(Exception):
        raise asyncio.CancelledError()

    with suppress(Exception, asyncio.CancelledError):
        raise asyncio.CancelledError()  # swallowed — no raise


def test_bundled_template_exists_and_loads() -> None:
    """The packaged default badge template ships and is a real image, so
    out-of-box removal needs no capture step."""
    path = bundled_template_path()
    assert path is not None and path.is_file()
    img = cv2.imread(str(path))
    assert img is not None and img.size > 0


def test_bundled_template_uses_its_own_ref_width() -> None:
    """When no template is captured, the remover falls back to the bundled
    default and uses BUNDLED_TEMPLATE_REF_WIDTH (not the 1280 custom default),
    so the scale search centers on the real badge size."""
    from swap_cli import config as cfgmod

    base = cfgmod.Config("L", "dct", None, None)
    cfg = base.__class__(**{**base.__dict__, "remove_watermark": True, "watermark_template": None})
    rem = WatermarkRemover.from_config(cfg, enabled=True)
    assert rem is not None
    assert rem._params.template_ref_width == BUNDLED_TEMPLATE_REF_WIDTH


def test_display_windowless_skips_cv2_window(monkeypatch) -> None:
    """show_window=False (macOS GUI): the loop opens NO cv2 window (the GUI pumps
    it from the main thread) and exposes frames via latest_frame()."""
    import asyncio
    from unittest.mock import MagicMock

    from swap_cli import display as disp_mod
    from swap_cli.display import Display

    called = {"w": False}
    for fn in ("namedWindow", "resizeWindow", "imshow", "waitKey",
               "destroyAllWindows", "setWindowProperty"):
        monkeypatch.setattr(disp_mod.cv2, fn,
                            lambda *a, **k: called.__setitem__("w", True))

    arr = np.zeros((12, 16, 3), np.uint8)
    arr[4:8, 4:12] = 200
    disp = Display(track=MagicMock(), show_window=False)
    calls = {"n": 0}

    async def _recv():
        calls["n"] += 1
        if calls["n"] >= 2:
            disp._stopped.set()
        frame = MagicMock()
        frame.to_ndarray = lambda format: arr  # to_ndarray(format=...) keyword
        return frame

    disp._track.recv = _recv
    asyncio.run(disp._loop())  # must not raise
    assert called["w"] is False  # no cv2 window touched off the main thread
    assert disp.latest_frame() is not None
    assert disp.latest_raw_frame() is not None


def test_capture_watermark_with_roi_writes_template(tmp_path: Path, monkeypatch) -> None:
    """The main-thread W-key path: capture_watermark(roi) crops the RAW frame,
    writes the template, and updates config — no cv2 window needed."""
    from unittest.mock import MagicMock

    from swap_cli import config as cfgmod
    from swap_cli import display as disp_mod
    from swap_cli.display import Display

    monkeypatch.setattr(cfgmod, "config_path", lambda: tmp_path / "config.toml")
    dest = tmp_path / "wm.png"
    monkeypatch.setattr(disp_mod, "default_watermark_template_path", lambda: dest)

    frame = _textured_frame()
    cv2.putText(frame, "AI GENERATED", (310, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    disp = Display(track=MagicMock(), show_window=False)
    disp._latest_raw_bgr = frame
    disp.capture_watermark((300, 60, 180, 40))

    assert dest.is_file()
    cfg = cfgmod.load()
    assert cfg.watermark_template == str(dest)
    assert cfg.watermark_template_width == frame.shape[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
