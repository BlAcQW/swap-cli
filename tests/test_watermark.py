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
    WatermarkParams,
    WatermarkRemover,
    _roi_to_px,
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
    assert rem._locked_scale is None
    box, _conf = rem._detect(gray)
    assert box is not None
    assert rem._locked_scale is not None  # winning scale cached for steady state


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
    rem = _remover(tmp_path)
    clean = _textured_frame()
    tpl = _make_watermark_template()
    x, y = 300, 60
    dirty = _stamp(clean, tpl, x, y)

    cleaned = rem.process(dirty)

    region = (slice(y, y + WM_H), slice(x, x + WM_W))
    before = np.abs(dirty[region].astype(int) - clean[region].astype(int)).mean()
    after = np.abs(cleaned[region].astype(int) - clean[region].astype(int)).mean()
    assert after < before  # watermark energy materially reduced


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
    rem = _remover(tmp_path, threshold=0.99)  # gate guarantees 'no match'
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
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
    rem = _remover(tmp_path, threshold=0.99)  # impossible gate
    frame = _stamp(_textured_frame(), _make_watermark_template(), 300, 60)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
