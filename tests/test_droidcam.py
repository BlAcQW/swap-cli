"""DroidCam sink: header protocol + YUYV conversion.

The Windows handles (file mapping, events) can't be exercised off-Windows, so
these cover the parts that carry the real risk of being silently wrong: the
header validation that decides whether an app is consuming the camera, and the
pixel packing the driver renders.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The sink imports cv2/numpy lazily inside functions; a sibling test stubs cv2,
# so evict a stub and pin the real ones (same dance as test_providers).
for _name, _marker in (("cv2", "imencode"),):
    _stub = sys.modules.get(_name)
    if _stub is not None and not hasattr(_stub, _marker):
        del sys.modules[_name]

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from swap_cli import droidcam  # noqa: E402


@pytest.fixture(autouse=True)
def _real_media_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "numpy", np)


def _header(width: int, height: int, interval: int, *, control=droidcam.CONTROL,
            checksum=None) -> bytes:
    if checksum is None:
        checksum = interval ^ width ^ height
    return struct.pack("<7i", 1, control, checksum, width, height, interval, 0)


# --- header protocol ----------------------------------------------------------

def test_valid_header_yields_the_requested_mode() -> None:
    """The driver publishes the geometry the consuming app asked for."""
    interval = 333_333  # ~30 fps in 100ns units
    mode = droidcam.parse_header(_header(1280, 720, interval))
    assert mode is not None
    assert (mode.width, mode.height) == (1280, 720)
    assert round(mode.fps) == 30


def test_header_without_the_control_word_means_no_consumer() -> None:
    """No app holding the camera → nothing to send, not an error."""
    assert droidcam.parse_header(_header(1280, 720, 333_333, control=0)) is None


def test_bad_checksum_is_rejected() -> None:
    """Guards against a half-written or stale header being taken as live."""
    assert droidcam.parse_header(_header(1280, 720, 333_333, checksum=12345)) is None


def test_absurd_geometry_is_rejected() -> None:
    for w, h in ((0, 720), (1280, 0), (99999, 720), (1280, 99999)):
        assert droidcam.parse_header(_header(w, h, 333_333)) is None


def test_short_buffer_is_rejected() -> None:
    assert droidcam.parse_header(b"\x00" * 12) is None


def test_zero_interval_does_not_divide_by_zero() -> None:
    mode = droidcam.parse_header(_header(640, 480, 0))
    assert mode is not None
    assert mode.fps == 0.0


# --- YUYV conversion ----------------------------------------------------------

def test_yuyv_output_is_two_bytes_per_pixel() -> None:
    frame = np.zeros((48, 64, 3), np.uint8)
    assert len(droidcam.bgr_to_yuyv(frame)) == 64 * 48 * 2


def test_yuyv_packs_in_y0_u0_y1_v0_order() -> None:
    """Byte order is what the driver renders — get it wrong and colours swap."""
    frame = np.zeros((2, 2, 3), np.uint8)
    frame[:, :] = (255, 0, 0)  # pure blue in BGR
    raw = droidcam.bgr_to_yuyv(frame)
    y0, u0, y1, v0 = raw[0], raw[1], raw[2], raw[3]
    # Blue: low luma, high U (blue-difference), low V.
    assert y0 == y1  # flat colour → equal luma
    assert u0 > 150
    assert v0 < 120


def test_limited_range_keeps_luma_inside_broadcast_bounds() -> None:
    """Webcam YUYV is studio swing; full-range luma would look crushed."""
    white = np.full((2, 2, 3), 255, np.uint8)
    black = np.zeros((2, 2, 3), np.uint8)
    assert droidcam.bgr_to_yuyv(white)[0] <= 235
    assert droidcam.bgr_to_yuyv(black)[0] >= 16


def test_full_range_escape_hatch_widens_the_scale() -> None:
    white = np.full((2, 2, 3), 255, np.uint8)
    assert droidcam.bgr_to_yuyv(white, full_range=True)[0] > (
        droidcam.bgr_to_yuyv(white)[0]
    )


def test_odd_width_is_trimmed_to_a_whole_pair() -> None:
    """YUYV encodes pixels two at a time — an odd width would tear."""
    frame = np.zeros((4, 7, 3), np.uint8)
    assert len(droidcam.bgr_to_yuyv(frame)) == 6 * 4 * 2


def test_chroma_is_averaged_across_the_pair_not_point_sampled() -> None:
    """Averaging avoids the shimmer of dropping every other column."""
    frame = np.zeros((2, 2, 3), np.uint8)
    frame[:, 0] = (255, 0, 0)  # blue
    frame[:, 1] = (0, 0, 255)  # red
    raw = droidcam.bgr_to_yuyv(frame)
    u_avg = raw[1]
    # Neither the pure-blue U (~high) nor the pure-red U (~low), but between.
    blue_u = droidcam.bgr_to_yuyv(np.full((2, 2, 3), (255, 0, 0), np.uint8))[1]
    red_u = droidcam.bgr_to_yuyv(np.full((2, 2, 3), (0, 0, 255), np.uint8))[1]
    assert min(red_u, blue_u) < u_avg < max(red_u, blue_u)


# --- geometry -----------------------------------------------------------------

def test_fit_to_mode_resizes_to_what_the_driver_asked_for() -> None:
    mode = droidcam.VideoMode(width=640, height=360, interval=333_333)
    out = droidcam.fit_to_mode(np.zeros((720, 1280, 3), np.uint8), mode)
    assert out.shape[:2] == (360, 640)


def test_fit_to_mode_is_a_noop_when_sizes_match() -> None:
    mode = droidcam.VideoMode(width=64, height=48, interval=333_333)
    frame = np.zeros((48, 64, 3), np.uint8)
    assert droidcam.fit_to_mode(frame, mode) is frame  # no needless copy


# --- platform guard -----------------------------------------------------------

async def test_sink_refuses_to_open_off_windows() -> None:
    """A Linux/macOS user must get a clear message, not a ctypes traceback."""
    if sys.platform == "win32":  # pragma: no cover
        pytest.skip("this asserts the non-Windows guard")
    with pytest.raises(droidcam.DroidCamUnavailableError, match="Windows-only"):
        droidcam.DroidCamSink().open()


def test_driver_detection_is_false_off_windows() -> None:
    if sys.platform == "win32":  # pragma: no cover
        pytest.skip("registry probe only meaningful on Windows")
    assert droidcam.is_driver_installed() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
