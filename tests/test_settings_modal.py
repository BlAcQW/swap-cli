"""Tests for the Settings panel helpers (Sprint 14l).

We test the pure-Python free functions, not the actual customtkinter
modal (which needs a real tk root and is too brittle in CI).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# gui.py imports customtkinter + PIL eagerly, and customtkinter pulls in
# a real tk root on import. Stub everything heavy so the free helpers
# inside gui.py are importable in CI.
def _ensure_stub(mod_name: str, attrs: dict | None = None) -> None:
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[mod_name] = mod


_ensure_stub("customtkinter", {
    "CTk": type("CTk", (object,), {}),
    "CTkToplevel": type("CTkToplevel", (object,), {}),
    "CTkFrame": MagicMock,
    "CTkLabel": MagicMock,
    "CTkButton": MagicMock,
    "CTkSwitch": MagicMock,
    "CTkEntry": MagicMock,
    "CTkOptionMenu": MagicMock,
    "CTkTextbox": MagicMock,
    "CTkFont": lambda *a, **kw: MagicMock(),
    "set_appearance_mode": lambda *a, **kw: None,
    "set_default_color_theme": lambda *a, **kw: None,
})
_ensure_stub("PIL", {})
_ensure_stub("PIL.Image", {"open": MagicMock(), "Resampling": MagicMock()})
import PIL  # noqa: E402
PIL.Image = sys.modules["PIL.Image"]  # type: ignore[attr-defined]
_ensure_stub("PIL.ImageTk", {"PhotoImage": MagicMock()})
_ensure_stub("cv2", {"VideoWriter_fourcc": MagicMock(), "VideoWriter": MagicMock()})
_ensure_stub("av", {"VideoFrame": MagicMock()})
_ensure_stub("aiortc", {"VideoStreamTrack": type("VideoStreamTrack", (object,), {})})
_ensure_stub("aiortc.mediastreams", {})
_ensure_stub("decart", {"DecartClient": MagicMock(), "models": MagicMock()})
_ensure_stub("decart.realtime", {"RealtimeClient": MagicMock(), "RealtimeConnectOptions": MagicMock()})
_ensure_stub("decart.types", {"ModelState": MagicMock(), "Prompt": MagicMock()})
_ensure_stub("pygrabber", {"dshow_graph": MagicMock()})
_ensure_stub("pygrabber.dshow_graph", {"FilterGraph": MagicMock()})


from swap_cli import gui  # noqa: E402


def test_redact_key_shape() -> None:
    """A typical key shows first-4 and last-4 with a middle ellipsis."""
    out = gui._redact_key("dct_abcdefghijklmnop")
    assert out.startswith("dct_")
    assert out.endswith("mnop")
    assert "…" in out


def test_redact_key_handles_none_and_short() -> None:
    assert gui._redact_key(None) == "—"
    assert gui._redact_key("") == "—"
    # 4-char or shorter values are returned unchanged (nothing to redact).
    assert gui._redact_key("dct") == "dct"


def test_apply_decart_key_update_persists(monkeypatch) -> None:
    """A valid key triggers config.update with the cache-reset fields."""
    captured: dict = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return MagicMock(decart_api_key=kwargs.get("decart_api_key"))

    monkeypatch.setattr(gui.config, "update", fake_update)

    gui.apply_decart_key_update("  dct_test_long_enough_to_pass_check  ")
    assert captured["decart_api_key"] == "dct_test_long_enough_to_pass_check"
    assert captured["license_cached_at"] is None
    assert captured["license_cached_valid_until"] is None


def test_apply_decart_key_update_rejects_bad_prefix(monkeypatch) -> None:
    """Missing 'dct_' prefix → ValueError + config.update NOT called."""
    called = []

    def fake_update(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(gui.config, "update", fake_update)
    with pytest.raises(gui.DecartKeyValidationError):
        gui.apply_decart_key_update("nope_definitely_not_a_dct_key_here")
    assert called == []


def test_apply_decart_key_update_rejects_short_key(monkeypatch) -> None:
    """Length < 20 → ValueError + config.update NOT called."""
    called = []
    monkeypatch.setattr(gui.config, "update", lambda **kw: called.append(kw))
    with pytest.raises(gui.DecartKeyValidationError):
        gui.apply_decart_key_update("dct_short")
    assert called == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
