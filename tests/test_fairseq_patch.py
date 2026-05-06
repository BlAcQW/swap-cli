"""Tests for the fairseq dataclass patch (Sprint 14g.1).

fairseq's archived `dataclass/configs.py` defines `FairseqConfig` with
11 mutable dataclass defaults that Python 3.11+ rejects with strict mode.
patch_fairseq_dataclass_defaults() fixes those in-place.

Pure Python — no fairseq install needed. We synthesize a fake
`fairseq.dataclass.configs` module on disk and run the patch on it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# Real snippet from upstream fairseq main, reproduced here for unit testing.
BROKEN_SAMPLE = """\
from dataclasses import dataclass, field

@dataclass
class CommonConfig:
    pass

@dataclass
class CommonEvalConfig:
    pass

@dataclass
class FairseqConfig:
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    name: str = "x"
"""


def _install_fake_fairseq(tmp_path) -> Path:
    """Create a minimal `fairseq.dataclass.configs` package on disk and
    register it on sys.path so importlib.find_spec can locate it.
    Returns the configs.py path."""
    pkg_root = tmp_path / "fake_site"
    fairseq_pkg = pkg_root / "fairseq"
    dataclass_pkg = fairseq_pkg / "dataclass"
    dataclass_pkg.mkdir(parents=True)
    (fairseq_pkg / "__init__.py").write_text("")
    (dataclass_pkg / "__init__.py").write_text("")
    configs_py = dataclass_pkg / "configs.py"
    configs_py.write_text(BROKEN_SAMPLE)
    sys.path.insert(0, str(pkg_root))
    # Bust the import cache so find_spec sees the new package.
    importlib.invalidate_caches()
    return configs_py


def _uninstall_fake_fairseq(pkg_root: Path) -> None:
    sys.path.remove(str(pkg_root))
    for mod in [m for m in sys.modules if m.startswith("fairseq")]:
        del sys.modules[mod]


def test_patch_rewrites_mutable_defaults(tmp_path) -> None:
    configs_py = _install_fake_fairseq(tmp_path)
    pkg_root = tmp_path / "fake_site"
    try:
        from swap_cli import voice_ops

        result = voice_ops.patch_fairseq_dataclass_defaults()
        assert result is True

        patched = configs_py.read_text()
        # Mutable defaults should be gone.
        assert "= CommonConfig()" not in patched
        assert "= CommonEvalConfig()" not in patched
        # Replaced with default_factory.
        assert "field(default_factory=CommonConfig)" in patched
        assert "field(default_factory=CommonEvalConfig)" in patched
        # Unrelated lines must be untouched.
        assert 'name: str = "x"' in patched
    finally:
        _uninstall_fake_fairseq(pkg_root)


def test_patch_is_idempotent(tmp_path) -> None:
    """Running the patch twice doesn't double-wrap or corrupt the file."""
    configs_py = _install_fake_fairseq(tmp_path)
    pkg_root = tmp_path / "fake_site"
    try:
        from swap_cli import voice_ops

        assert voice_ops.patch_fairseq_dataclass_defaults() is True
        first = configs_py.read_text()
        assert voice_ops.patch_fairseq_dataclass_defaults() is True
        second = configs_py.read_text()
        assert first == second
    finally:
        _uninstall_fake_fairseq(pkg_root)


def test_patch_no_op_when_fairseq_absent() -> None:
    """If fairseq isn't installed, the patch returns True (nothing to do)."""
    from swap_cli import voice_ops

    # Ensure no fake fairseq is shadowing.
    for mod in [m for m in sys.modules if m.startswith("fairseq")]:
        del sys.modules[mod]

    # In CI fairseq genuinely isn't installed, so this exercises the
    # "spec is None" early return.
    assert voice_ops.patch_fairseq_dataclass_defaults() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
