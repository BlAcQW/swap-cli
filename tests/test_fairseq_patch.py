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


def _install_fake_fairseq(tmp_path, monkeypatch) -> Path:
    """Create a minimal `fairseq.dataclass.configs` package on disk and
    redirect site.getsitepackages + sysconfig.get_paths to point at our
    temp dir, so the patch finds it without importing fairseq.

    We avoid sys.path tricks because the production code is intentionally
    NOT using importlib.find_spec (which would import the broken parent
    package). Returns the configs.py path.
    """
    pkg_root = tmp_path / "fake_site"
    fairseq_pkg = pkg_root / "fairseq"
    dataclass_pkg = fairseq_pkg / "dataclass"
    dataclass_pkg.mkdir(parents=True)
    (fairseq_pkg / "__init__.py").write_text("")
    (dataclass_pkg / "__init__.py").write_text("")
    configs_py = dataclass_pkg / "configs.py"
    configs_py.write_text(BROKEN_SAMPLE)

    import site
    import sysconfig

    monkeypatch.setattr(site, "getsitepackages", lambda: [str(pkg_root)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(pkg_root))
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **kw: {"purelib": str(pkg_root), "platlib": str(pkg_root)}
    )
    return configs_py


def test_patch_rewrites_mutable_defaults(tmp_path, monkeypatch) -> None:
    configs_py = _install_fake_fairseq(tmp_path, monkeypatch)
    from swap_cli import voice_ops

    result = voice_ops.patch_fairseq_dataclass_defaults()
    assert result is True

    patched = configs_py.read_text()
    assert "= CommonConfig()" not in patched
    assert "= CommonEvalConfig()" not in patched
    assert "field(default_factory=CommonConfig)" in patched
    assert "field(default_factory=CommonEvalConfig)" in patched
    # Unrelated lines must be untouched.
    assert 'name: str = "x"' in patched


def test_patch_is_idempotent(tmp_path, monkeypatch) -> None:
    """Running the patch twice doesn't double-wrap or corrupt the file."""
    configs_py = _install_fake_fairseq(tmp_path, monkeypatch)
    from swap_cli import voice_ops

    assert voice_ops.patch_fairseq_dataclass_defaults() is True
    first = configs_py.read_text()
    assert voice_ops.patch_fairseq_dataclass_defaults() is True
    second = configs_py.read_text()
    assert first == second


def test_patch_no_op_when_fairseq_absent(tmp_path, monkeypatch) -> None:
    """If fairseq isn't found in any reachable site-packages, return True."""
    import site
    import sysconfig

    empty = tmp_path / "empty_site"
    empty.mkdir()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(empty)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(empty))
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **kw: {"purelib": str(empty), "platlib": str(empty)}
    )

    from swap_cli import voice_ops

    assert voice_ops.patch_fairseq_dataclass_defaults() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
